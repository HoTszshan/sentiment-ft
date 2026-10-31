# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

舆情负面精判（sentiment classification）的小样本探针实验：SFT（LoRA）+ GRPO 两段式，全流程 下载 → 混合 → 训练 → 推理 → 评测，目标环境 AutoDL 单卡 24GB（RTX 4090/3090）。基座默认 `Qwen/Qwen3-4B-Instruct-2507`（ModelScope 无裸名 `Qwen3-4B-Instruct`，必须带 `-2507` 后缀）。项目文档与注释以中文为主。

## 常用命令

```bash
# 全量离线单测（零外部依赖，任何机器；~342 项）
python3 -m unittest discover -s unit -p "test_*.py"
# 单个文件
python3 unit/test_reward_fns.py

# SFT 一键流程（download → mix → train → merge → infer → eval）
./scripts/run_sft_e2e.sh              # 全流程
./scripts/run_sft_e2e.sh download|mix|train|merge|test|traintest   # 分步
./scripts/run_sft_e2e.sh --tag=POC1_epoch6 traintest               # 输出目录加 tag

# GRPO
DRY=1 ./scripts/run_grpo_train.sh     # --dry_run 体检（无 GPU 可跑）
./scripts/run_grpo_train.sh           # nohup 后台启动（NPROC=2 自动开 vLLM）
./scripts/run_grpo_probe.sh           # rollout 探针

# 评测 / 发布门 / 四档对照
python3 scripts/evaluate_sft.py --gold_file data/gold500_test.jsonl --pred_file output/pred_val.jsonl --out output/report.json
python3 scripts/release_gate.py --gold data/gold500_test.jsonl --pred output/pred_val.jsonl --out output/gate.json
python3 scripts/eval_four_tier.py --gold data/gold500_test.jsonl --pred_sft ... --out output/tier_report.json

# 无 GPU 干跑验证
SFT_POC_STAGE=POC1 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
python3 lib/reward_fns.py             # 打分器自检
```

依赖：`pip install -r requirements.txt`。**版本全部 `==` 锚定（trl==0.14.0 + transformers==4.51.0 真机验证组合），禁止自行升级 trl/vllm/torch**。

常用环境变量：`SYNC_WANDB=1`（默认 0 不上传 wandb）、`BG=1`（nohup 后台 + PID 文件）、`SFT_POC_STAGE=POC1|POC2|POC3`（loss 消融：前缀掩码 / +骨架掩码 / +字段加权）、`MODEL_DIR`/`DATA_DIR`（AutoDL 默认 `/root/autodl-tmp/models`、`/root/autodl-tmp/datasets`，本机退回仓库内）、`INFER_BACKEND=hf`（避免 PEFT 多卡切分 bug）、`FORCE=1`。

## 架构

三层分工，互不越界：

- **`lib/`** — 可复用纯逻辑，被 scripts/ 与 unit/ 共同 import。共享逻辑一律下沉到此层，CLI 脚本不再被当库用。关键模块：
  - `schema_contract.py` — 输出协议唯一判定源（闭集枚举 + `validate_schema` + `parse_output`），训练/评测/质检三侧同源；`corpus_schema.py`、`reward_fns.py` 从这里再导出保持兼容。
  - `prompt_template.py` — prompt 外壳渲染唯一真理源（apply_chat_template + 手写兜底 + **parity gate**），训练/推理/统计/GRPO 四方同源；`task_template.py` 提供任务指令（SFT/GRPO 共用，同行业逐字一致）。
  - `reward_fns.py` — GRPO 四维规则打分器 `R_format / R_grounding / R_accuracy / R_length`，总分 `[-2.5, +2.0]`。
  - `sft_dataset.py` — SFT Dataset/Collator（前缀掩码 + FIELD_W 加权，import 时零 torch）。
- **`scripts/`** — 只放 CLI 入口（可执行脚本 + `.sh` 启动器）。`run_sft_e2e.sh` / `run_grpo_train.sh` 是入口，脚本头注释有完整参数说明。
- **`unit/`** — 离线单测，零外部依赖。

数据流：`align_sft_data.py`/`convert_sft_data.py`（Schema v1 → 训练格式，纯规则零 LLM，见 `README_数据格式转换.md`）→ `mix_sft_dataset.py`（9:1 anchor 混合）→ `train_sft_lora.py` → `merge_sft_adapter.py`（GRPO 起点）→ `train_grpo.py`（TRL GRPOTrainer + LoRA）。GRPO 数据契约：`{prompt, gold, input_text, meta}`，**不带 completion（设计如此）**——CoT 由策略 on-policy 采样，攒离线 completion 会被误当 SFT 语料。

## 硬性纪律（README「防穿帮」口径）

1. **单测强制校验 reward 单调性铁律**：`-2.5 (协议/证据) < -2.0 (危机漏报) < -1.0 (误报) < 0 (判对但空壳) < +1~+2`。若协议扣分轻于漏报，模型会主动崩 JSON 止损骗分（对齐欺骗通道）。
2. **成绩单只有 `data/gold500_test.jsonl`（500 条纯人标金标）**。`sft_val.jsonl`（8 条）只用来观察 eval_loss 反弹；81 条 SFT / 50 条 GRPO 探针是 `provenance=synthetic` 构造区，与真实业务流量是两本账，严禁混写。
3. **分桶看结论，别看整体 F1**——升级基座的证据在 `irony / metaphor / counterfactual` 难例桶表现。
4. **GRPO 不切 dev 集**（RL 失效模式是通用退化与 Reward Hacking，dev 测不出）；监控靠 k=4 基线集 / k=0 难题集，500 金标只评不训。
5. **`--dry_run` 的数字是链路验证，不是模型成绩**（dry-run 用 gold 反构采样结果）。
6. 实验数字纪律：跑出来的数才回填，没跑的一律标 `[待实测]`，不编数（见 `TODO_*探针快速启动.md`、`docs/experiments/`）。

## 其他

- 关键文档：`docs/OPTIMIZATION_BACKLOG.md`（改造+防回退清单）、`docs/experiments/`（实验报告）、`notes/`（训练/评估/GRPO 手册，口径权威源）。
- 产物布局：一个 run 目录 = 一次完整实验（`output/run_<TS>[_<TAG>]/`），拷走即归档；日志按时间戳落 `logs/`。
- GRPO 双卡拓扑：训练进程数 = GPU 数 − 1，剩余卡给 colocate vLLM 独占采样；**不要在脚本里重排 `CUDA_VISIBLE_DEVICES`**（torch 初始化后改它不生效）。
- 评测脚本读外部下载文件时用 `python3 -I`（见 README_数据格式转换.md 的用法）。
