# 舆情负面精判 · 小样本探针实验（可一键跑通）

这是 SFT + GRPO 方案的**最小可执行闭环**：下载 → 混合 → 训练 → 推理 → 评测，全程落盘日志，训练指标同步 wandb。

> 本目录是仓库根（`/root/autodl-tmp/senti`）：默认路径 AutoDL 上指向数据盘（`/root/autodl-tmp/models` / `/root/autodl-tmp/datasets`），本机自动退回仓库内 `models/`、`data/`。
> 目标环境：AutoDL 单卡（RTX 4090 24GB / RTX 3090 24GB），Linux + Python 3.10+

---

## 0. 目录结构

```
<仓库根>/                       （AutoDL: /root/autodl-tmp/senti）
├── README.md                       ← 本文件
├── config/
│   └── accelerate_grpo.yaml        ← GRPO 双卡 accelerate 基线（DDP+bf16；仅 GRPO 用，SFT 走 NPROC；num_processes 占位，由 run_grpo_train.sh 覆盖）
├── docs/
│   ├── OPTIMIZATION_BACKLOG.md     ★ 落地改造清单：确定要做的改造 + 防回退清单
│   ├── AutoDL上常用指令.md
│   └── experiments/                ← 实验报告与实验待办
├── data/
│   ├── probe_test_sft_train_synth.jsonl ← SFT 探针训练数据 81 条（train_synth_81 经 align 重写 instruction，显式输出协议版；源文件已删，重跑 align 可再生成）
│   ├── probe_test_grpo_train_synth.jsonl    ← GRPO 探针数据 50 条：{prompt, gold, input_text, meta}
│   ├── gold500_test.jsonl           ★ 500 条冻结金标（seed500_final 经 convert_sft_data 转换，评测唯一成绩单）
│   └── round1_train_aligned.jsonl   ← round1 训练批次 1524 条（10 行业，convert_sft_data 转换）
│   # 运行后生成（不入库）：coig_cqia_anchor.jsonl（step 1）/ mixed/（step 2）/
│   # align_report.json（align）/ 对齐税基线集（从 prompts_probe 抽取，正式换 k=4 封存集）
├── lib/                            ← 共享层：被 scripts/ 与 unit/ 共同 import
│   ├── reward_fns.py               ★ GRPO 四维规则打分器（零外部依赖；枚举与 validate_schema 已下沉 schema_contract，本文件再导出保持兼容）
│   ├── schema_contract.py          ★ 输出协议唯一判定源：闭集枚举 + validate_schema + parse_output（训练/评测/质检三侧同源）
│   ├── eval_metrics.py             ★ 评估指标纯函数层：宏均 / Subset Accuracy / 负向召回×误报率成对 / 分桶 / 非法标签计数
│   ├── rubric_check.py             ★ 离线 Rubric 三判定（主体点名 / 推理无矛盾 / 证据支撑），可注入离线裁判
│   ├── release_gate.py             ★ 发布门四道纯逻辑（历史错题 / 金标不退化 / L3 安全 / 格式合规），N/A 不假装 PASS
│   ├── grpo_common.py              ★ GRPO 共享层：k 分桶 / 训练预算（零依赖）
│   ├── grpo_dataset.py             ★ GRPO 数据契约：JSONL 加载 / gold 校验 / 展平 TRL 标量训练行（零依赖，不做 Dataset 类）
│   ├── log_tee.py                  ★ SFT/GRPO 共用日志双写：终端 + 文件 / tqdm 换行 / 恢复与关闭
│   ├── grpo_probe_core.py          ★ GRPO 探针纯逻辑：打分入组 / rollout / k 分流
│   ├── corpus_schema.py            ★ 行业风险二级词表 + 提示词加载 + 闭集枚举（同源再导出 schema_contract）
│   ├── task_template.py            ★ 舆情精判任务指令/主体行协议（SFT/GRPO 共用；枚举从 schema_contract 插值）
│   ├── prompt_template.py          ★ 外壳渲染唯一真理源（apply_chat_template + 手写兜底 + parity gate，训练/推理/统计/GRPO 四方同源）
│   ├── sft_dataset.py              ★ SFT 专用 Dataset/Collator（前缀掩码 + FIELD_W 装配；import 时零 torch，无 GPU 可 preview/stats）
│   ├── format_canary.py            ★ 格式自检：协议 token id 解析 + teacher-forced 概率 + 贪心合规判定（训练回调与独立脚本共用）
│   ├── kappa_utils.py              ★ 一致性统计纯函数（cohen_kappa / risk_key，原 build_eval_set 内嵌实现下沉）
│   ├── llm_client.py               API 客户端（generate 的依赖）
│   └── sft_weight_planner.py       字段权重 span 规划器（train 的依赖，勿删）
├── unit/                           ← 离线单测（零外部依赖，任何机器都能跑）
│   ├── test_reward_fns.py          ★ 打分器单测 40 项
│   ├── test_weight_planner.py      ★ SFT 权重 span 规划器单测 12 项
│   ├── test_grpo_probe.py          ★ GRPO 链路单测 48 项
│   ├── test_log_tee.py             ★ SFT/GRPO 共用日志双写单测（恢复/追加/tqdm/描述符）
│   ├── test_eval_metrics.py        ★ 评估指标纯函数单测 25 项
│   ├── test_evaluate_sft.py        ★ 评测脚本纯逻辑单测 21 项（协议/指标/分桶/非法输出/证据分母/闭合率进 verdict）
│   ├── test_rubric_check.py        ★ 离线 Rubric 单测 20 项（三判定项 + 注入 judge）
│   ├── test_release_gate.py        ★ 发布门单测 25 项（四门 PASS/FAIL/N/A + 归因）
│   ├── test_prompt_template.py     ★ 外壳渲染单测（chat==手写 parity / thinking 版漂移拦截 / backend 降级）
│   └── test_sft_dataset.py         ★ SFT 专用 Dataset 单测（长度/掩码/权重三不变量 + parity gate）
├── prompts/                        ← gen/anno 双端提示词（单主体隔离锁在内）
├── scripts/                        ← 只放 CLI 入口（可执行脚本）
│   ├── run_sft_e2e.sh              ★ 一键串跑（5 步）
│   ├── run_grpo_train.sh           ★ GRPO 后台启动（nohup + 时间戳日志 + PID + 拓扑选择）
│   ├── run_grpo_probe.sh           ★ GRPO rollout 探针启动（与 run_grpo_train.sh 同产物布局）
│   ├── convert_sft_data.py         ★ Schema v1 批次转换（金标/训练批次 → 评测链路格式）
│   ├── human_recheck_kappa.py      ★ 人工复核 κ 一致性门禁
│   ├── business_verify.py          业务规则校验
│   ├── download_assets.py          1 下载：COIG-CQIA Anchor + 基座模型
│   ├── align_sft_data.py           1.5 对齐：原文+meta → {instruction(显式输出协议), input, output} + 质检（纯规则零 LLM，模板唯一源 lib/task_template.py）
│   ├── mix_sft_dataset.py          2 混合：9:1 + 分层切分 train/val
│   ├── train_sft_lora.py           3 训练：LoRA + 覆写 compute_loss（掩码/加权随 POC 阶段切换）
│   ├── sft_infer.py                4 推理：加载 Adapter 批量生成
│   ├── check_format.py             格式 canary 独立检查（不启动训练，测任意「基座+adapter」组合）
│   ├── evaluate_sft.py             5 评测：五段式（解析→指标→分桶→门禁→落盘 --out report.json）
│   ├── inspect_sft_dataset.py      数据模板预览（原文→messages→渲染文本→token/权重四层；--check_parity 只跑字节比对）
│   ├── stat_seq_len.py             序列长度统计（domain/anchor 分账 + Top-N 定位 OOM 长样本；--out 落统计 json / --log 追加日志，均可选；无 tokenizer 退化为字符级近似）
│   ├── pick_hard_cases.py          难例选样（按 hard_case_type 过滤打印行号+摘要，对比样张/四档对照选样用）
│   ├── eval_four_tier.py           ★ 四档控制变量对照（T1 基座/T2 提示词/T3 SFT/T4 GRPO → tier_report.json）
│   ├── release_gate.py             ★ 发布门 CLI（四道门独立判定 + 归因表落盘 gate.json）
│   ├── build_grpo_probe.py         ★ 语料池 → {prompt, gold} 契约转换
│   ├── run_grpo_probe.py           ★ 同题 G=4 采样 → 四维打分 → k∈{0..4} 分流
│   ├── merge_sft_adapter.py        ★ SFT adapter 合并（base + ΔSFT → 单一模型目录，GRPO 起点）
│   └── train_grpo.py               ★ TRL GRPOTrainer 训练主脚本（LoRA + 四维规则 Reward）
├── output/                         训练产物 / 预测文件 / 指标 JSON
└── logs/                           全流程日志（按时间戳）
```

> **三层分工**：`lib/` 是可复用的纯逻辑（被 import）、`scripts/` 是可执行的 CLI 入口
> （被 shell 调）、`unit/` 是离线单测。三层互不越界：CLI 脚本不再被当库 import，
> 共享逻辑一律下沉到 `lib/`。

---

## 1. 一键跑通（推荐）

```bash
cd /root/autodl-tmp/senti          # 本机则在仓库根
chmod +x scripts/run_sft_e2e.sh

# （可选）登录 wandb：仅当想在线看曲线时才需要，且要 SYNC_WANDB=1 才上传（见下表）
wandb login

./scripts/run_sft_e2e.sh          # 全流程：download → mix → train → infer → test
```

按步骤单独执行（模型已下好、想只重跑训练时很有用）：

```bash
./scripts/run_sft_e2e.sh download   # 步骤1
./scripts/run_sft_e2e.sh mix        # 步骤2
./scripts/run_sft_e2e.sh train      # 步骤3
./scripts/run_sft_e2e.sh test       # 步骤4+5（推理+评测）
./scripts/run_sft_e2e.sh traintest  # 步骤3+4+5（只训练+评测；复用已有混合数据与本地基座，不重复下载/混合）

# --tag=<实验标签>：输出目录名追加后缀，便于区分消融档（位置任意，不传保持原样）
#   例： run_20261024_180116 → run_20261024_180116_POC1_epoch6
./scripts/run_sft_e2e.sh --tag=POC1_epoch6 traintest
```

可用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SYNC_WANDB` | `0` | **wandb 开关**：`0`=探针默认不上传，只落本地日志；`=1` 且已登录（`WANDB_API_KEY` 或 `wandb login` 写入的 `~/.netrc`）才同步 wandb，项目 `sentiment-sft-probe` |
| `BG` | 未设 | `BG=1` 训练转后台：nohup + PID 文件 + 日志重定向，断 SSH 不死 |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | 换基座（对比 7B 消融时用；注意 ModelScope 无裸名 `Qwen3-4B-Instruct`，必须带 `-2507` 后缀） |
| `MODEL_DIR` | 自动 | 模型存放目录（AutoDL 默认 `/root/autodl-tmp/models`，本机退回仓库内 `models/`） |
| `DATA_DIR` | 自动 | 公开数据集与混合产物目录（AutoDL 默认 `/root/autodl-tmp/datasets`，本机退回仓库内 `data/`） |
| `EPOCHS` | `3` | 训练轮数 |
| `SFT_POC_STAGE` | `POC1` | **Loss 消融阶段**：`POC1`=仅前缀掩码 / `POC2`=+骨架掩码 / `POC3`=+字段加权 FIELD_W（详见 §训练 与 `sft_weight_planner.POC_STAGE`） |
| `TRAINER_BACKEND` | `auto` | 训练器后端：`auto`=有 trl 用 `trl.SFTTrainer` 否则 `transformers.Trainer` / `trl`=强制 trl（缺库报错）/ `hf`=强制 HF Trainer（同一份 compute_loss 覆写，加权口径一致） |
| `INFER_BACKEND` | `hf` | **推理后端**：`hf`=方案1 单卡 PyTorch 原生（默认，避免 PEFT 多卡切分设备不一致 bug）/ `vllm`=方案3 高并发批量推理引擎 |
| `INFER_GPU` | `0` | 方案 1 (hf) 绑定的单卡 GPU 编号（默认 0） |
| `UNFREEZE_PROTOCOL_TOKENS` | `1` | **协议 token 解冻**：给 `<think>` / `</think>` 两行 embedding 挂可训练偏移（约 10 KB，随 adapter 落盘）。设 `0` 关闭，用于消融对照。详见「步骤 3c」 |
| `THINK_PREFIX` | `0` | **前缀预置**：把 `<think>\n` 写进 assistant 前缀。默认关（已证实无额外收益，且会让 adapter 依赖推理端复现同一前缀）。设 `1` 开启 |
| `CANARY_EVERY` | `30` | 格式 canary 检测间隔（优化步）；`0` 关闭 |
| `CANARY_N` | `3` | canary 用几条验证样本 |
| `CANARY_MAX_NEW` | `400` | canary 贪心生成的 token 上限 |
| `NPROC` | 自动探测 | 卡数。显式指定可固定启动方式（`1`=python3 直启，`>1`=accelerate DDP），避免换机器后数值漂移 |
| `PER_DEV_BATCH` | `2` | 单卡 batch。有效 batch 恒为 8，梯度累积自动折算 |
| `MAX_LEN` | `2560` | 最大序列长度 |
| `FORCE=1` | `0` | 忽略已有产物强制重跑 |

```bash
# 例子：7B 基座对照 + 强制重下 + 训练 2 轮
MODEL_NAME=Qwen/Qwen3-4B-Instruct FORCE=1 EPOCHS=2 ./scripts/run_sft_e2e.sh

# 例子：跑 POC2 消融（只开骨架掩码、字段仍等权）
SFT_POC_STAGE=POC2 ./scripts/run_sft_e2e.sh train
```

### 快速消融对比：POC1 ~ POC3 运行指令集

三档共用同一份代码、只切一个环境变量，一次只动一个变量。

```bash
cd /root/autodl-tmp/senti          # 仓库根

# 离线干跑：无 GPU 验证权重分配
SFT_POC_STAGE=POC1 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
SFT_POC_STAGE=POC2 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
SFT_POC_STAGE=POC3 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl

# 真实探针训练
SFT_POC_STAGE=POC1 ./scripts/run_sft_e2e.sh train
SFT_POC_STAGE=POC2 ./scripts/run_sft_e2e.sh train
SFT_POC_STAGE=POC3 ./scripts/run_sft_e2e.sh train

# 全流程（训练 → 推理 → 评测）
SFT_POC_STAGE=POC1 ./scripts/run_sft_e2e.sh all
SFT_POC_STAGE=POC2 ./scripts/run_sft_e2e.sh all
SFT_POC_STAGE=POC3 ./scripts/run_sft_e2e.sh all
```

---

## 2. 手动分步执行（等价于上面的每一步）

### 步骤 1 · 下载公开数据集 + 基座模型

```bash
# 1a. 公开通用指令数据集 COIG-CQIA（防灾难性遗忘 Anchor，9:1 混入）
python3 scripts/download_assets.py --download_anchor --anchor_samples 100 --data_dir data
#   → data/coig_cqia_anchor.jsonl（已过滤任何带情感倾向的脏标）

# 1b. 基座模型（AutoDL 用 ModelScope 国内直连，不需梯子）
python3 scripts/download_assets.py --model Qwen/Qwen3-4B-Instruct-2507 \
    --model_dir /root/autodl-tmp/models --source modelscope
#   → /root/autodl-tmp/models/Qwen3-4B-Instruct-2507/
#   ⚠️ ModelScope 上没有裸名 Qwen/Qwen3-4B-Instruct（404），必须带 -2507 后缀
#   换 HuggingFace 源：--source huggingface
```

> 联网拉取 COIG-CQIA 失败时，脚本会**自动落盘内置的高质量通用问答兜底包**，不会卡住流程。

### 步骤 1.5 · 对齐器：原文+meta → SFT 训练格式

> 训练/探针用的 `data/probe_test_sft_train_synth.jsonl` 就是由本步产出；换新数据源时重跑即可。

```bash
python3 scripts/align_sft_data.py \
    --src data/train_synth_81.jsonl \        # 源: 原文+meta（或已带 output 的成品行，两种形态都吃）
    --out data/probe_test_sft_train_synth.jsonl \
    --report data/align_report.json          # 质检报告: total/kept/drop_reasons/cot_source
```

对齐器做什么（纯规则，零 LLM 调用）：

| 项 | 规则 |
|---|---|
| `instruction` | `lib/task_template.build_instruction(industry)` —— 角色+任务目标+行业风险体系+输出协议（枚举从 schema_contract 插值；**同行业逐字一致**） |
| `input` | user 侧题面 = `【目标主体】{subject}` + `【输入文本】{原文}`（subject 取 output gold，兜底 target_subject） |
| `output` | **CoT 只继承不生成**：源行自带 `<think>…</think>+JSON` 则逐字沿用；否则只拼裸 JSON。永不编造 CoT（模板化 CoT 会被死记并刷穿 GRPO rubric） |
| 质检（不过即剔除并计数） | schema 闭集 / evidence 是原文连续子串 4~80 字 / subject_hit（源行提供时）/ 行业越界 / 空或超长（>2000 字） |
| `meta` | 源字段全透传（provenance/hard_case_type/sentiment 供 mix 分层），补 `aligned_by` / `cot_source` 血缘 |

### 步骤 1.6 · Schema v1 批次转换（通用转换器 convert_sft_data.py）

> 适用任何 Schema v1 标注批次。转换只动三处：CoT 标签（`<|begin_of_thought|>`→`<think>`）、
> 补 instruction/题面（同训练模板）、meta 白名单瘦身（provenance dict 归一为 source 字符串，
> 完整审计信息挪 `provenance_detail`）；gold 六字段逐字继承（实测与 output 内嵌 JSON 零冲突）。

```bash
# 用例 A：500 条金标测试集（评测成绩单）
python3 -I scripts/convert_sft_data.py \
    --src ../../output/gold_eval_final/seed500_final.jsonl \
    --out data/gold500_test.jsonl \
    --report data/gold500_convert_report.json

# 用例 B：round1 训练批次（1524 条，10 行业）
python3 -I scripts/convert_sft_data.py \
    --src ../../output/sft_round1/round1_train.jsonl \
    --out data/round1_train_aligned.jsonl \
    --report data/round1_convert_report.json
```

### 步骤 2 · 领域数据 : 公开数据 = 9 : 1 混合

```bash
python3 scripts/mix_sft_dataset.py \
    --domain_file data/probe_test_sft_train_synth.jsonl \
    --anchor_file data/coig_cqia_anchor.jsonl \
    --out_dir data/mixed \
    --anchor_ratio 0.1 --val_ratio 0.1 --seed 42
```

产出（**分层切分**，按 `hard_case_type × sentiment` 保证 val 各类都有代表）：

```
领域数据 (Domain) : 81 条 (90.0%)
通用防遗忘 (Anchor): 9 条 (10.0%)
训练集 Train: 73 + 9 Anchor = 82 条  → data/mixed/sft_train.jsonl
验证集 Val  : 8 条                    → data/mixed/sft_val.jsonl
Val 分布: negative 4 / positive 3 / neutral 1（正负中齐备）
```

**关键约定**：Anchor 全量进 Train，**不进 Val**（防遗忘样本不参与舆情早停判据，两账分离）。

### 步骤 3 · LoRA SFT 探针训练（日志落盘 + wandb）

```bash
python3 scripts/train_sft_lora.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --train_file data/mixed/sft_train.jsonl \
    --val_file data/mixed/sft_val.jsonl \
    --output_dir /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --bf16 \
    --logging_steps 1 \
    --eval_steps 5 \
    --save_strategy no \
    --remove_unused_columns false \
    --log_file logs/train_probe.log \
    --wandb_project sentiment-sft-probe \
    --wandb_run_name probe-81seed
```

- **日志落盘**：`--log_file` 把终端输出**双写**到文件（`print` / transformers logging / tqdm 一并覆盖），中断也能回看。
- **wandb**：设了 `WANDB_API_KEY` 自动同步（超参、FIELD_W、数据血缘进 config，每步 loss 上报）；不想同步加 `--no_wandb`，未装 wandb 会自动降级不报错。
- **收尾产物**：`<OUTPUT_DIR>/run_<时间戳>/sft_lora/adapter_model.safetensors` + 同目录 `train_metrics.json`（含 `train_loss` / 耗时 / 样本数）。OUTPUT_DIR：AutoDL 默认 `/root/autodl-tmp/models/sft_senti`（本机退回 `output/`）；adapter 全路径 `cat <OUTPUT_DIR>/latest_adapter_path.txt` 直接拿。历史 run 目录里目录名是 `probe_lora`，新训的才是 `sft_lora`。
- **步数与停止口径**（启动时会打印「训练规模预算」区块）：

| | 单卡 batch | 梯度累积 | 实际(有效) batch | 训练样本 | 每 epoch 优化器步 | 总步数 |
|---|---|---|---|---|---|---|
| **探针 A · 未下 Anchor**（73 条 × 3 epoch，盘上现况） | 2 | 4 | **8** | 73 | 10 | **30** |
| **探针 B · 已下 Anchor**（**82** 条 × 3 epoch = 73 领域 train + 9 anchor；先跑 `download`+`mix` 才有 9:1 混入） | 2 | 4 | **8** | 82 | 11 | **33** |
| **正式**（2912 条 × 1 epoch，即 2880 领域×0.9 + 320 Anchor） | 4 | 2 | **8** | 2912 | 364 | **≈364** |

  > **探针口径提醒**：`data/coig_cqia_anchor.jsonl` 未下载时，`data/mixed/` 是 `probe_test_sft_train_synth.jsonl` 直切的 73/8（9:1 锚点尚未混入）；跑完步骤1+2 才是 81/9 的标准探针集。两档步数差 3 步，选哪个看前置条件。

  **不做早停**——SOP §训练纪律是「epoch 受控 + lr 1e-4（LoRA 口径）cosine 退火到 0」，靠 `eval_loss` 反弹**由人判定**过拟合，不让 Trainer 自动掐停。**探针的任务是冒烟验证**：按当前 `SFT_POC_STAGE` 对应的掩码/加权组合没写崩、loss 能压下去、adapter 能出盘，3 epoch（30~33 步）足够；它**不承担「定正式轮数」的职责**——正式轮数 SOP 已定死 1 Epoch。
  `--val_file` 会自动开 `eval_strategy=steps`，`eval_loss` 走同一套（同 POC 阶段）加权口径，训练结束会复测一次并存进 `train_metrics.json`。
- **无 GPU 时**先验证加权逻辑：`python3 scripts/train_sft_lora.py --dry_run --train_file data/probe_test_sft_train_synth.jsonl`
  会打印当前 POC 阶段、每个字段的 token 区间与权重。**各阶段该看到什么**：

  | 阶段 | 字段权重 | 骨架 | 有效 token 与 Σw |
  |---|---|---|---|
  | `POC1` | 全 `1.0x` | 不遮 | Σw **=** 有效 token 数（等权成立） |
  | `POC2` | 全 `1.0x` | 遮（`0.0x`） | 有效 token 变少、Σw = 新有效数；打印「骨架遮蔽(0×) 占输出」 |
  | `POC3` | `4.0/3.0/3.0/2.0/0.4` | 遮 | Σw **>** 有效 token 数；另打印高权/降权占比 |

  > `--dry_run` 用字符级伪编码（1 字符 = 1 token）做近似，**骨架遮蔽占比是字符级近似值、不是真实 token 占比**，真值需上真 tokenizer 才算数。
  >
  > **空表 `[]` 是例外**：`risk_categories: []` 的两个方括号单独登记 span、拿 `risk_categories` 的权重（POC2 `1.0` / POC3 `2.0`），不随骨架置 0——否则该字段在空表样本上无任何梯度，模型学不会「该空就空」。`subject` 回显值则相反：未登记 → 被骨架遮掉，属**主动不监督**。

### 步骤 3b · 格式 canary（训练期格式自检）
训练时定期在验证集观测格式学习情况("模型现在会不会输出规定格式")，上报日志/WandB，**不参与梯度更新**。
> loss 不能反映格式能力：存在 loss 接近、但格式合规率天差地别的情况，用于提前发现格式崩溃。

**检测项 [CANARY]**
| 指标 | 含义 |
|---|---|
| `p(</think>)` | 在该闭合的位置，模型对 `</think>` 的概率（teacher-forced，非生成） |
| `p(next\|after</think>)` | `</think>` 之后第一个 token 的概率（正确做法是读真实 token id，Qwen 把 `{"` 合成一个 token） |
| `p(<\|im_end\|>)` | 序列末尾对停笔 token 的概率 |
| `greedy_ok` | 贪心生成 M 条，格式全合规的条数 |

**参数（CLI，不读环境变量，代码：`lib/format_canary.py`）**
| 参数 | 默认 | 说明 |
|---|---|---|
| `--canary_every N` | 30 | 每N步检测，0=关闭 |
| `--canary_n M` | 3 | 验证样本数 |
| `--canary_max_new K` | 400 | 生成token上限 |

**开销&建议**：主要耗时来自贪心生成。正式训练：`--canary_every 60` 或关闭；**探针实验开启**，验证格式可学性。

**独立校验脚本**
```bash
python3 scripts/check_format.py \
    --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --adapter_path /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --data_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl
```

**判读**

| 观测 | 结论 |
|---|---|
| `p(</think>)` 上升 | 闭合在学，正常 |
| `greedy_ok` 里生成文本以 `<think>` 开头 | 起始标签也学会了 |
| `p(</think>)` 涨但开头不是 `<think>` | 起始标签没学会 |
| `p(</think>)` 恒 ≈0 | 协议 token 不可学，检查是否误加了 `UNFREEZE_PROTOCOL_TOKENS=0` |

### 步骤 3c · 协议 token 解冻（默认开启）


**功能**：仅给 `<think>` 和 `</think>` 两行 embedding 添加可训练增量参数 `(2, hidden)`（约10KB），随adapter保存；其余token embedding冻结。

**背景**：Qwen3-4B-Instruct-2507 为非思考型基座，词表虽包含这两个token，但从未训练：

| 事实 | 数值 |
|---|---|
| `<think>` / `</think>` 的行范数 | 0.367 / 0.370（普通词中位数 **1.106**） |
| 与 `</tool_call>` 的余弦相似度 | 0.9914 / 0.9187 —— **几乎是同一个方向** |
| 模型的 `p(</think>)` | 5.5e-4，排名 **112~149** |

开启权重绑定 `tie_word_embeddings=True` 时，它们同时是 `lm_head` 输出行；普通 LoRA 只更新 Attention / MLP，无法修改输出向量，模型会持续错输出 `</tool_call>`。

**POC 效果**

| 配置 | `p(</think>)` @60 步 | 贪心合规 | 格式门 |
|---|---|---|---|
| 解冻**关** | 0.001 | 0/3 | 0% |
| 解冻**开** | **0.829** | **3/3** | **100%** |


**使用方式**：默认开启。消融实验可关闭：

```
UNFREEZE_PROTOCOL_TOKENS=0 SFT_POC_STAGE=POC1 EPOCHS=6 ./scripts/run_sft_e2e.sh traintest
```

token id 通过 tokenizer（`format_canary.resolve_protocol_ids`）动态解析、无硬编码；基座词表缺失对应token会直接报错。

### 步骤 4 · 推理

```bash
# 方案 1（默认推荐）：单卡 PyTorch 原生推理（自动绑单卡，避开多卡张量不一致 bug）
python3 scripts/sft_infer.py \
    --backend hf \
    --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --adapter_path /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --data_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl \
    --out output/pred_val.jsonl

# 方案 3（大数据量 / 高并发）：vLLM 引擎（带 LoRA 批处理，需 pip install vllm）
python3 scripts/sft_infer.py \
    --backend vllm \
    --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --adapter_path /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --data_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl \
    --out output/pred_val.jsonl
```

实时输出三项合规率：**CoT 标签闭合率 / JSON 解析合法率 / evidence 原文命中率**。
`--adapter_path` 留空可跑**裸基座对照**，用来量化 SFT 带来的增益。
一键脚本中也可通过环境变量切换：`INFER_BACKEND=vllm ./scripts/run_sft_e2e.sh test`。

### 步骤 5 · 评测

```bash
# 模式 A：数据侧四查验收（确认喂进去的语料是干净的）
python3 scripts/evaluate_sft.py --test_file data/mixed/sft_train.jsonl

# 模式 B：模型对照评测（金标 vs 预测），输出逐难点/逐行业切片准确率
python3 scripts/evaluate_sft.py --gold_file data/mixed/sft_val.jsonl \
                                --pred_file output/pred_val.jsonl
```

> **基线（未微调）模型评测口径**：：基座输出常不符合`+JSON`规范、易截断，直接解析会语义分清零，因此失败时正则提取有效 JSON 字段计算 sentiment 等语义指标；但 JSON 合法率、标签闭合率严格判定，基线格式项 FAIL 为预期。

### 步骤 5b · 序列长度统计（OOM 排查 / 数据体检）

```bash
# AutoDL：真 token 口径 + Top-N 最长样本定位（domain / anchor 分账）
python3 scripts/stat_seq_len.py \
    --train_file data/mixed/sft_train.jsonl \
    --tokenizer models/Qwen3-4B-Instruct-2507 --top_n 5

# 可选产物（不给 = 只打印，零副作用）：
#   --out <json>  统计结果存档   --log <file>  日志追加写
python3 scripts/stat_seq_len.py \
    --train_file data/mixed/sft_train.jsonl \
    --tokenizer models/Qwen3-4B-Instruct-2507 \
    --out output/seq_len_stats.json --log logs/seq_len_stats.log

# 无 tokenizer 环境：字符级近似（1 汉字 ≈ 1.4 token）
python3 scripts/stat_seq_len.py --train_file data/probe_test_sft_train_synth.jsonl
```

输出 domain / anchor 两桶各自的 prompt / output / total 的 p50/p95/p99/max，
末尾自动打印归因速读（哪桶贡献长尾）。典型用途：
排查「大 vocab + 长 seq 的 CE 峰值 OOM」（长样本来源是领域数据还是锚点）、
确认训练数据是否逼近 `max_seq_length 2560`（超过即被截断）。

---

## 3. GRPO 探针（第二阶段：四维规则打分器）

> 真理源：`notes/舆情负面精判_GRPO数据准备实战手册.md` §0.2 落盘契约 + §五 四维 Reward 契约（※ 手册暂未随仓库提交，引用以实际入库位置为准）。
> 最终交付三样：`grpo_prompts_*.jsonl` + `reward_fns.py` + **零离线 completion**。

### 3.1 数据：题面 + 判卷标准，不带解题过程

`data/probe_test_grpo_train_synth.jsonl` — **50 条**（300 条构造池按 (hard_case_type × sentiment) 21 桶分层抽样，与 SFT 81 条 event_id / 原文零重叠，见 §3.3）

```json
{
  "prompt": "<|im_start|>system\n你是专业且客观严谨的舆情分析专家…【行业风险体系：汽车(automotive)】…【输出规范】…<|im_end|>\n<|im_start|>user\n【目标主体】极氪\n【输入文本】\n<待判原文><|im_end|>\n<|im_start|>assistant\n",
  "gold": {
    "subject": "极氪",
    "sentiment": "negative",
    "relevance": "core",
    "hard_case_type": "comparative",
    "risk_categories": ["product"],
    "evidence": "这次车机升级简直是灾难，黑屏、死机、智驾幽灵刹车全来了"
  },
  "input_text": "<待判原文，裸文本、不含题面壳>",
  "meta": {"industry": "automotive", "hard_case_type": "comparative",
           "sentiment": "negative", "relevance": "core", "provenance": "synthetic"}
}
```

- **不带 completion 是设计**：CoT 由策略 On-Policy 现场采样，算完组内优势即弃；攒着会被误拿去当 SFT 语料。
- **prompt 和SFT模板同源**：system 由 `lib/task_template.build_instruction(industry)` 生成，user = 目标主体+输入文本，统一用 `lib/prompt_template.render_prompt` 渲染；instruction 和训练样本 `probe_test_sft_train_synth.jsonl` 逐字对齐。
- **gold 6字段包含subject**：优先从输出JSON读取，兜底取prompt里的目标主体；供 `R_format` 做主体校验，rollout阶段无需解析prompt。
- **input_text 存原始文本**：给 `R_grounding` 做证据归属校验；包装后的prompt放在input，裸原文存 `meta.input_raw`。`--two_field` 模式可省略，由 `reward_fns.extract_input_text` 反向解析兜底。

**当前分布（50 条）**：难点 `explicit 13 · comparative 8 · fact_implicit 8 · irony 6 · metaphor 6 · counterfactual 5 · multi_subject 4`；极性 `neutral 20 / negative 17 / positive 13`；语篇 `core 33 · passing 14 · foil 3`；行业 `finance_insurance 14 · automotive 12 · fmcg 7 · real_estate 7 · electronics_3c 6 · beauty_cosmetics 4`。覆盖 `hard_case_type × sentiment` 全部 21 桶。

### 3.2 打分器：4 条纯 Python 规则

```bash
# 自检（零依赖，任何机器都能跑）
python3 lib/reward_fns.py

# 单测 40 项
python3 unit/test_reward_fns.py
```

| 函数 | 判据 | 值域 | 角色 |
|---|---|---:|---|
| `R_format` | `json.loads()` + 闭集枚举 + 平铺 + 尾赘 | `0 / -2.5` | 门禁，短路 |
| `R_grounding` | `evidence ⊂ input_text` 严格连续子串 | `0 / -2.5` | 门禁，短路 |
| `R_accuracy` | 危机漏报 / 误报 / 难点判错 / 全中；`relevance` 轻判 ±0.2 | `-2.0 ~ +1.0` | 判别主线 |
| `R_length` | 仅「判对+溯源」才激活；<40 罚 / 40~65 满 / 65~120 衰减 / ≥120 归 0 | `-1.0 ~ +1.0` | 卡空壳推理 |
| **总分** | | **`[-2.5, +2.0]`** | |

**单调性铁律**（单测强制校验）：

```
-2.5 (协议/证据)  <  -2.0 (危机漏报)  <  -1.0 (误报)  <  0 (判对但 CoT 空壳)  <  +1~+2
```

> 为什么必须这样排：若协议扣分只给 -1.0，就会**轻于**危机漏报的 -2.0 —— 模型拿不准时会**主动崩 JSON 止损骗分**，亲手开一条对齐欺骗通道。

三条门禁不变量（单测各一条）：
1. 门禁项不给正分（防「格式对 + 粘贴原文 + 判别全扔」白拿分）
2. 长度分只发给判对的（防交白卷套满分）
3. 短路罚 > 业务罚（-2.5 必须重于 -2.0）

组内聚合：`group_advantages(totals)` 组内相对归一（无 Critic），`zero_advantage_masking()` 对全错组整组优势清零。

### 3.3 语料 → GRPO 契约转换

探针数据来源：**300 条构造池**（与 81 条 SFT 训练集 event_id / 原文均零重叠），
先对齐到当前指令模板，再转成 GRPO `{prompt, gold}` 契约，最后按 21 桶分层抽样 50 条。

```bash
# ① 对齐（当前指令模板 → instruction / user 侧题面 / output）
python3 scripts/align_sft_data.py \
    --src ../../output/corpus/synth_300_stageA.dedup.jsonl \
    --out data/pool300_aligned.jsonl \
    --report data/pool300_align_report.json
# → 292 条入池（剔 8 条：subject_hit=false 6 / evidence 非子串 2）

# ② 转 GRPO 契约 + 分层抽样
python3 scripts/build_grpo_probe.py \
    --src data/pool300_aligned.jsonl \
    --out data/probe_test_grpo_train_synth.jsonl \
    --with_meta --sample 50
# → 50 条，覆盖 (hard_case_type × sentiment) 全部 21 桶
```

**两条硬约束**：

- **`prompt` 必须与 SFT 训练同模板**：走 `lib/task_template.py` 渲染，system=指令、user=`【目标主体】+【输入文本】`；手拼 ChatML 会导致 rollout 分布与 SFT 不匹配、reward 大面积掉底。
- **`input_text` 必须是裸原文**：`R_grounding` 拿它做 `evidence ⊂ 原文` 硬断言。对齐后 `input` 已包了题面壳，裸原文另存在 `meta.input_raw`，builder 优先取它。

`--sample N` 为按比例分层抽样（最大余数法）：每桶保底 1 条，余下按桶大小分配，避免 explicit×neutral 这类大桶吃光预算。

### 3.4 还缺什么

| 缺口 | 说明 |
|---|---|
| **`run_grpo_probe.py` / `train_grpo.py`（已写，未上 GPU 实测）** | 代码与单测（48 项）齐了，`--dry_run` 已在本地跑通；真实 rollout 与训练**必须等 SFT 探针训完 + AutoDL 环境** |
| 正式 `grpo_prompts_3000.jsonl` | 需生产日志（阶段一 A/B 双路筛题）+ 阶段二金标 + 阶段三 G=4 难度筛选，本地造不出来 |
| `irony` / `metaphor` 偏薄 | irony 6 条、metaphor 6 条，是 GRPO 高价值难例；扩样时优先加这两类（`--sample` 已是按比例分层，桶内条数受源池限制） |
| 阿里云百炼欠费 | 生成端 4/24 报 `Arrearage` HTTP 400，**充值前不要再跑生成/打标** |

> ⚠️ 这 50 条是**构造区探针数据**，与 500 条人标金标测试集是两本账，不得混着写进简历。

### 3.5 训练：`train_grpo.py`（TRL GRPOTrainer + 四维规则 Reward）

**数据侧不需要自定义 Dataset 类**（与 SFT 不同，备案）：SFT 要逐样本定制 `labels` / `loss_weights`（前缀掩码+字段加权），所以有 `lib/sft_dataset.py`；GRPO 的 rollout、组采样、优势计算全由 TRL `GRPOTrainer` 内部完成，数据侧只需把 `{prompt, gold…}` **展平成标量列**（`build_dataset_rows()`，dict 列在 datasets 版本间行为不稳），reward 函数经 `kwargs` 按列名取值。参考实现（gsm8k）同样是裸 `Dataset.from_list`。另外 `lib/task_template.py` / `prompt_template.py` 虽曾因历史文件名挂 SFT 名，实际是**任务协议与 prompt 外壳的两阶段共用层**（GRPO 探针构建和 rollout 同源），见 §0 目录注释。

**起点是一个完整模型目录，不是 adapter。** 先用 `merge_sft_adapter.py` 把 `base + ΔSFT` 合成一次：

```bash
# 0) 合并 SFT adapter（一次性操作；产出 /root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test）
python3 scripts/merge_sft_adapter.py --dry_run \
    --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --adapter /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --out /root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test

python3 scripts/merge_sft_adapter.py \
    --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --adapter /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --out /root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test
```

**为什么必须合并，而不是在训练脚本里 `load_adapter`**：TRL 算 KL 时走 `model.disable_adapter()`，它会关掉**全部** adapter。若 SFT adapter 与待训 GRPO LoRA 叠在同一个模型上，reference 会退化成裸基座 —— KL 惩罚变成「把策略往回忘掉 SFT」，β 越大坏得越快。合并以后 reference 天然就是 base+SFT，KL 才有意义。

合并脚本会做三件事：路径与基座一致性体检 → `merge_and_unload()` → **合并前后 logits 一致性校验**（argmax 必须相同、`max|Δlogit|` 在 bf16 舍入容差内，否则拒绝落盘）。产出 `merge_manifest.json` 作为血缘证据，`train_grpo.py` 会自动读取它并在 `grpo_metrics.json` 里记一笔。

> LoRA 增量可加，所以「训练时先合并」与「生产时 Base → SFT-LoRA → RL-LoRA 三段挂载」数学等价，部署形态不受影响。

```bash
# 1) 任何机器可跑的数据契约 + 预算体检（不 import torch / transformers / trl）
python3 scripts/train_grpo.py --data data/probe_test_grpo_train_synth.jsonl --dry_run

# 2) 真实训练（AutoDL，需 pip install trl）
./scripts/run_grpo_train.sh                       # 默认读 $MODEL（run_grpo_train.sh 默认 models/Qwen3-4B-Instruct-SFT）
GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh   # 先冒烟再放量

# 对照实验：从原厂基座起训（不要 SFT 底子）
MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-2507 ./scripts/run_grpo_train.sh
```

**干跑会打印这张超参表（每个数都能算出来源，不是拍的）**：

| 超参 | 值 | 来路 |
|---|---|---|
| 每步题数 | 8 = per_device 4 × grad_accum 2 | 与 SFT 有效 batch 8 对齐 |
| 生成组数 G | 4 | 组内要有方差，4 是最小偶数组 |
| **有效 batch** | **32 条轨迹/步 = 8 题 × G=4** | 手册 §1.3 定稿 |
| 采样温度 T | 0.7 | T=0 时 4 份采样相同 → std=0 → RL 退化成更贵的 SFT |
| 学习率 lr | 2e-5 | LoRA 口径：≈ SFT 1e-4 的 1/5。全参 RLHF 惯用的 1e-6~5e-6 不适用，低秩增量会挪不动 |
| KL 系数 β | 0.04 | 三闸①：常数 KL 管分布漂移，长度由 R_length 管，两者正交 |
| 输出预算 | **512** | 实测整段 max 383（CoT ~257 + JSON ~81），留 34% 余量（见下方陷阱） |
| 50 题 × epoch=1 | 7 步 | ceil(50/8)，正式 3000 题 × epoch=1 → 375 步 |

**三个抄了就崩的坑**（脚本里都有对应注释）：

1. **reward 不能拆成 4 个函数平权相加** —— 我们的门禁是**短路**语义（`R_format` 挂了，`R_accuracy`/`R_length` 归 0）。拆开相加会丢短路。所以只有一个 reward func，内部调 `compute_reward`。
2. **`max_completion_length` 不能按手册 65~105 截** —— 那一格说的是**扁平 JSON 段**（实测 ≈81 tokens），**不是整段输出**。SFT 73 条真分词：整段 p50 263 / p95 335 / max 383，CoT 占大头。按 105 或 200 截，**99% 的样本被砍在中间** → `json.loads` 失败 → 全池 `R_format` −2.5 → 训练废掉。
3. **优势只减组内均值、不除 std（`scale_rewards="none"`）** —— 别和 `loss_type` 搞混：`loss_type` 管 loss 分母，`scale_rewards` 管优势要不要标准化，两个独立开关。TRL 默认 `scale_rewards="group"` 会除 std，把「只差一档」的垃圾组放大成满格优势（`[-2.5,-2.5,-2.5,-2.0]` → +1.73），反而压过真有判别信息的组（`[+2,+1,-1,-2]` → +1.26），等于教模型「学垃圾里最不烂的那条」。我们的 reward 是固定尺度的离散阶梯，分数本身已校准，再标准化只会抹掉校准（推导见 `notes/舆情负面精判_GRPO训练实战手册.md` §1.2）。


**可选 eval reward 熔断**：训练脚本传 `--eval_file` 才接入熔断；训练开始前先用当前策略算 baseline，之后每 `--eval_steps` 步（默认 2）对 eval 集采样打分，与 baseline 比较。跌幅超过 `--fuse_threshold`（默认 2%）时只告警；加 `--fuse_hard` 才请求 Trainer 停训。`scripts/run_grpo_train.sh` 透传 `EVAL_FILE` / `EVAL_STEPS` / `FUSE_THRESHOLD` / `FUSE_HARD` 环境变量。


**探针熔断用途限制**：快速启动清单用的 8 条 eval 是从同一探针池抽取，会和训练数据重叠；它适合验证“训练前测 baseline → 周期 eval → 告警/停止”的功能链路，**不是独立泛化集，也不能据此声称测到了真实通用能力对齐税或防止了 reward collapse**。熔断只负责发现 eval reward 下跌并可选刹车，不阻止下跌；防止策略退化还依赖 KL、奖励契约、采样配置和数据筛选。

另有两处防御（启动时自动处理）：
- **prompt 上限体检**：实测 prompt 最长 token 数（含行业风险词表）若超 `max_prompt_length`，**自动放宽**——截断会砍掉待判原文 → `evidence` 必然溯源失败 → 全池短路。
- **Zero-Advantage Masking 逻辑检查**：同分组减均值后必为 0，所以 `[-2.5]×4` 没有梯度。`assert_zam_equivalence()` 只检查我方优势公式和「不除 std」口径，**不是对 TRL 源码实测**；TRL 是否采用预期配置，要看 `GRPOConfig(scale_rewards="none")` 字段在目标版本中实际存在并生效。检查失败会抛断言异常，训练拒启；**不会 hack TRL 私有 API**。



**版本要求**（2026-10-28 真机锚定，以 `requirements.txt` 为准）：`trl==0.14.0` + `transformers==4.51.0`（Qwen3 支持下限，勿装 5.x——trl 0.14 不兼容）+ `torch==2.5.1` + `peft==0.20.0`；**deepspeed 必须卸载**（与 transformers 4.51 循环导入）。安装一律 `pip install -r requirements.txt`，**不要自行升级 `trl` / `vllm` / `torch`**。`train_grpo.py` 内置 5 个环境补丁（vllm 检测元组 / scale_rewards=none 运行时 patch / vLLM max_model_len 注入 / 显存预算），换版本前先过 `notes/GRPO环境踩坑实录_20261027.md`。
**双卡 vLLM 采样（✅ 2026-10-28 已跑通，run_20261028_115727_smoke）**：`NPROC=2`（或机器 ≥2 卡自动）走「accelerate 1 训练进程 cuda:0 + vLLM 独占 cuda:1」。显存三件套已固化默认值：batch `2×4`（每步拓扑不变 8 题×G4）、`--gradient_checkpointing`、vLLM utilization `0.9`（KV 12.3G / 并发 ~35x，32 条 rollout 一波发完）；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 由 `run_grpo_train.sh` 自动 export。vllm 0.7.3 不在 requirements，需手动装（见文件头注释）。单卡环境自动降级 HF generate + `CUDA_VISIBLE_DEVICES=0`（DataParallel 防呆保留）。


### 3.6 难度分流：`run_grpo_probe.py`（只留 k∈{1,2,3}）

```bash
# 干跑：用 gold 反构「标准答案」当采样结果，验数据 + 打分器 + 分桶全链路
python3 scripts/run_grpo_probe.py --data data/probe_test_grpo_train_synth.jsonl --dry_run

# 真实 rollout（SFT 探针训完后；裸基座 + adapter 分体挂载。k 分布是「合并 SFT 模型」口径的话，
# 请先跑 §3.5 的 merge，再 BASE_MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-probe-test 且不传 --adapter）
python3 scripts/run_grpo_probe.py \
    --data data/probe_test_grpo_train_synth.jsonl \
    --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --adapter /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
    --g 4 --temperature 0.7 --max_new_tokens 512 \
    --out output/grpo/grpo_probe_kstats.json
```

**k = 同题 G 份采样里「判对 且 evidence 溯源过门禁」的份数**：

| k | 处置 | 为什么 |
|---|---|---|
| 0（全错） | 丢弃 | 组内 std=0，梯度为 0；RL 无米下锅，退回 SFT 补课 |
| **1~3** | **训练池** | 组内有正负样本可比，这是唯一有信息量的档 |
| 4（全对） | 丢弃 | 方差恒 0，花 RL 的钱跑了个更贵的 SFT |

产出：`kstats.json`（k 直方图 + 训练池规模 + 逐难点/极性分布）+ `kstats.rows.jsonl`（逐题明细）。**这就是正式池三阶段筛题的第 3 步**——探针阶段用它验证打分器与 rollout 链路，正式阶段用它把 3000 题筛成真正可训练的子集。

### 3.7 金标评测集（已归档 `build_eval_set.py`，κ ≥ 0.7 门禁由 `human_recheck_kappa.py` 承接）

原「三方异族一致性 → 金标评测集」脚本已归档至 `scripts/archive/build_eval_set.py`
（LLM 裁决不再需要；κ 计算下沉为 `lib/kappa_utils.py` 单一判定源）。
金标构建改为：标注流水线产出「原文+meta」→ `align_sft_data.py` 纯规则对齐 + 质检 →
人工复核走 `human_recheck_kappa.py`（κ ≥ 0.74 放行门槛不变）。

> **历史背景（已归档流程，供回溯）**：原脚本对同一条原文做三次**异族**打标，按分歧分三档：
> `gold`（全字段一致→金标）/ `boundary`（仅 hard_case_type 分歧→留 DPO）/ `drop`（sentiment/
> relevance/risk 分歧→丢弃）。**异族铁律**（生成端不参与裁决）仍然适用于任何 LLM 打标环节。
> ⚠️ **κ < 0.6 说明是标注法典本身有歧义**（不是模型差异），必须回炉重写标注规范，
> 不得靠「再跑几遍」蒙混过关。

---

## 4. 跑通后你会拿到什么

| 产物 | 位置 | 用途 |
|---|---|---|
| LoRA Adapter | `/root/autodl-tmp/models/sft_senti/run_*/sft_lora/`（本机 `output/run_*/`） | 下一步 GRPO 的起点 |
| 训练日志 | `logs/train_probe_*.log` | 断点回看、面试演示 |
| 训练指标 | `<OUTPUT_DIR>/run_*/sft_lora/train_metrics.json` | 多轮探针对比 |
| wandb 曲线 | `sentiment-sft-probe` 项目 | loss 收敛可视化 |
| 预测文件 | `output/pred_*.jsonl` | 喂给 `evaluate_sft` 模式 B |
| 评测报告 | 终端 + `logs/probe_e2e_*.log` | 难点/行业分桶准确率 |
| GRPO 打分器 | `lib/reward_fns.py`（40 项单测全绿） | 4 条规则 + 单调性 + 门禁不变量 |
| GRPO 探针数据 | `data/probe_test_grpo_train_synth.jsonl` | `{prompt, gold, input_text, meta}` 50 条（21 桶全覆盖），等 SFT 模型可 rollout |
| GRPO 难度分流 | `output/grpo/grpo_probe_kstats.json` + `.rows.jsonl` | k 直方图 + 训练池规模 + 逐题明细 |
| GRPO 训练日志 | `logs/grpo_train_*.log`（nohup 后台，时间戳命名） | 断点回看 |
| GRPO Adapter | `output/grpo_lora/` + `grpo_metrics.json` | SFT→GRPO 两段式终点 |
| GRPO 单测 | `unit/test_grpo_probe.py`（48 项） | 数据契约 / k 分桶 / reward 列对齐 / 版本兼容 |

**全量自检一条命令**（零外部依赖，任何机器都能跑）：

```bash
cd /root/autodl-tmp/senti          # 仓库根
python3 -m unittest discover -s unit -p "test_*.py"   # 342 项，当前全绿
```

### 5.1 评测与发布门（评估手册缺口 1~5、11~18 已落地）

```bash
# 模式 A：数据集自检（分母=总样本数；闭合率纳入 verdict）
python3 scripts/evaluate_sft.py --test_file data/gold500_test.jsonl --out output/report.json

# 模式 B：模型对照评测（四维分桶 + 非法标签计数 + 负向召回×误报率成对 + 发布门）
python3 scripts/evaluate_sft.py --gold_file data/gold500_test.jsonl \
                                --pred_file output/pred_val.jsonl \
                                --out output/report.json

# 发布门 CLI（四道门独立判定；缺输入的门返回 N/A，不假装 PASS）
python3 scripts/release_gate.py --gold data/gold500_test.jsonl \
                                --pred output/pred_val.jsonl \
                                --old_error_list output/old_errors.txt \
                                --baseline_json output/report_prev.json \
                                --out output/gate.json

# 四档控制变量对照（T1 基座 / T2 提示词加固 / T3 SFT / T4 GRPO；缺档自动 N/A）
python3 scripts/eval_four_tier.py --gold data/gold500_test.jsonl \
    --pred_base output/pred_t1.jsonl --pred_prompt output/pred_t2.jsonl \
    --pred_sft output/pred_t3.jsonl --pred_grpo output/pred_t4.jsonl \
    --out output/tier_report.json
```

---

## 5. 三个必须知道的口径（防穿帮）

1. **`sft_val.jsonl`（8 条）不是成绩单**——它只是**观察 `eval_loss` 是否反弹**（人判过拟合，不做自动早停）与各 POC 阶段对照的仪表盘。
   简历上能写的准确率，**只能来自 500 条纯人标金标测试集**（`plan/SFT负面精判_技术评审.md` 锁死的死数字）。
2. **这 81 条是构造区语料**（`provenance=synthetic`），与真实业务流量、BERT 时代的近 4 万条人标数据是**两本账**，严禁混着写。
3. **分桶看结论，别看整体 F1**——决定「要不要从 4B 升到 7B」的唯一证据，是 `irony / metaphor / counterfactual` 这几个难例桶在 500 金标上的表现，不是总分。
4. **GRPO 的 50 条探针 ≠ 训练池**——正式池是 `grpo_prompts_3000.jsonl`，必须经「生产日志筛题 → 金标 → G=4 难度筛选只留 k∈{1,2,3}」三阶段；探针这 50 条只用来**验证打分器与 rollout 链路通不通**。
5. **GRPO 不切 dev 集**——RL 的失效模式是通用能力退化和 Reward Hacking，dev 集测不出这两样；等价监控是 `k=4` 基线集盯通用能力掉没掉、`k=0` 集盯难题攻克进度，500 条金标只评不训。
6. **`run_grpo_probe.py --dry_run` 的 k=4 不是模型成绩**——干跑用 gold 反构标准答案当采样结果，只证明「数据 + 打分器 + 分桶」链路通。真实 k 分布必须等 SFT 探针训完再 rollout 才有数。同理，`train_grpo.py --dry_run` 打印的 32 轨迹/步、7 步是**预算**不是实测。

---

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `缺少依赖: torch transformers peft` | `pip install torch transformers peft accelerate wandb` |
| 推理报 `Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!` | 多卡环境下 PEFT `trainable_tokens` 跨卡切分 bug。切回默认方案 1 单卡跑：`INFER_BACKEND=hf INFER_GPU=0`（或直跑前加 `CUDA_VISIBLE_DEVICES=0`） |
| GRPO 报 `No module named 'trl'` | `pip install -r requirements.txt`（trl==0.14.0 + transformers==5.15.0 真机锚定组合，**别手装升级版**；踩坑见 `notes/GRPO环境踩坑实录_20261027.md`） |
| GRPO 启动就报「batch 不能被 G 整除」 | `--per_device_train_batch_size` 必须是 G 的倍数（默认 4，G=4 合法） |
| GRPO 日志提示「无 temperature 字段」 | TRL 版本与 `requirements.txt` 锚定组合不符；TRL 太老时 T 若为 0，组内方差恒 0，RL 会退化成更贵的 SFT |
| GRPO 慢 / rollout 占时间 | 双卡走 `NPROC=2` 自动开 vLLM（✅ 已跑通，utilization 0.9 / 并发 ~35x）；单卡自动降级 HF generate |
| GRPO OOM（24G 卡） | 默认已固化 batch 2×4 + `--gradient_checkpointing` + `expandable_segments:True`；仍 OOM 再降 `--per_device_train_batch_size 1 --gradient_accumulation_steps 8` |
| COIG-CQIA 拉取超时 | 脚本已自动落盘内置兜底包；或 `pip install modelscope` 后重试 |
| wandb 提示未登录 | `wandb login`，或加 `--no_wandb` 关掉 |
| OOM | 降 `--per_device_train_batch_size 1 --gradient_accumulation_steps 8` |
| 想换 7B 对照 | `MODEL_NAME=Qwen/Qwen3-4B-Instruct ./scripts/run_sft_e2e.sh all` |
| `loss_weights` 被 Trainer 丢掉 | 训练命令里必须保留 `--remove_unused_columns false` |
