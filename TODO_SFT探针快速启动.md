# SFT 探针实验 · 快速启动 TODO 清单

> **定位**：AutoDL 单卡（RTX 4090/3090 24GB）上把 SFT 探针（POC1→POC2→POC3 消融）从零跑通并回填实测数字的**一步步操作清单**。
> **口径权威源**：`notes/舆情负面精判_SFT训练实战手册.md`（训练）、`notes/舆情负面精判_评估手册_SFT_GRPO.md`（评估）、`tests/README.md`（脚本用法）。冲突以上游为准。
> **纪律**：跑出来的数才回填；没跑的一律标 `[待实测]`，不编数。本清单勾完即把训练手册 §1.4.6 的「从未实跑」红线解除。

---

## Step 0 · AutoDL 环境准备（单环境，SFT/GRPO 共用）

> **不建多个虚拟环境**：SFT 与 GRPO 依赖栈兼容（GRPO 只比 SFT 多 `trl`），统一装一份即可。

- [x] **0.1** 租卡：AutoDL 单卡 RTX 4090 24GB（镜像选 PyTorch 2.x + CUDA 12.x + Python 3.10+）
- [ ] **0.2** 进入实例后**只装一次依赖**（单环境，含 GRPO 需求，一步到位）：
  ```bash
  pip install "torch>=2.1" "transformers>=4.55" "trl>=0.19" "peft>=0.12" \
      accelerate datasets modelscope wandb
  ```
  验证版本：`python3 -c "import transformers, trl, peft; print(transformers.__version__, trl.__version__, peft.__version__)"`
  （Qwen3 要 transformers≥4.51；trl≥0.19 才认 `temperature`/`processing_class`）
- [ ] **0.3**（可选但推荐）`wandb login` 登录；不登录则自动降级只写本地日志。
  > **开关：`SYNC_WANDB=1` 才上传**。探针实验默认只落本地日志（`logs/train_probe*.log`），不污染 wandb 项目；正式跑或想看曲线时在命令前加 `SYNC_WANDB=1`（例：`SYNC_WANDB=1 SFT_POC_STAGE=POC1 ./scripts/run_probe_e2e.sh train`）。登录态认两条路径：`WANDB_API_KEY` 环境变量 或 `~/.netrc` 里 `api.wandb.ai` 凭据（`wandb login` 写入的即是后者）。上传目标项目固定为 `sentiment-sft-probe`，run 名 `probe_<时间戳>`。
- [ ] **0.4** 上传 `tests/` 目录（或 git clone 仓库），`cd tests && chmod +x scripts/*.sh`
- [ ] **0.5** 零依赖自检：`python3 -m unittest discover -s unit -p "test_*.py"` —— 全绿才继续

> **路径约定（AutoDL）**：模型下到 `/root/autodl-tmp/models`、公开数据集下到 `/root/autodl-tmp/datasets`（数据盘，容量大）。脚本已适配：
> - `download_assets.py` 的 `--model_dir` / `--data_dir` 默认值即上述两个目录；
> - `run_probe_e2e.sh` 支持环境变量 `MODEL_DIR` / `DATA_DIR`（默认指向 AutoDL 数据盘），全流程自动衔接；
> - 领域训练数据 `train_synth_81.jsonl` 是仓库内置文件，留在 `tests/data/` 不外移；混合产物 `data/mixed/` 由 `DATA_DIR/mixed` 派生。

## Step 1 · 下载模型与公开数据集

- [ ] **1.1** 基座模型（ModelScope 国内直连，不需梯子；**注意仓库名是 `-2507` 后缀**）：
  ```bash
  python3 scripts/download_assets.py --model Qwen/Qwen3-4B-Instruct-2507 \
      --model_dir /root/autodl-tmp/models --source modelscope
  # → /root/autodl-tmp/models/Qwen3-4B-Instruct-2507/（约 8GB，检查 config.json + *.safetensors 齐全）
  ```
  > ⚠️ ModelScope 上没有裸名 `Qwen/Qwen3-4B-Instruct`（404 record not found）；`-2507` 是官方重训的非思考指令版，与本项目「CoT 由 SFT 注入、不用基座原生思考」的选型一致。
- [ ] **1.2** 公开 Anchor 数据集（COIG-CQIA，防遗忘锚点）：
  ```bash
  python3 scripts/download_assets.py --download_anchor --anchor_samples 100 \
      --data_dir /root/autodl-tmp/datasets
  # → /root/autodl-tmp/datasets/coig_cqia_anchor.jsonl（已过滤带情感倾向的脏标）
  ```
  > 联网拉取失败会自动落盘内置兜底包，不卡流程；但优先确认真包下载成功：
  > `wc -l /root/autodl-tmp/datasets/coig_cqia_anchor.jsonl` 且抽查第 1 条非重复模板（兜底包只有 4 条循环采样）。

## Step 2 · 数据集验证（先验数据，再训练）

- [ ] **2.1** 9:1 混合 + 分层切分（新增统计日志与四查落盘）：
  ```bash
  python3 scripts/mix_sft_dataset.py \
      --domain_file data/train_synth_81.jsonl \
      --anchor_file /root/autodl-tmp/datasets/coig_cqia_anchor.jsonl \
      --out_dir /root/autodl-tmp/datasets/mixed --anchor_ratio 0.1 --val_ratio 0.1 --seed 42
  ```
  预期：Train 82 条（73 领域 + 9 anchor）/ Val 8 条；**Anchor 全进 Train 不进 Val**（脚本末尾有分账核对行，异常会打 `!!`）
  > 用 `run_probe_e2e.sh mix` 跑会自动多做两件事：混合统计写进总日志、混合后自动对 Train 集跑四查并落盘
  > `logs/mix_check_<时间戳>.log` + `output/report_train_check_<时间戳>.json`。
- [ ] **2.2** 数据侧四查验收（模式 A）：
  ```bash
  python3 scripts/evaluate_sft.py --test_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl
  python3 scripts/evaluate_sft.py --test_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl
  ```
  **Anchor 分账说明**：锚点数据是通用指令（自然语言答案，无 CoT+JSON 协议），**不走舆情协议四查**（对它查 JSON 合法率/evidence 子串会 100% 假 FAIL，属口径错配）。脚本已改为分账处理——锚点只查行级结构合法（报告 `anchor_partition` 字段），四查只对领域样本执行。验收看：领域分账四查全过 + `Anchor 行级结构: OK`。
- [ ] **2.3** 离线 dry-run 验证加权逻辑（无 GPU 成本，三档都要跑）：
  ```bash
  SFT_POC_STAGE=POC1 python3 scripts/train_sft_lora.py --dry_run --train_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl
  SFT_POC_STAGE=POC2 python3 scripts/train_sft_lora.py --dry_run --train_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl
  SFT_POC_STAGE=POC3 python3 scripts/train_sft_lora.py --dry_run --train_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl
  ```
  验收判据：POC1 `Σw` **=** 有效 token 数（等权成立）；POC3 `Σw` **>** 有效 token 数（加权生效）；POC2/3 打印骨架遮蔽占比。不成立 = span 定位失效，修完再训。
  > dry-run 只吃领域样本即可（锚点 output 无 JSON 协议，span 规划对它是 no-op）。

## Step 3 · SFT 探针训练（POC1→POC3 消融，一次只动一个变量）

> 三档共用同一份代码，只切 `SFT_POC_STAGE`。探针规模 82 条 / 33 步 / 3 epoch（探针 B）。
> `run_probe_e2e.sh` 会自动从 `MODEL_DIR` 找基座、从 `DATA_DIR/mixed` 找数据，adapter 按时间戳落盘（`output/probe_lora_<时间戳>`，不互相覆盖），路径记录在 `output/latest_adapter_path.txt`。

- [ ] **3.1 POC1（baseline：仅前缀掩码）**：
  ```bash
  SFT_POC_STAGE=POC1 ./scripts/run_probe_e2e.sh train
  ```
- [ ] **3.2 POC2（+骨架掩码）**：
  ```bash
  SFT_POC_STAGE=POC2 ./scripts/run_probe_e2e.sh train
  ```
- [ ] **3.3 POC3（+字段加权，完整体）**：
  ```bash
  SFT_POC_STAGE=POC3 ./scripts/run_probe_e2e.sh train
  ```
  > 每档 adapter 目录带时间戳自动区分；回填时用 `cat output/latest_adapter_path.txt` 找到当前档，或直接取各时间戳目录。
- [ ] **3.4 每档训练中盯防**（本地 `logs/train_probe*.log`；要看曲线就 `SYNC_WANDB=1` 重跑同步 wandb，项目 `sentiment-sft-probe`）：
  - [ ] loss 能压下去、无 NaN（NaN = 整轮作废，查毒样本）
  - [ ] `grad_norm` 无连续 >10 暴跳
  - [ ] `eval_loss`（每 5 步一采）是否反弹 → **人判**，不自动早停；反弹则记录拐点（这正是「多轮过拟合 → 支撑正式轮锁 1 epoch」的证据）

## Step 4 · 推理与评估（每档都要跑）

> 以下 `<MODEL>` = `/root/autodl-tmp/models/Qwen3-4B-Instruct-2507`，`<MIXED>` = `/root/autodl-tmp/datasets/mixed`，`<ADAPTER>` = 对应 POC 档的 `output/probe_lora_<时间戳>`。

- [ ] **4.1** 推理（以 POC3 为例；POC1/POC2 换对应 adapter 路径）：
  ```bash
  python3 scripts/probe_infer.py \
      --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter_path output/probe_lora_<POC3时间戳> \
      --data_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl \
      --out output/pred_val_poc3.jsonl
  ```
  另跑一档**裸基座对照**（`--adapter_path` 留空）量化 SFT 增益
- [ ] **4.2** 模型对照评测（模式 B）：
  ```bash
  python3 scripts/evaluate_sft.py --gold_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl \
      --pred_file output/pred_val_poc3.jsonl --out output/report_poc3.json
  ```
- [ ] **4.3** 消融验收对照（训练手册 §1.4.4）：

  | 对比 | 验什么 | 验收线 |
  |---|---|---|
  | POC1 → POC2 | JSON 合规率有没有掉 | 不降（目标 ≥99.8% 方向） |
  | POC2 → POC3 | 核心标签（hard_case/sentiment/relevance）宏 F1 有没有涨、evidence 溯源有没有掉 | 涨且不掉 |
  | 全程 | `eval_loss` 反弹 | 无反弹 |
- [ ] **4.4**（可选加分）四档对照起步：裸基座 pred 即 T1 档输入；`scripts/eval_four_tier.py` 汇总
  （T2 公平提示词档需另造 instruction 变体数据，探针阶段可缓）

### 4.5 直观前后对比（难例样张，强烈建议做）

> **目的**：指标是抽象的，样张是直观的——选 1~2 条难例（优先 `irony` 反讽 / `multi_subject` 主体错挂 / `counterfactual`），把**裸基座输出**和**SFT 后输出**并排落盘，肉眼对比差异。这也是评估手册 §2 四档对照（T1 vs T3）的样本级直观版，可直接当面试演示素材。

- [ ] **4.5.1** 从 val 里挑难例样本（按 `hard_case_type` 过滤），确认其原文与金标：
  ```bash
  python3 -c "
  import json
  for i, line in enumerate(open('/root/autodl-tmp/datasets/mixed/sft_val.jsonl')):
      d = json.loads(line)
      gold = d.get('output') or d.get('gold') or {}
      if gold.get('hard_case_type') in ('irony','multi_subject','counterfactual'):
          print(i, gold.get('hard_case_type'), '|', d.get('input_text', d.get('text',''))[:50])
  "
  ```
  > 记下选中的行号（8 条 val 若无难例，就从 `data/train_synth_81.jsonl` 全量里挑，用同一段原文临时构造输入）。
- [ ] **4.5.2** 分别跑两版推理（同一份输入、同一解码配置）：
  ```bash
  # T1 · 裸基座
  python3 scripts/probe_infer.py \
      --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter_path "" \
      --data_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl \
      --out output/pred_base.jsonl
  # T3 · SFT 后
  python3 scripts/probe_infer.py \
      --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --adapter_path output/probe_lora_<POC3时间戳> \
      --data_file /root/autodl-tmp/datasets/mixed/sft_val.jsonl \
      --out output/pred_sft.jsonl
  ```
- [ ] **4.5.3** 把选中样本的「原文 + 金标 + 两版完整输出（含 CoT）」整理成一份对比样张，落盘：
  `output/对比样张_难例_<难例类型>.md`，表格结构：

  | | 裸基座（T1） | SFT 后（T3） |
  |---|---|---|
  | 原文摘要 | （同一段原文） | |
  | 金标 | sentiment / hard_case_type / risk / evidence | |
  | 完整输出 | CoT + JSON 原样粘贴 | CoT + JSON 原样粘贴 |
  | 差异点评 | 格式崩/主体错挂/反讽判反/证据非原文子串 | 输出协议齐 + 判对 + evidence 逐字子串 |

- [ ] **4.5.4** 对比时重点看四点（对应模型层指标②的五组）：
  1. 输出格式：基座是否写得出 `CoT + 闭集 JSON` 协议（大概率写不出或键值漂移）；
  2. 主体归属：难例上基座是否把「别人家的负面」错挂到目标主体（multi_subject 类的典型崩法）；
  3. 反讽极性：字面夸奖实为负面，基座是否判成 positive；
  4. evidence：基座的证据是否改写了原文（非连续子串），SFT 后是否逐字可溯源。

  > ⚠️ 样张只用于直观展示与演示，**不是指标**；结论仍以 4.2 的量化报告为准。8 条 val 的样本量不构成任何比例结论。

## Step 5 · 指标回收与参数回填清单

> 跑完上述步骤后，把下列数字逐项回填到各文档对应 `[待实测]` 处（回填后同步更新本清单勾选状态）。

### A. 训练过程指标（→ `train_metrics.json` / `logs/train_probe*.log`；开 `SYNC_WANDB=1` 跑过则有 wandb）

- [ ] **A1** 每档 `train_loss` 曲线与终值（POC1/2/3 各一份）
- [ ] **A2** `final_eval_loss`（训练结束 `trainer.evaluate()` 终考值），以及全程 eval_loss 是否反弹、拐点在第几步
- [ ] **A3** 训练耗时 / 显存峰值 / 实际步数（应 = 33 步，对账）
- [ ] **A4** 可训参数占比（`print_trainable_parameters()` 输出，量级 1% 上下）
- [ ] **A5** 真实 token 口径：骨架遮蔽占输出 / 占 JSON 段的真实比例（覆盖 dry-run 的字符级近似 17.8% / 68.8%）
- [ ] **A6** 判别字段在有效 token 中的真实占比（对照字符级：CoT 78.5% / 判别字段合计 8.3%）

### B. 模型层指标（→ `evaluate_sft` 模式 B 报告，每档一份）

- [ ] **B1** JSON 合规率（POC1 vs POC2 对照 → 消融结论 ①）
- [ ] **B2** evidence 原文子串命中率（POC2 vs POC3 → 消融结论 ②，确认未掉）
- [ ] **B3** hard_case 7 类宏均 F1（POC2 vs POC3 → 消融结论 ②主验）
- [ ] **B4** sentiment 负向召回 × 误报率（**成对报**，pa_fp_pair）
- [ ] **B5** risk Subset Accuracy
- [ ] **B6** 思考标签闭合率（须 100%）
- [ ] **B7** 裸基座对照（SFT 净增益 = T3 − T1 口径的探针版）
- [ ] **B8** 分桶切片（hard_case_type × sentiment × industry 四维），小样本桶（n<5）只报数不报率

### C. 待回填文档映射

| 实测项 | 回填到 |
|---|---|
| A1/A2 | `notes/舆情负面精判_SFT训练实战手册.md` §1.4.4 表「状态」列、§1.4.6 红线解除 |
| A5/A6 | 同上 §1.4.5 / §2.2 的「真 BPE 分词口径 `[待实测]`」 |
| B1~B4 | 同上 §1.4.4 消融验收表 + FIELD_W 权重 4.0/3.0/2.0/0.4 的消融支撑（SOP §8.8 V0/V1 表） |
| A3/A4 | 训练手册 §3.2 步数预算表实测列 |
| B1~B7 | `notes/舆情负面精判_评估手册_SFT_GRPO.md` 模块一② 的 `[待实测回填]` 项 |

### D. 防串号纪律（回填时自查）

- [ ] Val 8 条 ≠ 成绩单：所有数字标注「探针 8 条 val 口径」，**不外推全量**，不写进简历
- [ ] 8 条 val 的 95% CI 约 ±34.7pp —— 只用于消融相对趋势判断，不报绝对准确率
- [ ] 探针数字与 500 条金标、单客户运营量级是三本账，严禁混写
- [ ] FIELD_W 绝对数值（4.0/3.0/2.0/0.4）消融完成前不写入简历正文

## Step 6 · 后续（本清单之外，SFT 收口后）

- [ ] GRPO 探针：`run_grpo_probe.py --dry_run` → 真实 rollout（依赖本清单 Step 3 产物 adapter）
- [ ] GRPO 训练冒烟：`GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh`（同一环境，无需重装依赖）
