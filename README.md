# 舆情负面精判 · 小样本探针实验（可一键跑通）

这是 SFT + GRPO 方案的**最小可执行闭环**：下载 → 混合 → 训练 → 推理 → 评测，全程落盘日志，训练指标同步 wandb。

> 本目录是自包含的：所有默认路径都指向 `tests/` 内部，**不需要**依赖上层目录。
> 目标环境：AutoDL 单卡（RTX 4090 24GB / RTX 3090 24GB），Linux + Python 3.10+

---

## 0. 目录结构

```
MVP/tests/
├── README.md                       ← 本文件
├── data/
│   ├── train_synth_81.jsonl        ← 真实金标语料 81 条（异族打标 + 四查全过）
│   ├── grpo_probe_pool.jsonl       ← GRPO 新生成语料池 51 条（种子 777+888，与 SFT 集去重）
│   ├── grpo/grpo_prompts_probe.jsonl ← GRPO 探针数据 51 条：只有 {prompt, gold}
│   ├── train_golden_sample_100.jsonl  Mock 训练桩（CI 用，不可用于真实训练）
│   ├── test_golden_sample_50.jsonl    Mock 测试桩（单测用）
│   ├── coig_cqia_anchor.jsonl      ← 运行 step 1 后生成（公开通用 Anchor）
│   └── mixed/                      ← 运行 step 2 后生成
│       ├── sft_train.jsonl         分层切分后的训练集（含 9:1 Anchor）
│       └── sft_val.jsonl           分层切分后的内部验证集
├── lib/                            ← 共享层：被 scripts/ 与 unit/ 共同 import
│   ├── reward_fns.py               ★ GRPO 四维规则打分器（零外部依赖；枚举与 validate_schema 已下沉 schema_contract，本文件再导出保持兼容）
│   ├── schema_contract.py          ★ 输出协议唯一判定源：闭集枚举 + validate_schema + parse_output（训练/评测/质检三侧同源）
│   ├── eval_metrics.py             ★ 评估指标纯函数层：宏均 / Subset Accuracy / 负向召回×误报率成对 / 分桶 / 非法标签计数
│   ├── rubric_check.py             ★ 离线 Rubric 三判定（主体点名 / 推理无矛盾 / 证据支撑），可注入离线裁判
│   ├── release_gate.py             ★ 发布门四道纯逻辑（历史错题 / 金标不退化 / L3 安全 / 格式合规），N/A 不假装 PASS
│   ├── grpo_common.py              ★ GRPO 共享层：数据展平 / k 分桶 / 训练预算（零依赖）
│   ├── grpo_probe_core.py          ★ GRPO 探针纯逻辑：打分入组 / rollout / k 分流
│   ├── corpus_schema.py            ★ 行业风险二级词表 + 提示词加载 + 闭集枚举（同源再导出 schema_contract）
│   ├── llm_client.py               API 客户端（generate 的依赖）
│   └── sft_weight_planner.py       字段权重 span 规划器（train 的依赖，勿删）
├── unit/                           ← 离线单测（零外部依赖，任何机器都能跑）
│   ├── test_reward_fns.py          ★ 打分器单测 40 项
│   ├── test_build_eval_set.py      ★ 金标构建器纯逻辑单测
│   ├── test_weight_planner.py      ★ SFT 权重 span 规划器单测 12 项
│   ├── test_grpo_probe.py          ★ GRPO 链路单测 48 项
│   ├── test_eval_metrics.py        ★ 评估指标纯函数单测 25 项
│   ├── test_evaluate_sft.py        ★ 评测脚本纯逻辑单测 21 项（协议/指标/分桶/非法输出/证据分母/闭合率进 verdict）
│   ├── test_rubric_check.py        ★ 离线 Rubric 单测 20 项（三判定项 + 注入 judge）
│   └── test_release_gate.py        ★ 发布门单测 25 项（四门 PASS/FAIL/N/A + 归因）
├── prompts/                        ← gen/anno 双端提示词（单主体隔离锁在内）
├── scripts/                        ← 只放 CLI 入口（可执行脚本）
│   ├── run_probe_e2e.sh            ★ 一键串跑（5 步）
│   ├── run_grpo_train.sh           ★ GRPO 后台启动（nohup + 时间戳日志 + PID + 拓扑选择）
│   ├── download_assets.py          1 下载：COIG-CQIA Anchor + 基座模型
│   ├── mix_sft_dataset.py          2 混合：9:1 + 分层切分 train/val
│   ├── train_sft_lora.py           3 训练：LoRA + 覆写 compute_loss（掩码/加权随 POC 阶段切换）
│   ├── probe_infer.py              4 推理：加载 Adapter 批量生成
│   ├── evaluate_sft.py             5 评测：五段式（解析→指标→分桶→门禁→落盘 --out report.json）
│   ├── stat_seq_len.py             序列长度统计（domain/anchor 分账 + Top-N 定位 OOM 长样本；无 tokenizer 退化为字符级近似）
│   ├── pick_hard_cases.py          难例选样（按 hard_case_type 过滤打印行号+摘要，对比样张/四档对照选样用）
│   ├── eval_four_tier.py           ★ 四档控制变量对照（T1 基座/T2 提示词/T3 SFT/T4 GRPO → tier_report.json）
│   ├── release_gate.py             ★ 发布门 CLI（四道门独立判定 + 归因表落盘 gate.json）
│   ├── generate_corpus.py          语料量产（生成+异族打标+四查）
│   ├── build_eval_set.py           ★ 三方异族一致性 → 金标评测集（κ ≥ 0.7 门禁）
│   ├── build_grpo_probe.py         ★ 语料池 → {prompt, gold} 契约转换
│   ├── run_grpo_probe.py           ★ 同题 G=4 采样 → 四维打分 → k∈{0..4} 分流
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
cd MVP/tests
chmod +x scripts/run_probe_e2e.sh

# （可选）登录 wandb，登录后训练曲线会同步在线；不登录则只写本地日志
wandb login

./scripts/run_probe_e2e.sh          # 全流程：download → mix → train → infer → test
```

按步骤单独执行（模型已下好、想只重跑训练时很有用）：

```bash
./scripts/run_probe_e2e.sh download   # 步骤1
./scripts/run_probe_e2e.sh mix        # 步骤2
./scripts/run_probe_e2e.sh train      # 步骤3
./scripts/run_probe_e2e.sh test       # 步骤4+5（推理+评测）
```

可用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `WANDB_API_KEY` | 未设 | 设了就同步 wandb，不设只写本地日志 |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct` | 换基座（对比 7B 消融时用） |
| `EPOCHS` | `3` | 训练轮数 |
| `SFT_POC_STAGE` | `POC1` | **Loss 消融阶段**：`POC1`=仅前缀掩码 / `POC2`=+骨架掩码 / `POC3`=+字段加权 FIELD_W（详见 §训练 与 `sft_weight_planner.POC_STAGE`） |
| `FORCE=1` | `0` | 忽略已有产物强制重跑 |

```bash
# 例子：7B 基座对照 + 强制重下 + 训练 2 轮
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct FORCE=1 EPOCHS=2 ./scripts/run_probe_e2e.sh

# 例子：跑 POC2 消融（只开骨架掩码、字段仍等权）
SFT_POC_STAGE=POC2 ./scripts/run_probe_e2e.sh train
```

### 快速消融对比：POC1 ~ POC3 运行指令集

三档共用同一份代码、只切一个环境变量，一次只动一个变量。

```bash
cd tests/

# 离线干跑：无 GPU 验证权重分配
SFT_POC_STAGE=POC1 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
SFT_POC_STAGE=POC2 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
SFT_POC_STAGE=POC3 python3 scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl

# 真实探针训练
SFT_POC_STAGE=POC1 ./scripts/run_probe_e2e.sh train
SFT_POC_STAGE=POC2 ./scripts/run_probe_e2e.sh train
SFT_POC_STAGE=POC3 ./scripts/run_probe_e2e.sh train

# 全流程（训练 → 推理 → 评测）
SFT_POC_STAGE=POC1 ./scripts/run_probe_e2e.sh all
SFT_POC_STAGE=POC2 ./scripts/run_probe_e2e.sh all
SFT_POC_STAGE=POC3 ./scripts/run_probe_e2e.sh all
```

---

## 2. 手动分步执行（等价于上面的每一步）

### 步骤 1 · 下载公开数据集 + 基座模型

```bash
# 1a. 公开通用指令数据集 COIG-CQIA（防灾难性遗忘 Anchor，9:1 混入）
python3 scripts/download_assets.py --download_anchor --anchor_samples 100 --data_dir data
#   → data/coig_cqia_anchor.jsonl（已过滤任何带情感倾向的脏标）

# 1b. 基座模型（AutoDL 用 ModelScope 国内直连，不需梯子）
python3 scripts/download_assets.py --model Qwen/Qwen3-4B-Instruct \
    --model_dir models --source modelscope
#   → models/Qwen3-4B-Instruct/
#   换 HuggingFace 源：--source huggingface
```

> 联网拉取 COIG-CQIA 失败时，脚本会**自动落盘内置的高质量通用问答兜底包**，不会卡住流程。

### 步骤 2 · 领域数据 : 公开数据 = 9 : 1 混合

```bash
python3 scripts/mix_sft_dataset.py \
    --domain_file data/train_synth_81.jsonl \
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
    --model_name_or_path models/Qwen3-4B-Instruct \
    --train_file data/mixed/sft_train.jsonl \
    --val_file data/mixed/sft_val.jsonl \
    --output_dir output/probe_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --lr_scheduler_type cosine \
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
- **收尾产物**：`output/probe_lora/adapter_model.safetensors` + `output/probe_lora/train_metrics.json`（含 `train_loss` / 耗时 / 样本数）。
- **步数与停止口径**（启动时会打印「训练规模预算」区块）：

| | 单卡 batch | 梯度累积 | 实际(有效) batch | 训练样本 | 每 epoch 优化器步 | 总步数 |
|---|---|---|---|---|---|---|
| **探针 A · 未下 Anchor**（73 条 × 3 epoch，盘上现况） | 2 | 4 | **8** | 73 | 10 | **30** |
| **探针 B · 已下 Anchor**（**82** 条 × 3 epoch = 73 领域 train + 9 anchor；先跑 `download`+`mix` 才有 9:1 混入） | 2 | 4 | **8** | 82 | 11 | **33** |
| **正式**（2912 条 × 1 epoch，即 2880 领域×0.9 + 320 Anchor） | 4 | 2 | **8** | 2912 | 364 | **≈364** |

  > **探针口径提醒**：`data/coig_cqia_anchor.jsonl` 未下载时，`data/mixed/` 是 `train_synth_81.jsonl` 直切的 73/8（9:1 锚点尚未混入）；跑完步骤1+2 才是 81/9 的标准探针集。两档步数差 3 步，选哪个看前置条件。

  **不做早停**——SOP §训练纪律是「epoch 受控 + lr 2e-5 cosine 退火到 0」，靠 `eval_loss` 反弹**由人判定**过拟合，不让 Trainer 自动掐停。**探针的任务是冒烟验证**：按当前 `SFT_POC_STAGE` 对应的掩码/加权组合没写崩、loss 能压下去、adapter 能出盘，3 epoch（30~33 步）足够；它**不承担「定正式轮数」的职责**——正式轮数 SOP 已定死 1 Epoch。
  `--val_file` 会自动开 `eval_strategy=steps`，`eval_loss` 走同一套（同 POC 阶段）加权口径，训练结束会复测一次并存进 `train_metrics.json`。
- **无 GPU 时**先验证加权逻辑：`python3 scripts/train_sft_lora.py --dry_run --train_file data/train_synth_81.jsonl`
  会打印当前 POC 阶段、每个字段的 token 区间与权重。**各阶段该看到什么**：

  | 阶段 | 字段权重 | 骨架 | 有效 token 与 Σw |
  |---|---|---|---|
  | `POC1` | 全 `1.0x` | 不遮 | Σw **=** 有效 token 数（等权成立） |
  | `POC2` | 全 `1.0x` | 遮（`0.0x`） | 有效 token 变少、Σw = 新有效数；打印「骨架遮蔽(0×) 占输出」 |
  | `POC3` | `4.0/3.0/3.0/2.0/0.4` | 遮 | Σw **>** 有效 token 数；另打印高权/降权占比 |

  > `--dry_run` 用字符级伪编码（1 字符 = 1 token）做近似，**骨架遮蔽占比是字符级近似值、不是真实 token 占比**，真值需上真 tokenizer 才算数。
  >
  > **空表 `[]` 是例外**：`risk_categories: []` 的两个方括号单独登记 span、拿 `risk_categories` 的权重（POC2 `1.0` / POC3 `2.0`），不随骨架置 0——否则该字段在空表样本上无任何梯度，模型学不会「该空就空」。`subject` 回显值则相反：未登记 → 被骨架遮掉，属**主动不监督**。

### 步骤 4 · 推理

```bash
python3 scripts/probe_infer.py \
    --base_model models/Qwen3-4B-Instruct \
    --adapter_path output/probe_lora \
    --data_file data/mixed/sft_val.jsonl \
    --out output/pred_val.jsonl
```

实时输出三项合规率：**CoT 标签闭合率 / JSON 解析合法率 / evidence 原文命中率**。
`--adapter_path` 留空可跑**裸基座对照**，用来量化 SFT 带来的增益。

### 步骤 5 · 评测

```bash
# 模式 A：数据侧四查验收（确认喂进去的语料是干净的）
python3 scripts/evaluate_sft.py --test_file data/mixed/sft_train.jsonl

# 模式 B：模型对照评测（金标 vs 预测），输出逐难点/逐行业切片准确率
python3 scripts/evaluate_sft.py --gold_file data/mixed/sft_val.jsonl \
                                --pred_file output/pred_val.jsonl
```

### 步骤 5b · 序列长度统计（OOM 排查 / 数据体检）

```bash
# AutoDL：真 token 口径 + Top-N 最长样本定位（domain / anchor 分账）
python3 scripts/stat_seq_len.py \
    --train_file data/mixed/sft_train.jsonl \
    --tokenizer models/Qwen3-4B-Instruct-2507 --top_n 5

# 无 tokenizer 环境：字符级近似（1 汉字 ≈ 1.4 token）
python3 scripts/stat_seq_len.py --train_file data/train_synth_81.jsonl
```

输出 domain / anchor 两桶各自的 prompt / output / total 的 p50/p95/p99/max，
末尾自动打印归因速读（哪桶贡献长尾）。典型用途：
排查「大 vocab + 长 seq 的 CE 峰值 OOM」（长样本来源是领域数据还是锚点）、
确认训练数据是否逼近 `max_seq_length 2560`（超过即被截断）。

---

## 3. GRPO 探针（第二阶段：四维规则打分器）

> 真理源：`MVP/notes/舆情负面精判_GRPO数据准备实战手册.md` §0.2 落盘契约 + §五 四维 Reward 契约。
> 最终交付三样：`grpo_prompts_*.jsonl` + `reward_fns.py` + **零离线 completion**。

### 3.1 数据：只有 `{prompt, gold}`，不带解题过程

`data/grpo/grpo_prompts_probe.jsonl` — **51 条**（种子 777+888 两批生成，与 SFT 81 条跨文件去重，实测最大 Jaccard 0.095、0 重复）

```json
{
  "prompt": "<|im_start|>system\n你是舆情负面精判模型…风险词表…<|im_end|>\n<|im_start|>user\n<待判原文><|im_end|>\n<|im_start|>assistant\n",
  "gold": {
    "sentiment": "positive",
    "relevance": "core",
    "hard_case_type": "counterfactual",
    "risk_categories": [],
    "evidence": "幸亏审计部门及时发现并制止"
  },
  "meta": {"industry": "finance_insurance", "provenance": "synthetic"}
}
```

- **不带 completion 是设计**：CoT 由策略 On-Policy 现场采样，算完组内优势即弃；攒着会被误拿去当 SFT 语料。
- `gold` 只有 5 个字段（**没有 `subject`**），但 `R_format` 要求模型回显 `subject` —— rollout 时从 prompt 的 `目标主体：【…】` 补上。
- `evidence` 原文可用 `reward_fns.extract_input_text(prompt)` 反解，**两字段自洽**（40/40 已验证）。

**当前分布**：难点 `explicit 13 · counterfactual 8 · comparative 8 · metaphor 6 · multi_subject 6 · fact_implicit 6 · irony 4`；极性 `negative 29 / positive 16 / neutral 6`；语篇 `core 46 · passing 3 · foil 2`；六行业全覆盖。（短板已补：fact_implicit 2→6、neutral 3→6；positive 未下采样，占比 40%→31%）

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

```bash
python3 scripts/build_grpo_probe.py \
    --src data/grpo_probe_pool.jsonl \
    --out data/grpo/grpo_prompts_probe.jsonl \
    --with_meta
# 结果: 51 条入池，0 条丢弃（gold 5 字段齐 + evidence 全是原文合法子串）
```

要**新生成一批**语料池（需要 dashscope key，走真实 API）：

```bash
python3 scripts/generate_corpus.py --n 60 --seed 777 --concurrency 6 \
    --gen_provider dashscope --gen_model qwen-max \
    --anno_provider dashscope --anno_model deepseek-v4-pro \
    --out data/grpo_probe_pool.jsonl --append \
    --cross_dedup data/train_synth_81.jsonl    # ← 与 SFT 训练集隔离
```

### 3.4 还缺什么

| 缺口 | 说明 |
|---|---|
| **`run_grpo_probe.py` / `train_grpo.py`（已写，未上 GPU 实测）** | 代码与单测（48 项）齐了，`--dry_run` 已在本地跑通；真实 rollout 与训练**必须等 SFT 探针训完 + AutoDL 环境** |
| 正式 `grpo_prompts_3000.jsonl` | 需生产日志（阶段一 A/B 双路筛题）+ 阶段二金标 + 阶段三 G=4 难度筛选，本地造不出来 |
| `irony` / `metaphor` 偏薄 | irony 仅 4 条，是 GRPO 高价值难例；下轮可 `--filter_hard_cases irony metaphor` 再补 |
| 阿里云百炼欠费 | 生成端 4/24 报 `Arrearage` HTTP 400，**充值前不要再跑生成/打标** |

> ⚠️ 这 51 条是**构造区探针数据**，与 500 条人标金标测试集是两本账，不得混着写进简历。

### 3.5 训练：`train_grpo.py`（TRL GRPOTrainer + 四维规则 Reward）

**对标参考取其可用、弃其不可用**：`llm-engineering-core/12S2W10_rlhf/GRPO/train.py`（TRL 栈、契约化 reward、wandb 降级、`--limit` 冒烟）被吸收；`13S3W5_rm_r1/RM-R1/rm_r1`（verl 全参分布式）**弃用**——与「LoRA + 单卡 24GB」正面冲突。

```bash
# 1) 任何机器都能跑的干跑体检（数据契约 + 门禁不变量 + 训练预算 + 依赖版本）
python3 scripts/train_grpo.py --data data/grpo/grpo_prompts_probe.jsonl --dry_run

# 2) 真实训练（AutoDL，需 pip install trl）
./scripts/run_grpo_train.sh                       # 自动按 GPU 数选拓扑
GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh   # 先冒烟再放量
```

**干跑会打印这张超参表（每个数都能算出来源，不是拍的）**：

| 超参 | 值 | 来路 |
|---|---|---|
| 每步题数 | 8 = per_device 4 × grad_accum 2 | 与 SFT 有效 batch 8 对齐 |
| 生成组数 G | 4 | 组内要有方差，4 是最小偶数组 |
| **有效 batch** | **32 条轨迹/步 = 8 题 × G=4** | 手册 §1.3 定稿 |
| 采样温度 T | 0.7 | T=0 时 4 份采样相同 → std=0 → RL 退化成更贵的 SFT |
| 学习率 lr | 2e-6 | RL 叠在 SFT 已收敛策略上，比 SFT 的 2e-5 低一个量级 |
| KL 系数 β | 0.04 | 三闸①：常数 KL 管分布漂移，长度由 R_length 管，两者正交 |
| 输出预算 | 200 | **CoT≤120 + JSON 骨架~70 + 余量**（见下方陷阱） |
| 51 题 × epoch=1 | 7 步 | ceil(51/8)，正式 3000 题 × epoch=1 → 375 步 |

**三个抄了就崩的坑**（脚本里都有对应注释）：

1. **reward 不能拆成 4 个函数平权相加** —— 我们的门禁是**短路**语义（`R_format` 挂了，`R_accuracy`/`R_length` 归 0）。拆开相加会丢短路。所以只有一个 reward func，内部调 `compute_reward`。
2. **`max_completion_length` 不能按手册 65~105 截** —— 那一格说的是 **CoT 档位**（`R_length` 只量 CoT），不是整段预算。按 105 硬截会把 JSON 砍在中间 → 全池 `R_format` −2.5 短路 → 训练废掉。
3. **`loss_type` 必须是 `grpo`（均值中心化 + std 归一）** —— 与 `reward_fns.group_advantages` 公式逐字一致；新版 TRL 默认 `dr_grpo` 不做中心化，优势带非零均值即偏离契约。

另有两处防御（启动时自动处理）：
- **prompt 上限体检**：实测 prompt 最长 token 数（含行业风险词表）若超 `max_prompt_length`，**自动放宽**——截断会砍掉待判原文 → `evidence` 必然溯源失败 → 全池短路。
- **Zero-Advantage Masking 无需改 Trainer**：我们的实装条件是「整组都触顶门禁罚」→ 4 个分数相同 → 方差 0，TRL 的 `(r−mean)/std` 天然给 0 优势。启动时打 `ZAM 自检 PASS` 即确认两者语义重合，**不去 hack TRL 私有 API**。

**版本要求**：`transformers>=4.55`（Qwen3 架构要 4.51+，参考仓库那套 `trl==0.14` 不认 Qwen3）、`trl>=0.19`（拿到 `temperature` / `processing_class=`）。

### 3.6 难度分流：`run_grpo_probe.py`（只留 k∈{1,2,3}）

```bash
# 干跑：用 gold 反构「标准答案」当采样结果，验数据 + 打分器 + 分桶全链路
python3 scripts/run_grpo_probe.py --data data/grpo/grpo_prompts_probe.jsonl --dry_run

# 真实 rollout（SFT 探针训完后）
python3 scripts/run_grpo_probe.py \
    --data data/grpo/grpo_prompts_probe.jsonl \
    --base_model models/Qwen3-4B-Instruct \
    --adapter output/probe_lora \
    --g 4 --temperature 0.7 --max_new_tokens 200 \
    --out output/grpo/grpo_probe_kstats.json
```

**k = 同题 G 份采样里「判对 且 evidence 溯源过门禁」的份数**：

| k | 处置 | 为什么 |
|---|---|---|
| 0（全错） | 丢弃 | 组内 std=0，梯度为 0；RL 无米下锅，退回 SFT 补课 |
| **1~3** | **训练池** | 组内有正负样本可比，这是唯一有信息量的档 |
| 4（全对） | 丢弃 | 方差恒 0，花 RL 的钱跑了个更贵的 SFT |

产出：`kstats.json`（k 直方图 + 训练池规模 + 逐难点/极性分布）+ `kstats.rows.jsonl`（逐题明细）。**这就是正式池三阶段筛题的第 3 步**——探针阶段用它验证打分器与 rollout 链路，正式阶段用它把 3000 题筛成真正可训练的子集。

### 3.7 金标评测集：`build_eval_set.py`（三方异族一致性，κ ≥ 0.7 门禁）

语料池里现成的打标只有一家的判断，**不能直接当金标**。本脚本对同一条原文做三次**异族**打标，
按分歧程度分三档处置（`materials/DataPipeline_SOP_v4.md` §3.2/§3.3）：

| 档 | 条件 | 处置 |
|---|---|---|
| `gold` | A(DeepSeek) = B(第三方1) = C(第三方2) 全字段一致 | 入 `eval_gold_probe.jsonl`，可直接喂 `evaluate_sft` |
| `boundary` | 仅 `hard_case_type` 分歧 | 入 `eval_boundary.jsonl`，**不当金标**，留给 DPO / 难例复盘 |
| `drop` | `sentiment` / `relevance` / `risk_categories` 分歧 | 丢弃（真歧义，进训练只会教错） |

> **异族铁律**：生成端（Qwen-Max）是写手，**不参与裁决**；judge 必须与被测样本的打标模型异族。

```bash
# 先只摸底 20 条
python3 scripts/build_eval_set.py \
    --src output/corpus/eval_probe_pool.jsonl \
    --out_dir output/eval --limit 20 --concurrency 4 \
    --third1_provider openai --third1_model claude-opus-5.5 \
    --third2_provider openai --third2_model gpt-6-luna

# 铺量 80 条（--limit 80）
```

产出：`eval_gold_probe.jsonl`（金标）+ `eval_boundary.jsonl`（边界样本）+ `eval_agreement_report.json`
（逐字段、逐模型对的 κ 与一致率 + `PASS / BOUNDARY / FAIL` 裁决）。

> ⚠️ **κ < 0.6 说明是标注法典本身有歧义**（不是模型差异），必须回炉重写标注规范，
> 不得靠「再跑几遍」蒙混过关。

---

## 4. 跑通后你会拿到什么

| 产物 | 位置 | 用途 |
|---|---|---|
| LoRA Adapter | `output/probe_lora*/` | 下一步 GRPO 的起点 |
| 训练日志 | `logs/train_probe_*.log` | 断点回看、面试演示 |
| 训练指标 | `output/probe_lora*/train_metrics.json` | 多轮探针对比 |
| wandb 曲线 | `sentiment-sft-probe` 项目 | loss 收敛可视化 |
| 预测文件 | `output/pred_*.jsonl` | 喂给 `evaluate_sft` 模式 B |
| 评测报告 | 终端 + `logs/probe_e2e_*.log` | 难点/行业分桶准确率 |
| GRPO 打分器 | `lib/reward_fns.py`（40 项单测全绿） | 4 条规则 + 单调性 + 门禁不变量 |
| GRPO 探针数据 | `data/grpo/grpo_prompts_probe.jsonl` | `{prompt, gold}` 51 条，等 SFT 模型可 rollout |
| GRPO 难度分流 | `output/grpo/grpo_probe_kstats.json` + `.rows.jsonl` | k 直方图 + 训练池规模 + 逐题明细 |
| GRPO 训练日志 | `logs/grpo_train_*.log`（nohup 后台，时间戳命名） | 断点回看 |
| GRPO Adapter | `output/grpo_lora/` + `grpo_metrics.json` | SFT→GRPO 两段式终点 |
| GRPO 单测 | `unit/test_grpo_probe.py`（48 项） | 数据契约 / k 分桶 / reward 列对齐 / 版本兼容 |
| 金标评测集 | `output/eval/eval_gold_probe.jsonl` | 三方异族全一致的金标（`evaluate_sft` 模式 A/B 直接读） |
| 边界样本 | `output/eval/eval_boundary.jsonl` | 仅难点类型分歧，留给 DPO / 难例复盘 |
| 一致性报告 | `output/eval/eval_agreement_report.json` | 逐字段 κ + 一致率 + PASS/BOUNDARY/FAIL 裁决 |

**全量自检一条命令**（零外部依赖，任何机器都能跑）：

```bash
cd MVP/tests
python3 -m unittest unit.test_reward_fns unit.test_build_eval_set unit.test_weight_planner unit.test_grpo_probe unit.test_eval_metrics unit.test_evaluate_sft unit.test_rubric_check unit.test_release_gate
# 212 tests · OK
```

### 5.1 评测与发布门（评估手册缺口 1~5、11~18 已落地）

```bash
# 模式 A：数据集自检（分母=总样本数；闭合率纳入 verdict）
python3 scripts/evaluate_sft.py --test_file data/test_golden_sample_50.jsonl --out output/report.json

# 模式 B：模型对照评测（四维分桶 + 非法标签计数 + 负向召回×误报率成对 + 发布门）
python3 scripts/evaluate_sft.py --gold_file data/test_golden_sample_50.jsonl \
                                --pred_file output/pred_val.jsonl \
                                --out output/report.json

# 发布门 CLI（四道门独立判定；缺输入的门返回 N/A，不假装 PASS）
python3 scripts/release_gate.py --gold data/test_golden_sample_50.jsonl \
                                --pred output/pred_val.jsonl \
                                --old_error_list output/old_errors.txt \
                                --baseline_json output/report_prev.json \
                                --out output/gate.json

# 四档控制变量对照（T1 基座 / T2 公平提示词 / T3 SFT / T4 GRPO；缺档自动 N/A）
python3 scripts/eval_four_tier.py --gold data/test_golden_sample_50.jsonl \
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
4. **GRPO 的 51 条探针 ≠ 训练池**——正式池是 `grpo_prompts_3000.jsonl`，必须经「生产日志筛题 → 金标 → G=4 难度筛选只留 k∈{1,2,3}」三阶段；探针这 51 条只用来**验证打分器与 rollout 链路通不通**。
5. **GRPO 不切 dev 集**——RL 的失效模式是通用能力退化和 Reward Hacking，dev 集测不出这两样；等价监控是 `k=4` 基线集盯通用能力掉没掉、`k=0` 集盯难题攻克进度，500 条金标只评不训。
6. **`run_grpo_probe.py --dry_run` 的 k=4 不是模型成绩**——干跑用 gold 反构标准答案当采样结果，只证明「数据 + 打分器 + 分桶」链路通。真实 k 分布必须等 SFT 探针训完再 rollout 才有数。同理，`train_grpo.py --dry_run` 打印的 32 轨迹/步、7 步是**预算**不是实测。

---

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `缺少依赖: torch transformers peft` | `pip install torch transformers peft accelerate wandb` |
| GRPO 报 `No module named 'trl'` | `pip install 'transformers>=4.55' 'trl>=0.19' 'peft>=0.12' datasets`（Qwen3 要 4.51+，TRL 0.14 不认） |
| GRPO 启动就报「batch 不能被 G 整除」 | `--per_device_train_batch_size` 必须是 G 的倍数（默认 4，G=4 合法） |
| GRPO 日志提示「无 temperature 字段」 | TRL 太老；不升级的话 T 若为 0，组内方差恒 0，RL 会退化成更贵的 SFT |
| GRPO 慢 / rollout 占时间 | 单卡默认 HF generate；有 2 卡用 `run_grpo_train.sh`（自动开 vLLM，训练 1 进程 + vLLM 独占 cuda:1） |
| COIG-CQIA 拉取超时 | 脚本已自动落盘内置兜底包；或 `pip install modelscope` 后重试 |
| wandb 提示未登录 | `wandb login`，或加 `--no_wandb` 关掉 |
| OOM | 降 `--per_device_train_batch_size 1 --gradient_accumulation_steps 8` |
| 想换 7B 对照 | `MODEL_NAME=Qwen/Qwen2.5-7B-Instruct ./scripts/run_probe_e2e.sh all` |
| `loss_weights` 被 Trainer 丢掉 | 训练命令里必须保留 `--remove_unused_columns false` |
