# SFT+GRPO 面试数字清单 · 实验回填版

> 用途：上 GPU 做实验时逐项填死，面试前背熟。每条都标了「来源」与「状态」。
>
> **状态图例**：
> - ✅ **已焊死** —— 代码默认值 / 文档单一权威源，可直接背，不用改
> - 🟡 **待实测回填** —— 上 GPU 跑完才能填（坑位已标好，别编）
> - 🔴 **口径有坑** —— 面试一碰就穿帮，务必先想清楚说辞再开口
>
> **口径铁律**：探针（81 条 SFT / 51 条 GRPO）与生产设计（3200 条）**是两套数字**，
> 面试被问「训练数据多少」必须先分清对方问的是哪一套，严禁串号。

---

## 1. 基座与卡型（✅ 已焊死）

> 🔴 **两套硬件必须分开答**：**实验环境 = RTX 3090 / 4090（单卡 24GB）**，
> **生产环境 = A10 24GB 企业卡集群**。两者同为 24GB 显存档，**显存预算与 OOM 结论可直接平移**；
> 但**算力、带宽、是否 ECC 不同，绝对耗时不通用**——4090 的 wall-clock 不能当 A10 的生产数字报。

### 1.1 实验环境（探针真实跑卡）

| 项 | 值 | 说明 |
|---|---|---|
| 卡型 | **RTX 4090 / 3090，单卡 24GB** | `TODO_GRPO探针快速启动.md:3` 明确写定 |
| 显存对齐 | 与生产 A10 同为 24GB → **显存预算、OOM 阈值、gradient checkpointing 必要性可平移** | 这是选消费卡做探针的最大理由 |
| bf16 支持 | ✅ 两者均支持（3090 = Ampere SM 8.6；4090 = Ada SM 8.9），bf16 张量核可用 | 无需降级 fp16 + loss scaling |
| flash-attn | ✅ Ampere/Ada 均支持 FlashAttention-2 | 与 A10 同代技术栈 |
| 算力差 | 4090（~165 TFLOPS FP16）≈ **1.3~2× A10（~125）**；3090（~71）反而慢于 A10 | **耗时不通用，见 5.2 坑 C** |
| 短板 | 无 ECC、无 NVLink（多卡只能走 PCIe） | 单卡探针不受影响；多卡 GRPO 时注意通信 |

### 1.2 模型与部署（生产口径）

| 项 | 值 | 来源 |
|---|---|---|
| 基座模型 | Qwen3-4B-Instruct（4B） | `train_sft_lora.py` / `train_grpo.py` 默认值 |
| 训练精度 | bf16 | 两脚本 `bf16=torch.cuda.is_available()` |
| LoRA | r=16 / alpha=32 / dropout=0.05，target=全量线性层（q/k/v/o/gate/up/down），bias=none | `train_sft_lora.py:194-196`、`train_grpo.py:510-516` |
| 生产部署 | A10 24GB 集群，INT4(AWQ) + vLLM，吞吐基线 ~4 张 + 双机容灾 → 4~8 张弹性区间 | `SFT负面精判_MVP设计架构.md:52` |

**面试话术锚点**：4B bf16 权重 ≈ 8G，加 seq² activations + 151936-vocab logits（`train_sft_lora.py:574` 注释有完整显存账本）。

---

## 2. 数据侧（探针 vs 生产 分开答）

### 2.1 探针实验（真实已落盘，上 GPU 前先看清规模）

| 项 | 值 | 来源 |
|---|---|---|
| SFT 探针数据 | `data/probe_test_sft_train_synth.jsonl`，**81 行**（实测 `wc -l`） | 已核 |
| GRPO 探针数据 | `data/probe_test_grpo_train_synth.jsonl`，**51 行**，契约 `{prompt, gold}` | 已核 |
| GRPO eval 基线集 | `data/grpo/grpo_eval_probe.jsonl`（8 条）——🟡 **文件未提交**，需在 AutoDL 按 `TODO_GRPO探针快速启动.md:76-78` 的 snippet 现生成 | 见下 |
| 7 类难例 + 15% 空表 | 由 `align_sft_data.py` / `build_grpo_probe.py` 质检保证 | TODO 2.1 |

> 🔴 **待办**：TODO 5.19 声称 eval 基线集「已落盘」，但 `find` 全仓无 `grpo_eval_probe.jsonl`。
> 它是在 AutoDL 上 on-the-fly 生成、不入 git。上机第一件事就是确认生成成功，否则 eval 链路空跑。

### 2.2 生产设计（不写进简历，但面试「真正线上」追问时要能报）

| 项 | 值 | 来源 |
|---|---|---|
| 冻结训练集 | **3200 条** = 领域 2880 + anchor 320 | `../../CLAUDE.md`（舆情五级金字塔） |
| anchor 混入 | 9:1（约 10%）防遗忘 | 同上 |
| 训练集极性配比 | negative ~55%（**数据构造富集比例**，与线上流量漏斗无关，严禁串号） | 同上 |
| 测试集 | 锁 **500 条纯人标金标**（死数字，不分约数）；标注一致性 κ≥0.7 | 同上 |

### 2.3 🟡 数据侧待实测回填（上 GPU 后用 `stat_seq_len.py` 拿）

- [ ] 平均输入长度 / P95 输入长度（`--tokenizer` 出 CoT 分位，顺带解锁 TODO 5.20）
- [x] **CoT 实测中位数 / P95** —— ✅ 已回填（2026-10-27）：73 条真 tokenizer 实测 p50=201/p95=276/p99=303，三档已重设 120/200/320（旧「177 tokens」为字符级近似口径的误读）。⚠️ 750 条正式数据需再复核（字符级粗估更长）
- [ ] 总 token 数（比样本数更关键，面试官爱问）
- [ ] 一个 epoch 有多少 step（脚本 `print_budget` 会直接打印，别自己算）

---

## 3. SFT 超参（✅ 已焊死，逐项可背）

| 项 | 值 | 来源 |
|---|---|---|
| 学习率 lr | **1e-4**（LoRA 口径，已由 2e-5 修正） | `train_sft_lora.py:24` |
| scheduler | Cosine + `warmup_ratio=0.05` | 同上 |
| 正式 Epoch | **1**（防过拟合 + 给 GRPO 留策略熵）；探针跑 3 只为看拐点 | `train_sft_lora.py:23` |
| 早停 | **不做自动早停**（SOP §训练纪律：epoch 受控 + cosine 退火） | `train_sft_lora.py:23` |
| max_seq_length | **2560**（SOP §6.3 P99 定值） | `train_sft_lora.py:209` |
| batch / 累积 | 探针 `--per_device_train_batch_size 2`；正式默认见 `TrainingArguments` | `train_sft_lora.py:34` |
| 损失重构 | 前缀差分掩码 + 字段级加权（hard_case 4.0× / sentiment·relevance 3.0× / risk 2.0× / evidence 0.4× / CoT 1.0×）＋有效 token 动态分母 | `sft_weight_planner.py` |

**面试话术锚点（global batch 公式）**：`有效 batch = micro_batch × grad_accum × 卡数`，脱口而出。
脚本会把「实际 batch / 每 epoch 步数 / 合计步数 / warmup 实际步数」全打屏（`train_sft_lora.py:653-677`），照抄即可。

---

## 4. GRPO 超参（✅ 已焊死，区分度最高）

| 项 | 值 | 来源 |
|---|---|---|
| 生成组数 G | **4**（代码注释：组内要有方差，4 是最小偶数组） | `train_grpo.py:377` |
| 采样温度 T | **0.7**（T=0 会退化成更贵的 SFT） | `train_grpo.py:378` |
| 学习率 lr | **2e-5** —— 🔴 关键词术：**≈ SFT 1e-4 的 1/5；全参 RL 的 1e-6~5e-6 不适用** | `train_grpo.py:379` |
| KL 系数 β | **0.04**（token 级常数 KL，管分布漂移；长度由 R_length 管，两者正交） | `train_grpo.py:380` |
| max_prompt_length | **2048**（= 2560 SFT − 512 completion） | `train_grpo.py:391` |
| max_completion_length | **512**（CoT 预算） | `train_grpo.py:392` |
| per_device batch | 4 × grad_accum 2 → 每步 8 题 × G4 = **32 轨迹/步** | `train_grpo.py:383-384` |
| Epoch | 1.0；warmup_ratio 0.03；max_grad_norm 1.0 | `train_grpo.py:385-387` |
| 优化器 | adam β1=0.9 / β2=0.99 / weight_decay=0.01 / cosine | `train_grpo.py:522-527` |
| advantage | **只减组内均值、不除 std**（`scale_rewards="none"`）；`loss_type="grpo"` 只决定 loss 分母，两者独立。推导见 GRPO 手册 §1.2 | `train_grpo.py` 文件头硬约束 3 + `_opt` |
| 架构 | **无 Critic / 无 RM / 4 条纯 Python 规则 Reward** | `train_grpo.py` 标题 |
| seed | 42 | `train_grpo.py:393` |

### 🔴 两个一碰就穿帮的坑

1. **clip ε 未显式设**：`GRPOConfig` 构造里没传 epsilon，走 TRL 默认值。
   面试被问「clip 多少、为什么」→ 别背 0.2，要说清「我们 reward 是纯规则可硬断言，ε 不参与微妙平衡」或去代码显式补上并记理由。**上 GPU 前二选一。**
2. **reference 必须冻结且是完整模型**：`train_grpo.py:26-28` 注释 —— GRPO 起点只能传 merge 后的完整模型目录，不能是 adapter，否则 TRL 算 KL 时 reference 退化成裸基座，KL 惩罚变成「往回忘掉 SFT」。这是碾压式深水区话术，面试主动甩。

### 四维 Reward（✅ 已焊死，值域 R ∈ [−2.5, +2.0]）

| 维 | 逻辑 | 值域 |
|---|---|---|
| R_format | Pydantic 合规 / `json.loads()`；门禁项只给 {0, −2.5}，**绝不给正分** | `reward_fns.py` |
| R_grounding | evidence = 原文字符级连续子串 Exact Match；门禁项只给 {0, −2.5} | 同上 |
| R_accuracy | 金标命中；漏报 = 误报 ×2（−2.0 vs −1.0），显式编码不对称 | 同上 |
| R_length | CoT 三档；只有「判对 + 溯源双通过」才激活 | 同上 |

单调铁律：`−2.5(协议/证据) < −2.0(危机漏报) < −1.0(误报) < 0(判对但CoT空壳) < +1.0~+2.0`（`reward_fns.py` 头部设计要点 5）。

---

## 5. 显存与吞吐（部分 ✅ / 部分 🟡）

### 5.1 显存

| 项 | 值 | 状态 |
|---|---|---|
| 4B bf16 权重 | ≈ 8G | ✅ `train_sft_lora.py:574` |
| 省显存手段 | gradient checkpointing（`use_reentrant=False` + `enable_input_require_grads()`）、bf16、LoRA 冻结基座、micro batch=1 + 累积 | ✅ `train_sft_lora.py:589-595` |
| gradient checkpointing 代价 | ~30% 速度，换 ~80% 激活显存 | ✅ 同处注释 |
| 🟡 实测峰值显存（4090/3090） | **__ GB/卡**（SFT / GRPO 各记一次，`nvidia-smi` 采样） | 🟡 待回填 |
| 🟡 GRPO / SFT 显存倍数 | **__ ×**（理论 1.5~2×：G 条序列 + reference 常驻 + 长序列激活三段叠加） | 🟡 待回填 |

> ✅ **好消息**：实验卡 4090/3090 与生产 A10 **同为 24GB**，所以「第 5 个 micro-batch 顶穿 23.5G OOM」
> 「必须开 gradient checkpointing」这类显存结论**可以直接平移到 A10 生产环境**，不用打折。

**面试话术锚点（OOM 真实素材，主动讲）**：`train_sft_lora.py:574-595` 记录了三处真实踩坑——
① 24G 卡 batch=1 + max_len=1024，第 ~5 个 micro-batch 顶穿 23.5G OOM，定位到 dropout；
② 冻结基座 + LoRA + gradient checkpointing 的静默假训练坑，必须 `use_reentrant=False` + `enable_input_require_grads()` 两步配套；
③ 漏掉任一步的症状是「trainable params 正常、loss 微降、推理输出还是基座原样」——这正是上轮 33 步后格式没学会的头号嫌疑。

### 5.2 算力账本 🔴 三个必须先裁决的口径问题

**坑 A：单卡吞吐口径串号。** TODO 4.3 写「单卡 **5~7 QPS**，8 卡 40~56」，但当前权威源
`SFT负面精判_MVP设计架构.md:53` 是「单卡 **~10 QPS** 级」；5~7 只存在于 `raw/` 与
`notes/archive/` 的旧版。**两套不能同时出现在面试里。** 建议统一按 10（当前权威源）。

**坑 B：账本基数没咬死（「条」vs「次调用」）。**

| 基数 | 日均量 | 均值 QPS | 峰值 3× |
|---|---|---|---|
| 按「条」 | 近百万条 | **11.5** | **34.5** |
| 按「次调用」（含 1.2~1.5× 主体 fan-out） | 110~170 万次 | **12.7~19.7** | **38~59** |

**能自圆其说的推导链（推荐这套说辞）**：

```
日均近百万条 → 11.5 QPS 均值 → 峰值比 3× → 34.5 QPS
单卡 INT4+vLLM ~10 QPS → 基线 4 张 = 40 QPS ≥ 34.5  ✅ 刚好覆盖
fan-out 上浮至 170 万次 → 峰值 59 QPS → 弹性扩到 8 张 = 80 QPS ✅
→ 正好落进 CLAUDE.md 的「~4 张基线 + 双机容灾冗余 → 4~8 张弹性区间」
```

> 🔴 **前提**：单卡必须按 10 QPS。若按 5~7，8 卡 = 40~56，覆盖 59 就**差一口气**，自洽性崩。

**坑 C：4090/3090 的 wall-clock 不能当 A10 生产数字报。**
探针实测的「训练耗时 X 分钟」是 **4090 的数字**，A10 FP16 ≈ 4090 的 0.6~0.75、
3090 ≈ 4090 的 0.4~0.5。面试若被问「生产要跑多久」，正确说法是：
「探针在 4090 上 X 分钟；生产是 A10，按算力折算约 Y 小时」——**折算系数说出口，才算真跑过**。

| 项 | 值 | 来源 |
|---|---|---|
| 端到端延迟 | 稳态 2~3s；高危特征规则层直报 ≤1.5s | ✅ `MVP设计架构.md:53` |
| 全链路 SLA | ≤5s | ✅ 同上 |
| L1 粗筛 | TorchScript JIT 2,500+ QPS | ✅ `MVP设计架构.md:21` |
| 🟡 单卡实测 QPS（4090，压测） | **__ QPS** —— 决定上面整套账本；A10 需按算力折算 | 🟡 待回填 |
| 🟡 训练总时长 | SFT 正式轮设计 3~6 小时（`TODO_SFT探针快速启动.md:114`，A10 口径）；探针 33 步 2~3 分钟 | 🟡 待实测（注明卡型） |

### 5.3 训练工程细节（TODO 4.3 Q1 延伸，易被追问）

| 项 | 值 | 来源 |
|---|---|---|
| optimizer | AdamW（β1=0.9 / β2=0.99 GRPO 侧；weight_decay 0.01） | `train_grpo.py:522-527` |
| warmup | SFT ratio 0.05；GRPO ratio 0.03（脚本打实际步数） | 两脚本 |
| packing | **未启用**——数据已预 tokenize（`input_ids/labels/loss_weights`），必须跳过 SFTTrainer 自带 prepare，否则二次处理 | `train_sft_lora.py:343-346` |
| 超长处理 | max_seq_length=2560（SOP §6.3 P99 定值）截断；GRPO prompt 上限 2048（实测 max 774，超则截原文） | `train_sft_lora.py:209`、`train_grpo.py` |
| best ckpt | **无早停、不按 metric 选 best**；靠 epoch 受控 + cosine 退火 | `train_sft_lora.py:23` |
| seed | 42 | `train_grpo.py:393` |

---

## 6. 结果三行对比（🟡 全部待实测，实验后填死）

> 面试必问三行：**Base → SFT → SFT+GRPO**。探针实验跑完逐格填，指标按任务选。

| 指标 | Base | SFT | SFT+GRPO |
|---|---|---|---|
| 🟡 判别准确率 / F1 | | | |
| 🟡 格式合规率（目标解析成功率 ≥99.8% 是设计值） | | | |
| 🟡 mean reward（及四维分项走势） | — | — | |
| 🟡 平均输出长度变化（GRPO 通常会变长，要能解释） | | | |
| 🟡 CoT 三档落档分布 | | | |
| 🟡 漏报 / 误报成对（`pa_fp_pair`） | | | |

**训练稳定性监控（🟡 待回填，用来证「没崩」）**：
- [ ] KL 走势
- [ ] entropy 走势
- [ ] clip fraction
- [ ] 跑了几个 seed / 方差多大
- [ ] 崩没崩过、怎么救的（OOM 见 §5，是真素材）

---

## 7. 面试速查 10 秒版（背这 6 个就够开场）

1. **基座**：Qwen3-4B-Instruct，bf16，LoRA r16/α32。实验 4090/3090 24GB，生产 A10 24GB。
2. **数据**：探针 81 SFT + 51 GRPO；生产 3200（2880 领域 + 320 anchor，9:1 混入）。
3. **SFT**：lr 1e-4，epoch 1，max_len 2560，前缀掩码 + 字段加权（4/3/2/0.4×）。
4. **GRPO**：G=4，T=0.7，lr 2e-5（=SFT 1/5），β=0.04，无 Critic，纯规则 Reward。
5. **Reward**：四维 R∈[−2.5,+2.0]，门禁短路 −2.5 重于漏报 −2.0。
6. **产出**：65~105 tokens（从 ~180 压降），A10 单卡 ~10 QPS，端到端 2~3s。

> ⚠️ 报 QPS/耗时前先自问一句：**这是 4090 的实测，还是 A10 的生产口径？** 两者不能混。

---

## 8. 待办依赖链（实验时的执行顺序）

```
2.4 SFT 复跑(lr=1e-4) ──> stat_seq_len.py --tokenizer 拿 CoT 分位
                                  │
                                  ├──> 5.20 ✅ 已回填 R_length 三档阈值(40/65/120 → 120/200/320)
                                  │
2.5 GRPO rollout+训练 ──> 填 §4/§6 所有 🟡 格
                                  │
                                  └──> 确认 grpo_eval_probe.jsonl 生成成功(§2.1 坑)
```

> 改动三档阈值时，四件套要联动：`reward_fns.py` 常量 + `test_reward_fns.py` 用例 + 数据准备手册 §五 + GRPO 手册 §2.5（见 `../../CLAUDE.md` 四件套互引约定）。
