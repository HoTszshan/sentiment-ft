# 落地改造清单（舆情负面精判 · SFT 探针）

> **定位**：只记**确定要落地**的改造项。需要跑实验才能决定的事不放这里（见
> `docs/experiments/TODO.md`）；已完成的改动留在 git history 与实验报告里。
> **最近更新**：2026-10-27

---

## 〇、当前状态快照

| 维度 | 数值 | 门槛 | 判定 |
|---|---|---|---|
| 思考标签闭合率 | **100%** | 100% | ✅ |
| JSON 合法率 | **100%** | ≥99.8% | ✅ |
| evidence 严格子串率 | **100%** | 100% | ✅ |
| gate4（格式） | **PASS** | — | ✅ |
| gate2（安全字段） | irony 召回 **0/1** | ≥90% | 探针期记 N/A |
| sentiment Macro-F1 | **90.48%** | — | 良好 |
| hard_case Macro-F1 | **43.33%** | — | 待提升 |
| risk Subset Accuracy | **87.5%** | ≥90% | 接近 |

格式问题已解决，瓶颈在内容判别（irony / service）。

---

## 一、防回退清单（已修，勿改回）

只列结论；症状、根因与实测证据见 `experiments/2026-10-24_POC1_协议token解冻-vs-开口播种.md`。

| 已修项 | 若改回会怎样 |
|---|---|
| `compute_loss` 只在训练态除以梯度累积步数 | `eval_loss` 虚低 4 倍 |
| 除数用 `current_gradient_accumulation_steps` 而非固定 `gas` | epoch 末步 loss 假跌 1/4，该步梯度也只有 1/4 |
| `LoraConfig(trainable_token_indices=[151667,151668])` | `<think>`/`</think>` 输出行冻结，格式永远学不会 |
| canary `p_json` 读真实下一 token id（勿写死 `ord("{")`=90） | 实际 token 是 `'{"'`=4913，读数恒 0 |
| `parse_output` 解析前剥离 `<\|im_end\|>` / `<\|endoftext\|>` | 尾赘率恒 100% |
| gate2 在预测全部不可解析时记 `N/A` | 格式全崩的版本会显示安全字段 PASS |

---

## 二、待落地

### L1 · 扩充 irony / service 样本，扩验证集

- **现状**：训练 73 条中 irony 仅 8 条；验证 8 条中 irony 仅 1 条。
- **要做**：irony / service 每类 ≥20 条；验证集扩到 ≥40 条。
- **为什么必须**：8 条验证集下，单条样本就能让召回率在 0% / 100% 之间跳变，指标没有统计意义。
- **验收**：irony 召回 ≥90%，且验证集支撑样本数足够参与判定。

### L2 · 恢复 gate2 对 irony 的判定

- **依赖**：L1 完成。
- **现状**：探针期 `gate2 irony_recall` 视为 N/A，不参与判定。
- **要做**：数据到位后恢复判定，不再跳过。
- **验收**：正常轮次下 gate2 的 `judged_fields` 包含 `irony_recall`。

### L3 · 重新引入 anchor 数据（协议对齐版）

- **现状**：9 条 COIG-CQIA anchor 已从训练集剔除（82 → 73）。
- **原因**：原 9 条 **0 条**带 `<think>`、**7 条**连 JSON 都解析不开，输出是「是」「地名」「新闻标题」，
  在教模型「不要 think、答案可以是一个裸词」。
- **要做**：用同一套 6 字段协议重写 anchor 的 output 后再混入；正式轮启用（防灾难性遗忘）。
- **⚠️ 约束**：若同时开启前缀预置（`--think_prefix 1`），会在 prompt 塞 `<think>\n`
  而金标 output 直接是 JSON，造出「开了口却不闭合」的畸形监督。两者不可同时开。
- **触发条件**：正式轮。

### R · r_accuracy 的 risk/hard_case 严格集相等疑过严——待探针实测后决策

- **现状**：`reward_fns.r_accuracy` 中 `risk_ok = (g_rk == p_rk)` 是 set 严格相等，多打/少打一类 = 全题「普通判错 -1.0」+ `accuracy_pass=False`（连锁：r_length 置 0、不进 k）。hard_case 同理直接 -1.0。
- **质疑依据**：打标一致性数据（`output/eval/kappa_v22/`）——仲裁稿上 risk κ 仅 0.511~0.769、hard_case 0.693~0.848，**裁判自身意见分歧大，却按零容忍罚模型**，训练信号混入裁判口味噪声；GRPO 组内常只差 risk 一项，噪声直接决定优势方向。
- **候选改法（未实施）**：risk 改 Jaccard 部分命中——命中过半 (≥0.5) 时从「-1.0 普通判错」降为「R_CORRECT_FULL 上扣 -0.5 档」，**accuracy_pass 门票语义不动**（k/r_length 不受污染）；结构性错误（partial<0.5 / 空表错打 / 金标非空全漏）维持 -1.0。sentiment 不动（κ 0.93+）。hard_case 观察后再定（有 7 类裁决链兜底）。
- **⚠️ 契约改动流程**：须先落真理源（`SFT负面精判_MVP设计架构.md` + 数据手册 §五）再改代码，`assert_contract_invariants` 单调链同步加档；-0.5 档位值与 0.5 阈值为设计值，待拍板。
- **决策实验（AutoDL 上做）**：跑 50 条探针 → `summarize_buckets` 看 risk/hard_case 难例在 keep 桶的分布。risk 难例未被系统性 drop → 现状可容忍，推迟到正式 3000 条；被误杀 → 按候选改法先改文档再动代码。

### P · 推理侧加速（探针 / eval 熔断）——默认不实施

- **对象**：`run_probe`（逐条逐次生成，50×4≈200 条）与 `_eval_reward_mean`（训练内逐条 batch=1，每次 eval 全量重跑）。
- **方向（按性价比序）**：① eval 降频/固定子集（最便宜，熔断 2% 阈值本就容忍噪声）；② `_eval_reward_mean` 批量生成（decoder-only 须 left padding，注意 truncation 联动）；③ 换 vLLM sampler（`run_probe` 的 sampler 是注入契约，写 `VllmSampler` 主循环零改动，3000 条正式筛题时再写）。
- **决策口径**：探针是一次性诊断，几分钟可接受；eval 是否拖慢训练**无实测数据**——先跑通真实训练，拿「eval 占总时长」实测：**<10% 永久不做，>30% 按 ①→② 动手**，每次只改一处并单测对拍。
- **不提前做的理由**：两段代码的价值在「简单可信」，加速引入的 padding 方向/采样语义/新依赖出错面，排查成本远超省下的分钟级时间。

### K · GRPO R_length 长度阈值按实测分位校准（40/65/120 → ~120/200/320）

- **现状**：`lib/reward_fns.py` 里写死 `COT_SHORT=40, COT_IDEAL_HI=65, COT_DEAD=120`。
- **实测冲突**：`stat_seq_len.py` 真分词统计（73 条领域训练集）显示 CoT 真实分位为：
  - `p50 = 201 token`、`p95 = 276`、`p99 = 303`、`max = 303`
  - 导致当前数据在 R_length 判定下 **100% 落在 `≥120` 归零档**（过短/满分/衰减三档全为 0 条）。
- **后果**：进入 GRPO 训练后，无论模型三跳推理写得多好，`R_length` 算出来恒等于 0，长度塑形维度彻底失效（虽不影响判别正确性，但丧失了卡空壳推理与压缩冗长 CoT 的能力）。
- **要做**：
  1. 将阈值重对齐实测分布，建议设为：`COT_SHORT=120`（过短罚）、`COT_IDEAL_HI=200`（满分带对准 p50）、`COT_DEAD=320`（归零对准 max 上限余量）；
  2. 阈值修改需同步更新 `lib/reward_fns.py`，并回归单测 `unit/test_reward_fns.py`（确保单调链不变量 `assert_contract_invariants` 仍然全绿）；
  3. 文档同步更新：`MVP/notes/舆情负面精判_GRPO数据准备实战手册.md` §五 Reward 契约表格。
- **触发条件**：启动 GRPO 真实训练前（SFT 阶段不调 reward_fns，不影响当前 SFT 训练）。

---

## 三、参考

- 实验待跑项：`docs/experiments/TODO.md`
- 已评估但暂不实施的方案：实验报告末的「附录 B」
- 历史基线与口径对照：实验报告「附：复现命令」与各 run 目录
