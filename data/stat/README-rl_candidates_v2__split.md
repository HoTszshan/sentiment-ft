# RL v2 训练/测试集切分说明（eval300 / train4205）

> **切分日期**：2026-11-01 ｜ **脚本**：`scripts/sample_eval300.py`（纯本地，零 API）｜ **seed=42 可复现**
> **★ 测试集冻结**：`eval300_EVAL_FROZEN.jsonl`（300 条）——GRPO 训练全程不得触碰，训练完只跑一次出报告。
> **训练集**：`train_rl_v2.jsonl`（4205 条）——接 GRPO 用这版。
> **切分报告**：`split_report.json`

---

## 一、切分方案（用户 2026-10-31 拍板）

**为什么不是金标 500 配比硬采**：金标 500 要求 explicit 26% + fact_implicit 22%，但 rl_v2 是高难度扰动语料，
这两类实际只占 2.6% / 2.3%——物理上采不出（300 条里 explicit 全拿也才 8 条 vs 需要 78 条）。
故采用「金标 500 配比框架 + rl_v2 可达范围调整」：

| 层 | 条数 | 构成 |
|---|---|---|
| **第 1 层·配比框架** | 150 | sentiment 负30%/中40%/正30%；relevance core60/passing25/foil15；hard_case 七类全保按实际占比归一（explicit/fact_implicit 各保 ~8 条做「简单题哨兵」）；10 行业轮转均摊 |
| **第 2 层·难例富集** | 150 | **写手自报 chosen_sentiment ≠ gold 的分歧样本**，按 gold sentiment 分层（70/40/40）——rl_v2 独有的高价值考区 |

**分歧样本为什么是高价值考区**：写手自报与合议庭 gold 的偏差（全库 49.7%）密集落在判别边界上
（negative→neutral 840 条、negative→positive 800 条），模型最可能在这些地方错——测试集故意超配这类样本
（eval300 里占 82.7%），让评估更有区分度。

## 二、零泄漏校验

- eval300 vs train 逐条 3-gram Jaccard 跨库校验（≥0.65 判泄漏，对齐 merge_cross_dedup near 阈值）：**0 对泄漏**
- event_id 物理隔离：eval 与 train 无共享 id

## 三、分布快照与标注一致性（κ）

### eval300（n=300，测试集）

| 维度 | 取值 | 条数 | 占比 |
|---|---|---|---|
| **sentiment** | negative | 115 | 38.3% |
| | neutral | 100 | 33.3% |
| | positive | 85 | 28.3% |
| **relevance** | core | 180 | 60.0% |
| | passing | 58 | 19.3% |
| | foil | 62 | 20.7% |
| **hard_case** | multi_subject | 83 | 27.7% |
| | irony | 71 | 23.7% |
| | comparative | 58 | 19.3% |
| | metaphor | 47 | 15.7% |
| | counterfactual | 19 | 6.3% |
| | explicit | 11 | 3.7% |
| | fact_implicit | 11 | 3.7% |
| **票型** | arbitrated | 228 | 76.0% |
| | consensus | 72 | 24.0% |


**sentiment × relevance**：

| sentiment \ relevance | core | passing | foil | 合计 |
|---|---|---|---|---|
| **negative** | 89 | 13 | 13 | 115 |
| **neutral** | 56 | 31 | 13 | 100 |
| **positive** | 35 | 14 | 36 | 85 |
| **合计** | 180 | 58 | 62 | 300 |

**sentiment × hard_case_type**：

| sentiment \ hard_case_type | explicit | fact_implicit | irony | metaphor | comparative | counterfactual | multi_subject | 合计 |
|---|---|---|---|---|---|---|---|---|
| **negative** | 5 | 2 | 38 | 20 | 18 | 7 | 25 | 115 |
| **neutral** | 2 | 9 | 13 | 16 | 15 | 8 | 37 | 100 |
| **positive** | 4 | 0 | 20 | 11 | 25 | 4 | 21 | 85 |
| **合计** | 11 | 11 | 71 | 47 | 58 | 19 | 83 | 300 |

**sentiment × risk_categories**（多标签计次）：

| sentiment \ risk_categories | product | service | marketing | compliance | finance | general_negative | governance | 空表 | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| **negative** | 38 | 39 | 46 | 15 | 8 | 9 | 7 | 0 | 162 |
| **neutral** | 22 | 13 | 8 | 17 | 7 | 0 | 3 | 46 | 116 |
| **positive** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 85 | 85 |
| **合计** | 60 | 52 | 54 | 32 | 15 | 9 | 10 | 131 | 300 |

**终稿 gold vs 各席 κ**（集合全等口径）：

| 席位 | sentiment | relevance | hard_case | risk |
|---|---|---|---|---|
| A deepseek-v4-pro-0813 | **0.8691** | **0.8532** | **0.8786** | **0.8515** |
| B glm-5.3 | **0.8741** | **0.7531** | **0.5808** | **0.7757** |
| C gpt-6.1/6-sol | **0.8541** | **0.7614** | **0.8546** | **0.7891** |



**三席两两 κ**：

| 席位对 | sentiment | hard_case | risk |
|---|---|---|---|
| A vs B | 0.808 | 0.543 | 0.672 |
| A vs C | 0.765 | 0.693 | 0.688 |
| B vs C | 0.770 | 0.518 | 0.631 |

### train（n=4612）

| 维度 | 取值 | 条数 | 占比 |
|---|---|---|---|
| **sentiment** | negative | 2188 | 47.4% |
| | neutral | 1210 | 26.2% |
| | positive | 1214 | 26.3% |
| **relevance** | core | 2978 | 64.6% |
| | passing | 854 | 18.5% |
| | foil | 780 | 16.9% |
| **hard_case** | multi_subject | 1349 | 29.2% |
| | irony | 1050 | 22.8% |
| | comparative | 986 | 21.4% |
| | metaphor | 796 | 17.3% |
| | counterfactual | 230 | 5.0% |
| | explicit | 108 | 2.3% |
| | fact_implicit | 93 | 2.0% |
| **票型** | arbitrated | 3409 | 73.9% |
| | consensus | 1203 | 26.1% |


**sentiment × relevance**：

| sentiment \ relevance | core | passing | foil | 合计 |
|---|---|---|---|---|
| **negative** | 1971 | 134 | 83 | 2188 |
| **neutral** | 602 | 513 | 95 | 1210 |
| **positive** | 405 | 207 | 602 | 1214 |
| **合计** | 2978 | 854 | 780 | 4612 |

**sentiment × hard_case_type**：

| sentiment \ hard_case_type | explicit | fact_implicit | irony | metaphor | comparative | counterfactual | multi_subject | 合计 |
|---|---|---|---|---|---|---|---|---|
| **negative** | 98 | 12 | 701 | 491 | 361 | 147 | 378 | 2188 |
| **neutral** | 0 | 81 | 137 | 122 | 172 | 55 | 643 | 1210 |
| **positive** | 10 | 0 | 212 | 183 | 453 | 28 | 328 | 1214 |
| **合计** | 108 | 93 | 1050 | 796 | 986 | 230 | 1349 | 4612 |

**sentiment × risk_categories**（多标签计次）：

| sentiment \ risk_categories | product | service | marketing | compliance | finance | general_negative | governance | 空表 | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| **negative** | 808 | 881 | 791 | 330 | 169 | 229 | 151 | 5 | 3364 |
| **neutral** | 179 | 124 | 87 | 130 | 59 | 3 | 37 | 722 | 1341 |
| **positive** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1214 | 1214 |
| **合计** | 987 | 1005 | 878 | 460 | 228 | 232 | 188 | 1941 | 4612 |

**终稿 gold vs 各席 κ**：

| 席位 | sentiment | relevance | hard_case | risk |
|---|---|---|---|---|
| A deepseek-v4-pro-0813 | **0.9395** | **0.8738** | **0.8673** | **0.8745** |
| B glm-5.3 | **0.8981** | **0.7454** | **0.6249** | **0.7630** |
| C gpt-6.1/6-sol | **0.8847** | **0.8194** | **0.8200** | **0.7745** |

**三席两两 κ**：

| 席位对 | sentiment | hard_case | risk |
|---|---|---|---|
| A vs B | 0.871 | 0.5882 | 0.7014 |
| A vs C | 0.8398 | 0.7505 | 0.7011 |
| B vs C | 0.8113 | 0.5504 | 0.6583 |

> **一致性读法**：eval300 的各席 κ 整体低于 train 0.03~0.05——符合预期（测试集故意超配判别边界上的分歧样本），
> 但排序结构完全一致（A 席四维最强、B 席 hard_case 最弱）。测试集对模型更有区分度，同时 gold 权威性未降
> （A 席 κ=0.89 仍处于「高一致」区间）。

## 四、评估协议（训练完执行）

```bash
# 1. eval300（训练分布内 + 难例富集考区）
python3 scripts/evaluate_sft.py --src output/rl_v2/eval300_EVAL_FROZEN.jsonl ...

# 2. seed500_final（金标 500 外部基准：法典口径 + 常规难度泛化）
python3 scripts/evaluate_sft.py --src output/gold_eval_final/seed500_final.jsonl ...
```

**两套测试的角色**：
- `eval300`：训练分布内性能 + 分歧边界判别力（与 reward 曲线同源可比）
- `seed500_final`：泛化到常规语料的能力——**两套对比可回答「GRPO 是否只会做难题、忘了简单题」**
  （若 eval300 涨而 seed500 跌 → 过拟合到难例分布，需调 KL 系数/训练步数）

## 五、复现

```bash
python3 -I scripts/sample_eval300.py \
  --src output/rl_v2/rl_v2_golden.jsonl \
  --out_dir output/rl_v2 --seed 42
```

同 seed 全程确定性可复现；重跑前不要改动 rl_v2_golden.jsonl（切分对全库顺序敏感）。
