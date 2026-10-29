# 抢救 RL 金标数据集 · 统计与验收报告（rl_salvaged_golden）

> **★ 最终交付文件：`rl_salvaged_golden.clean2.jsonl`（1404 条，Schema v1 格式）——接 GRPO 用这版**
> 原始入库池 `rl_salvaged_golden.jsonl`（1572 条）是中间产物；clean2 = 原始池经 §四去重
> （Jaccard 对 Round-2 训练集 + 血缘校验，剔 168）+ SimHash 补刀（剔 0）后的终版。
> **生产脚本**：`scripts/salvage_dropped_to_rl.py` ｜ **验收脚本**：`scripts/verify_salvage_rl.py`
> **验收日期**：2026-10-29（clean2 终版复验 PASS，`verify_report_clean2.json`）｜ **规则底稿**：`scripts/README_SALVAGE_RL.md`

---

## 一、终版数据链路（2026-10-29）

```
rl_salvaged_golden.jsonl (1572, 四模型投票+四查入库池)
  → §四.4 Jaccard 对 round2_train.jsonl（merge_cross_dedup.py）
      剔 dup 134 + 同文不同标 conflict 9 + 血缘不全 25 → clean 1404
  → §四.4b SimHash 补刀（Hamming≤6）→ 剔 0 → clean2 1404
  → verify_salvage_rl.py 复验：投票门禁 1404/1404 PASS（rc=0）
```

与 SFT 训练集的交叉实测：255 条同 event_id（97 条同文不同标，含 16 条 sentiment/hard_case_type
硬冲突）——全部被去重剔除，clean2 对 `round2_train.jsonl` 残留交叉 **0**；对 500 金标
`seed500_final.jsonl` 无泄漏（最大 3-gram Jaccard 0.1761）。

---

## 二、验收结论：≥3 票投票门禁 1404/1404 全部通过 ✅

对每条样本，`gold` 四维逐一与 `meta.field_votes` 票型比对（`verify_report_clean2.json`）：

| 维度 | 4 票共识 | 3:1 多数 | <3 票违规 |
|---|---|---|---|
| **sentiment** | 1189 | 215 | **0** |
| **hard_case_type** | 1098 | 306 | **0** |
| **relevance** | 1305 | 99 | **0** |
| **risk_categories** | 每个 Key ≥3 票 | — | **0** |

附带通过：evidence ⊆ input 严格子串 1404/1404；output 尾部 JSON ≡ gold（Schema v1 `assert_valid`）1404/1404。

选席模式：`exact_consensus` 1368 条（四维全匹配席原装 CoT+JSON）+ `risk_trimmed` 36 条（仅裁剪 risk 至共识 Key）。

---

## 三、标签分布（极性 × 其他维度 联合分布，n=1404）

**极性 × 难点类型**：

| 极性 \ 难点 | comparative | counterfactual | explicit | fact_implicit | irony | metaphor | multi_subject | 合计 |
|---|---|---|---|---|---|---|---|---|
| negative | 48 | 68 | 86 | 27 | 102 | 35 | 17 | **383** |
| neutral | 62 | 68 | 8 | 540 | 2 | 17 | 155 | **852** |
| positive | 108 | 15 | 10 | 2 | 18 | 10 | 6 | **169** |
| **合计** | 218 | 151 | 104 | 569 | 122 | 62 | 178 | **1404** |

> 直观读法：negative 主力是 irony(102)/explicit(86)/counterfactual(68)——难例集中区；neutral 的
> 绝对主力是 fact_implicit(540)——客观陈述/合规通报天然中性（占该难点 95%）；comparative 的
> positive(108) 近半——「被比好放行」是对比难点的正向主形态。

**极性 × 语篇地位**：

| 极性 \ 语篇 | core | passing | foil | 合计 |
|---|---|---|---|---|
| negative | 362 | 15 | 6 | 383 |
| neutral | 662 | 172 | 18 | 852 |
| positive | 81 | 5 | 83 | 169 |
| **合计** | 1105 | 192 | 107 | **1404** |

> 直观读法：negative/neutral 以 core 为主（362/662）；passing 里 neutral 占绝对多数(172)——顺带
> 提及天然中性；foil 里 positive 最多(83)——拿好评反衬他方出事，是 foil 的典型写法。

**单维边际**：sentiment = neutral 60.7% / negative 27.3% / positive 12.0%；
relevance = core 78.7% / passing 13.7% / foil 7.6%。
（⚠️ RL 池极性与 SFT 训练集 48/31/21 **刻意不同构**：抢救源是「目标未命中」弃样，天然中性多；
RL 靠 reward 信号训练、不靠先验配比，此分布非缺陷，不要与训练集配比对齐。）

**极性 × 风险Key**（多标签计次；空表=无风险标签，positive 全部空表）：

| 极性 \ 风险Key | product | service | marketing | compliance | finance | general_negative | governance | 空表 | 合计(条) |
|---|---|---|---|---|---|---|---|---|---|
| negative | 144 | 171 | 128 | 78 | 13 | 46 | 15 | 16 | 383 |
| neutral | 273 | 181 | 151 | 212 | 70 | 1 | 23 | 209 | 852 |
| positive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 169 | 169 |
| **Key计次** | 417 | 352 | 279 | 290 | 83 | 47 | 38 | 394 | 1404 |

> 直观读法：negative 风险集中在 service(171)/product(144)/marketing(128)——售后、质量、营销三大
> 来源；neutral 里 compliance(212) 突出——合规通报/监管披露是典型中性事实；positive 169 条
> **全部空表**——好评零风险标签，符合「宁可窄勿宽」法典口径。

**极性 × 行业**：

| 极性 \ 行业 | automotive | beauty_cosmetics | electronics_3c | finance_insurance | fmcg | food_catering | healthcare_pharma | logistics_express | online_edu | real_estate | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| negative | 30 | 47 | 41 | 24 | 33 | 39 | 32 | 46 | 44 | 47 | 383 |
| neutral | 90 | 82 | 83 | 75 | 81 | 79 | 109 | 88 | 86 | 79 | 852 |
| positive | 20 | 21 | 25 | 16 | 19 | 9 | 10 | 16 | 11 | 22 | 169 |
| **合计** | 140 | 150 | 149 | 115 | 133 | 127 | 151 | 150 | 141 | 148 | **1404** |

> 直观读法：10 行业体量均衡（115~151，全在 8.2%~10.8%）；healthcare_pharma 的 neutral(109) 最多
> ——医药公告天然客观；food_catering/online_edu 的 positive(9/11) 偏少——餐饮/教培好评稀缺符合业务直觉。

---

## 四、投票与选席情况

- **票型**：consensus_4_0 = 1189 (84.7%) ｜ majority_3_1 = 215 (15.3%) ｜ 平票/分散 = 0
- **选席模式**：exact_consensus 1368 ｜ risk_trimmed 36
- **output 来源席位**（`chosen_model`）：kimi-k3 420 (29.9%) ｜ gpt-6-sol 382 (27.2%) ｜ deepseek-v4-pro-0813 346 (24.6%) ｜ glm-5.3 256 (18.2%)

> A/B/C 三席带历史三审遗产（pred_a/b/c 预先存在），D 席为本轮新打标——份额差异来自「有历史
> 判定即可入围抽选」的机制，非选席偏差。

各席原始 sentiment 判定分布（n=1404）：

| 席位 | negative | neutral | positive |
|---|---|---|---|
| A deepseek-v4-pro | 404 | 834 | 166 |
| B glm-5.3 | 377 | 856 | 171 |
| C gpt-6-sol | 378 | 861 | 165 |
| D kimi-k3 | 409 | 827 | 168 |

---

## 五、终稿(gold) vs 各席原始判定 Cohen's κ（n=1404）

| 维度 | deepseek-v4-pro (A) | glm-5.3 (B) | gpt-6-sol (C) | kimi-k3 (D) |
|---|---|---|---|---|
| sentiment | 0.945 (97.0%) | 0.898 (94.4%) | 0.929 (96.2%) | **0.947** (97.1%) |
| relevance | 0.978 (99.2%) | 0.888 (96.0%) | 0.953 (98.4%) | **0.982** (99.4%) |
| hard_case_type | 0.975 (98.1%) | 0.835 (87.6%) | 0.936 (95.0%) | **0.968** (97.9%) |
| risk_categories* | 0.870 | 0.791 | 0.833 | **0.885** |

\* risk 为多标签集合，κ 按逐 Key one-vs-rest 取平均；括号内为一致率。

> **读法**：kimi-k3（D 席，v2.2 法典最新口径本轮新打标）四维全面最高；deepseek-v4-pro 紧随其后
> （再现金标 500 重建报告的「首席法官」效应）；glm-5.3 在 hard_case/risk 上最弱——与金标 500
> 报告中 B 席的历史观察同构。全部 κ ≥ 0.79，极性 ≥ 0.89，终版数据集内一致性高。

---

## 六、复现与审计

```bash
# 去重（复用跨库脚本，见 scripts/README_SALVAGE_RL.md §四.4 / §四.4b）
python3 scripts/merge_cross_dedup.py \
    --src output/salvage_dropped_to_rl/rl_salvaged_golden.jsonl \
    --ref output/sft_round2/round2_train.jsonl \
    --out output/salvage_dropped_to_rl/rl_salvaged_golden.clean.jsonl \
    --conflict_out output/salvage_dropped_to_rl/golden.cross_conflict.jsonl \
    --dup_out output/salvage_dropped_to_rl/golden.dup_removed.jsonl \
    --report output/salvage_dropped_to_rl/golden_dedup_report.json
python3 scripts/simhash_dedup.py \
    --src output/salvage_dropped_to_rl/rl_salvaged_golden.clean.jsonl \
    --out output/salvage_dropped_to_rl/rl_salvaged_golden.clean2.jsonl \
    --dropped_out output/salvage_dropped_to_rl/golden.simhash_dropped.jsonl --max_dist 6

# 验收（纯离线，退出码 1 = 有违规）
python3 scripts/verify_salvage_rl.py --src output/salvage_dropped_to_rl/rl_salvaged_golden.clean2.jsonl \
                                    --out output/salvage_dropped_to_rl/verify_report_clean2.json
```

**数据血缘**：每条 meta 含 `field_votes`（各维票型）、`pred_a/b/c/d`（四席原始判定）、
`chosen_seat/chosen_model`（output 来源席）、`salvage_source`（原废弃原因），可完整追溯投票过程。
去重中间产物（conflict / dup_removed / simhash_dropped / clean 版）归档在 `archive/`。

---

## 附：历史快照（2026-10-28，已过时仅留档）

本 README 最初统计的是当时入库的 1120 条（4:0=956/3:1=164；生产漏斗候选 1756 → 排重跳过 1130
→ 元评论拦截 74 → 四席打标 609 → 净入库 1120）。此后追加打标至 1572 条，再经 2026-10-29 去重
收敛为终版 1404 条；上文各表均为终版口径。
