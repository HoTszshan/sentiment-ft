# RL v2 金标数据集 · 合并去重与统计验收报告（rl_v2_golden）

> **★ 最终交付文件：`rl_v2_golden.jsonl`（4912 条，RL 训练四键契约 `{instruction, input, output, meta}`）——接 GRPO 用这版**
> 来源：`rl_v3_batch1~7` 七批 golden（5144 条）经 **三重去重**（Jaccard → SimHash → 百炼 embedding 语义聚类）合并。
> **生产脚本**：`scripts/merge_rl_v2.py`（合并+去重）｜ **统计脚本**：`scripts/stats_rl_v2.py` ｜ **切分脚本**：`scripts/sample_eval300.py`
> **报告日期**：2026-11-02 ｜ **去重口径**：Jaccard ≥0.80 / SimHash Hamming ≤6 / cosine >0.85 簇内采样（对齐 salvage 链路）
> 上游链路文档：`scripts/README_RL_V3.md`（生成/打标全流程）｜ 打标机制底稿：`scripts/README_SALVAGE_RL.md`

---

## 一、数据链路与漏斗

```
rl_v3_batch1/golden.jsonl  (146)  ┐
rl_v3_batch2/golden.jsonl  (597)  │
rl_v3_batch3/golden.jsonl  (592)  │
rl_v3_batch4/golden.jsonl  (597)  ├→ 5144 条（七批合并）
rl_v3_batch5/golden.jsonl  (1186) │     batch7 为 GRPO 评测后按考卷盲区定向补题
rl_v3_batch6/golden.jsonl  (1595) │     （victim_passing 240 / cf_narrow_escape 160 / cross 40）
rl_v3_batch7/golden.jsonl  (431)  ┘
  → event_id 同文真重复 剔 1；同 id 不同文本加 #2 后缀保留 143（batch5/6 同 seed 撞车，不误杀）
  → 阶段1 Jaccard 3-gram（≥0.80）             剔 0   （保留池最大相似度 0.1702）
  → 阶段2 SimHash 64-bit（Hamming ≤6）        剔 0
  → 阶段3 百炼 text-embedding-v4 语义聚类      剔 231（4912 簇：孤立 4751 + 重复簇 161，cosine >0.85）
  → 终版 rl_v2_golden.jsonl (4912)
```

中间产物归档在 `archive/`。

---

## 二、标注机制与票型（n=4912）

- **合议庭**：三席盲测（A deepseek-v4-pro-0813 / B glm-5.3 / C gpt-6.1-sol・早期批为 gpt-6-sol/gpt-5.6-sol）
  → 四字段全一致直采 / 任一维分歧进 kimi-k3 仲裁
- **票型**：consensus = 1275 (26.0%) ｜ arbitrated = 3637 (74.0%)
- **查④复核**：MiniMax-M3 只校验「首要 risk Key + 极性」，3 轮换证据抢救后才拦截

---

## 三、标签分布（n=4912）

**单维边际**：

| 维度 | 分布 |
|---|---|
| sentiment | negative 2303 (46.9%) / neutral 1310 (26.7%) / positive 1299 (26.4%) |
| relevance | core 3158 (64.3%) / passing 912 (18.6%) / foil 842 (17.1%) |
| hard_case_type | multi_subject 1432 / irony 1121 / comparative 1044 / metaphor 843 / counterfactual 249 / explicit 119 / fact_implicit 104 |
| risk Key 计次 | service 1057 / product 1047 / marketing 932 / compliance 492 / finance 243 / general_negative 241 / governance 198 ｜ 空表 2072 条 |

**极性 × 难点类型**：

| 极性 \ 难点 | explicit | fact_implicit | irony | metaphor | comparative | counterfactual | multi_subject | 合计 |
|---|---|---|---|---|---|---|---|---|
| negative | 103 | 14 | 739 | 511 | 379 | 154 | 403 | 2303 |
| neutral | 2 | 90 | 150 | 138 | 187 | 63 | 680 | 1310 |
| positive | 14 | 0 | 232 | 194 | 478 | 32 | 349 | 1299 |
| **合计** | " + " | ".join(map(str, tot_hc)) + f" | **4912** |

**极性 × 语篇地位**：

| 极性 \ 语篇 | core | passing | foil | 合计 |
|---|---|---|---|---|
| negative | 2060 | 147 | 96 | 2303 |
| neutral | 658 | 544 | 108 | 1310 |
| positive | 440 | 221 | 638 | 1299 |
| **合计** | 3158 | 912 | 842 | **4912** |

---

## 四、模型间 Cohen's κ（n=4912，与金标500同口径 risk=集合全等）

| 席位对 | sentiment κ (一致率) | hard_case_type κ (一致率) |
|---|---|---|
| deepseek-v4-pro-0813 vs glm-5.3 | 0.8668 (0.9153) | 0.5845 (0.6724) |
| deepseek-v4-pro-0813 vs gpt-6.1-sol/gpt-6-sol | 0.8362 (0.8945) | 0.7508 (0.8035) |
| glm-5.3 vs gpt-6.1-sol/gpt-6-sol | 0.8076 (0.8758) | 0.5495 (0.6397) |
| deepseek-v4-pro-0813 vs kimi-k3(仲裁) | 0.9121 (0.9442) | 0.8222 (0.8603) |
| glm-5.3 vs kimi-k3(仲裁) | 0.8603 (0.9109) | 0.4953 (0.5961) |
| gpt-6.1-sol/gpt-6-sol vs kimi-k3(仲裁) | 0.8418 (0.898) | 0.7604 (0.8106) |

> 极性全体 κ ≥ 0.81；hard_case 是主要分歧源（B 席 vs 仲裁 κ=0.50 最低）——复合难点 + 扰动语料的
> 预期分歧，仲裁机制兜底，也是 RL 高价值难例来源。与金标 500 的对照结论不变：
> risk 三对统计持平、排序结构（A 最强/B 最弱）复现、sentiment/relevance 显著更低系语料难度上移所致
> （详见上一版报告的 Bootstrap CI 对照，结论不随扩量改变）。

---

## 五、终稿(gold) vs 各席原始判定 Cohen's κ（n=4912）

| 维度 | deepseek-v4-pro (A) | glm-5.3 (B) | gpt-6.1/6-sol (C) | kimi-k3 (仲裁) |
|---|---|---|---|---|
| sentiment | **0.9351** | 0.8967 | 0.8829 | 1.000* |
| relevance | **0.8725** | 0.746 | 0.8156 | 1.000* |
| hard_case_type | **0.868** | 0.6222 | 0.8222 | 1.000* |
| risk_categories | **0.8732** | 0.7638 | 0.7754 | 1.000* |

\* 仲裁席 κ=1.0 为结构性现象（74.0% 样本 gold 即仲裁判定）。A 席四维全面最高
（首席法官效应复现），B 席 hard_case(0.6222)/relevance(0.746) 最弱——与历史观察同构。

---

## 六、写手自报极性 vs gold（扰动效果验证）

自报与 gold 一致率 **49.7%**（2439/4912）——「双候选标签扰动」持续生效：写手端不掌握真实标签，
spec-gold 冲突从机制上不存在，全部样本以合议庭判定为准。迁移矩阵明细见 `stats_report.json` 的
`写手自报_vs_gold` 字段。

---

## 七、训练/测试切分（详见 SPLIT_README.md）

| 文件 | 条数 | 用途 |
|---|---|---|
| `eval300_EVAL_FROZEN.jsonl` | 300 | **测试集（冻结）**——两层设计：150 配比框架 + 150 分歧难例富集 |
| `train_rl_v2.jsonl` | 4612 | **训练集**——接 GRPO |

零泄漏（eval vs train Jaccard≥0.65 = 0 对）；评估协议 = eval300（分布内+边界判别）+ seed500_final（常规语料泛化）双基准。

**盲区补题效果（batch7）**：train 中 A 盲区 negative×passing 106→134 条（+26%，batch7 贡献 17 条
neg×pass + 9 条 neg×pass 相关难点变体），双盲区交汇 negative×passing×counterfactual 1→4 条；
B 盲区 counterfactual×negative 37 条新增（hit 率 24%，含语篇变体后净增 ~147 条计数区域）。
命中率偏低的主因：victim_passing 的双候选扰动让写手常把情感落在正面（52/237 落 positive）；
batch8 若继续补题，建议该槽位情感候选收敛为 [negative, neutral]。

---

## 八、复现与审计

```bash
# 合并 + 三重去重
python3 -I scripts/merge_rl_v2.py \
  --src output/corpus/rl_v3_batch1/golden.jsonl \
        output/corpus/rl_v3_batch2/golden.jsonl \
        output/corpus/rl_v3_batch3/golden.jsonl \
        output/corpus/rl_v3_batch4/golden.jsonl \
        output/corpus/rl_v3_batch5/golden.jsonl \
        output/corpus/rl_v3_batch6/golden.jsonl \
        output/corpus/rl_v3_batch7/golden.jsonl \
  --out_dir output/rl_v2

# 统计 + κ（纯离线）
python3 -I scripts/stats_rl_v2.py \
  --src output/rl_v2/rl_v2_golden.jsonl \
  --out output/rl_v2/stats_report.json

# 训练/测试切分（seed=42 可复现）
python3 -I scripts/sample_eval300.py \
  --src output/rl_v2/rl_v2_golden.jsonl --out_dir output/rl_v2 --seed 42
```

**数据血缘**：每条 meta 含 `pred_a/b/c`（三席原始判定）、`arbiter`（仲裁卷宗）、`anno_mode`、
`chosen_seat/chosen_model`、`source_batch/source_file`、`target_subject/anchor_source`、
`sentiment_candidates/hard_case_main/hard_case_aux`（扰动槽位）、batch7 另含 `target_slot`（盲区题型标记，
从 event_id 第三段可恢复：victim/cf_nar/cross_）。去重中间产物在 `archive/`。
