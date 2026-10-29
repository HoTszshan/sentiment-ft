# 黄金评测集 · 终库说明（seed500_final · Schema v1）

> **交付日期**：2026-10-26 ｜ **规范版本**：`Schema v1`（数据格式规范见 `materials/语料数据格式规范_v1.md`）
> **总量**：500 条 ｜ **跨库去重**：零冲突、零重复（对旧训练库最大 Jaccard 5.39%，远低于 0.8 红线）
> **全流程总控**：`bash scripts/rebuild_gold500.sh` ｜ **方法论底稿**：`plan/500条种子语料_两阶段动态补齐执行计划.md`

---

## 一、文件清单与路径

| 文件 | 内容 | 条数 |
|---|---|---|
| **`seed500_final.jsonl`** | **★ 最终交付物**（Schema v1 标准格式，500 条硬断言通过） | 500 |
| `dist_report.json` | 终库四维/打标/写手分布统计报告（由 `scripts/gold_eval_stats.py` 产出） | — |
| `archive/seed_all_merged_raw.jsonl`| 五阶段合并去重全量底库（A+B+C+D+E，未下采样前） | 669 |
| `archive/seed500_gold.jsonl` | 配比下采样候选池（500 条） | 500 |
| `archive/seed500_retired.jsonl` | 配比平衡退役样本（软删除留档，可回滚） | 159 |
| `archive/convert_report.json` | Schema v1 格式转换审计报告（0 拒绝，100% 通过） | — |
| `archive/cross_conflict_report.json`| 跨库去重与冲突检测报告（0 冲突，0 纯重复） | — |
| `archive/meta_scan_report.json` | 元评论与拒稿腔扫描报告（10 条 soft 命中已全量剔除） | — |
| `archive/kappa_report.json` | Cohen's κ 矩阵与审官模型贡献报告（由 `scripts/gold_eval_kappa.py` 产出） | — |

**统计与校验指令**：
```bash
python3 scripts/gold_eval_stats.py --src output/gold_eval_final/seed500_final.jsonl
python3 scripts/gold_eval_kappa.py --src output/gold_eval_final/seed500_final.jsonl
```

---

## 二、标签分布（金标真值 gold，四维全达标）

**sentiment**（目标 负30% / 中50% / 正20%）：

| 极性 | negative | neutral | positive |
|---|---|---|---|
| 条数 | 166 | 226 | 108 |
| 占比 | 33% | 45% | 22% |

**relevance**（目标 core 60% / passing 30% / foil 10%）：

| 语篇地位 | core | passing | foil |
|---|---|---|---|
| 条数 | 328 | 136 | 36 |
| 占比 | 66% | 27% | 7% |

**hard_case_type**（七大难点全序覆盖 · 多主体配比优化版）：

| 类型 | 条数 | 占比 | 类型 | 条数 | 占比 |
|---|---|---|---|---|---|
| explicit 词面直白 | 129 | 26% | comparative 显式对比 | 45 | 9% |
| fact_implicit 客观隐式 | 111 | 22% | irony 反讽修辞 | 45 | 9% |
| **multi_subject 多主体** | **80** | **16%** | metaphor 隐喻黑话 | 40 | 8% |
| counterfactual 反事实 | 50 | 10% | | | |

> 💡 **配比结构优化说明**：工业级舆情公网长文中，多主体事件交织（multi_subject）远比显式单挑对比（comparative）更普遍、更容易诱发小模型误报。阶段 E 专门注入 mimo-v2.6-flash 撰写的 30 条高质量多主体样本，将 `multi_subject` 占比拉升至 16%（80条），与 `comparative`（45条）形成近 2:1 的梯队，真实考核模型在多品牌共存场景下的主体定位与抗假阳能力。

**industry**（六大支柱行业均衡平摊）：
`electronics_3c` 92 (18%) ｜ `automotive` 90 (18%) ｜ `beauty_cosmetics` 85 (17%) ｜ `finance_insurance` 79 (16%) ｜ `fmcg` 79 (16%) ｜ `real_estate` 73 (15%)

**risk_categories**：
空表（中性/正面无风险）264 条（53%）｜ marketing 42 ｜ product 41 ｜ service 34 ｜ product+service 30 ｜ compliance 14（其余为多元多 Key 组合）

**文本长度**：
54 ~ 359 字，均值 175 字。

---

## 三、写手分布（17 款写手模型跨族群分散）

| 写手模型（通道） | 条数 | 占比 | 写手模型（通道） | 条数 | 占比 |
|---|---|---|---|---|---|
| gpt-6-sol (vectide) | 90 | 18% | deepseek-v4.1-flash (dashscope) | 18 | 4% |
| gpt-6-luna (vectide) | 88 | 18% | doubao-seed-2-0-lite (aihubmix) | 15 | 3% |
| MiniMax-M3 (vectide) | 48 | 10% | doubao-seed-2-1-turbo (aihubmix) | 15 | 3% |
| claude-haiku-5-5 (aihubmix) | 47 | 9% | doubao-seed-2-1-pro (aihubmix) | 14 | 3% |
| glm-5.3 (vectide) | 44 | 9% | qwen3.6-plus 等 4 款 | 18 | 4% |
| **mimo-v2.6-flash (vectide)** | **30** | **6%** | | | |
| deepseek-v4-flash (dashscope) | 27 | 5% | | | |
| kimi-k3 (vectide) | 27 | 5% | | | |
| qwen3.7-max (autodl) | 19 | 4% | | | |

---

## 四、打标一致性 · Cohen's κ 与最终采纳模型

**打标合议庭**：主审 A=`deepseek-v4-pro-0813`，主审 B=`glm-5.3`，主审 C=`gpt-6-sol`；终审仲裁官=`kimi-k3`；查④证据复核=`MiniMax-M3`。系统法典版本 `anno_prompt_v2.2`。

### 1. 终稿与产出大模型分布（`chosen_model`）

终库 500 条样本的最终 CoT 与答案来源大模型（全票真随机等概率抽选，0 个 unknown）：

| 胜出模型 (`chosen_model`) | 席位 (`chosen_seat`) | 模式说明 | 采纳条数 | 终库占比 |
|---|---|---|---|---|
| **kimi-k3** | arbiter 席 | 争议难例仲裁定稿 | **222 条** | **44.4%** |
| **deepseek-v4-pro-0813** | A 席 | 主审全票随机 / 多数胜出 | **109 条** | **21.8%** |
| **gpt-6-sol** | C 席 | 主审全票随机 / 多数胜出 | **89 条** | **17.8%** |
| **glm-5.3** | B 席 | 主审全票随机 / 多数胜出 | **80 条** | **16.0%** |

> 💡 **席位均衡说明**：除仲裁官定稿（44.4%）外，主审全票样本（185条）通过等概率真随机算法在 A/B/C 三席中均衡抽选；多数表决（93条）在多位候选席位间等权抽选。主审三席分布均衡（A: 109, C: 89, B: 80），彻底杜绝了席位固化偏置。

### 2. 打标模式构成

| 打标模式 (`meta.anno_mode`) | 条数 | 占比 | 仲裁规则解释 |
|---|---|---|---|
| **仲裁官定稿** (`arbitrated`) | 222 | 44.4% | sentiment 或 risk 存在分歧，强制触发 kimi-k3 深度卷宗仲裁 |
| **三审全票** (`consensus_abc`) | 185 | 37.0% | 三席盲审 100% 全票一致，**等概率随机抽选** 一席原装 CoT |
| **多数表决直采** (`majority_vote`)| 93 | 18.6% | 硬维度全票，软维度 2:1 多数直采，直接取多数派原装 CoT |

### 3. 仲裁官（kimi-k3）vs 各主审法官 Cohen's κ（仅仲裁样本，n=222）

在全库 222 条出现分歧并触发深度仲裁的难例上，仲裁官终审定案结论与各主审原始盲审的一致性矩阵：

| 评测维度 | 法官 A (`deepseek-v4-pro-0813`) | 法官 B (`glm-5.3`) | 法官 C (`gpt-6-sol`) | 关键技术结论 |
|---|---|---|---|---|
| **sentiment** | **0.914** (一致率 95.0%) | 0.883 (一致率 93.2%) | 0.846 (一致率 91.0%) | 争议难例中，A 席极性把控最贴近仲裁定案 |
| **relevance** | **0.914** (一致率 96.0%) | 0.817 (一致率 91.4%) | 0.793 (一致率 89.6%) | A 席在语篇地位判别上领跑（一致率高达 96%） |
| **hard_case_type** | **0.872** (一致率 89.2%) | 0.676 (一致率 73.0%) | 0.716 (一致率 76.1%) | 隐喻/反讽等高阶句式中，B/C 席多与终审分歧 |
| **risk_categories** | **0.787** (一致率 82.0%) | **0.531** (一致率 59.9%) | 0.598 (一致率 67.6%) | **硬门禁分流的核心证据**：B 席在风险上 κ 仅 0.531 |

> 💡 **业务与算法洞察**：
> 1. **首席法官效应**：在三方各执一词的争议难例中，仲裁官 kimi-k3 在全部四维上均最倾向于采纳法官 A（deepseek-v4-pro）的判定，四维一致率达 82%~96%；
> 2. **仲裁规则 v2 的客观依据**：法官 B 在 `risk_categories` 上 κ 仅 0.531（多发生漏标或错标治理/营销风险），这为“*只要 risk 有 1 席分歧必强制送仲裁*”的设计提供了无可辩驳的数据支撑。

### 4. 主审三方盲测两两 Cohen's κ（n=500，审官间一致性）

| 评测维度 | A vs B (deepseek/glm) | A vs C (deepseek/gpt) | B vs C (glm/gpt) | 维度共识说明 |
|---|---|---|---|---|
| **sentiment** | **0.925** (95.2%) | **0.909** (94.2%) | **0.897** (93.4%) | 极性共识度极高，均 ≥0.90 |
| **relevance** | **0.859** (93.2%) | **0.813** (90.6%) | **0.784** (89.2%) | 语篇地位判别共识稳健（均 ≥0.78） |
| **hard_case_type** | **0.702** (75.4%) | **0.737** (78.0%) | **0.662** (72.2%) | 7类全序裁定难度最高，分歧多交仲裁 |
| **risk_categories** | **0.692** (77.8%) | **0.724** (81.0%) | **0.629** (74.2%) | 风险维度集合严格全等比（B席多漏标） |

### 5. 终稿 gold vs 各主审原始判定 Cohen's κ（全量口径，n=500）

| 评测维度 | 法官 A (deepseek-v4-pro) | 法官 B (glm-5.3) | 法官 C (gpt-6-sol) |
|---|---|---|---|
| **sentiment** | **0.966** (一致率 97.8%) | 0.953 (一致率 97.0%) | 0.938 (一致率 96.0%) |
| **relevance** | **0.951** (一致率 97.6%) | 0.876 (一致率 94.0%) | 0.861 (一致率 93.0%) |
| **hard_case_type** | **0.899** (一致率 91.6%) | 0.775 (一致率 81.6%) | 0.817 (一致率 84.8%) |
| **risk_categories** | **0.887** (一致率 92.0%) | 0.750 (一致率 82.2%) | 0.788 (一致率 85.6%) |

> 读法：法官 A（deepseek-v4-pro）在所有四个维度上与最终金标的 Cohen's κ 全面领跑（κ 达 0.89~0.97），说明其对法典规则的遵循度最扎实；在触发仲裁的难例中，仲裁官采纳法官 A 观点的频次也显著高于 B/C 席。

---

## 五、数据格式规范（Schema v1）

终库 `seed500_final.jsonl` 全量对齐 **Schema v1**。顶层仅保留 6 个高频操作 key，其余元数据按语义分归三块。

### 1. 结构概览

```json
{
  "event_id": "s3C-elec-explicit-passing-n-029",
  "target_subject": "Find X系列",
  "industry": "electronics_3c",
  "input": "今年高端机大盘是真卷，供应链那边镜头、存储全在涨价...顺带一提，Find X系列也在这波新品名单里。",
  "output": "<|begin_of_thought|>\nStep 1: ...\nStep 2: ...\nStep 3: ...\n<|end_of_thought|>{\"subject\": \"Find X系列\", \"relevance\": \"passing\", \"sentiment\": \"neutral\", \"hard_case_type\": \"fact_implicit\", \"risk_categories\": [], \"evidence\": \"Find X系列也在这波新品名单里\"}",
  "gold": {
    "subject": "Find X系列",
    "sentiment": "neutral",
    "relevance": "passing",
    "hard_case_type": "fact_implicit",
    "risk_categories": [],
    "evidence": "Find X系列也在这波新品名单里"
  },
  "spec": { ... },
  "provenance": { ... },
  "meta": { ... }
}
```

### 2. 字段分块说明

| 字段块 | 包含字段 | 语义定位与注意事项 |
|---|---|---|
| **顶层 (Top-level)** | `event_id`, `target_subject`, `industry`, `input`, `output`, `gold` | 极简操作面。`target_subject` 与 `industry` 直读；`gold` 是**下游评测计算 F1/准确率的唯一真值源**。 |
| **`gold`** | `subject`, `sentiment`, `relevance`, `hard_case_type`, `risk_categories`, `evidence` | 黄金标注真值字典。`evidence` 必须无差错作为 `input` 的严格连续子串。 |
| **`spec`** | `target_subject`, `industry`, `sentiment`, `relevance`, `hard_case_type`, `risk_keys`, `theme`, `platform`, `persona`, `length_hint` 等 | **写手原始输入规格**。⚠️ 警示：`spec` 内的极性/难点/地位是写手扰动意图，**不等同于 gold**！若写手写偏，打标裁定修正后记录在 `gold`，`spec` 仍保留写手初衷以供溯源审计。 |
| **`provenance`** | `source="synthetic"`, `stage` (A~E), `writer_model`, `gen_model`, `gen_seed`, `length`, `subject_hit` | 生产批次血缘、写手模型与门禁抽检指标。 |
| **`meta`** | `anno_mode`, `judges`, `chosen_seat`, `chosen_model`, `pred_a/b/c`, `arbiter`, `judge4`, `diff_fields` | 打标全过程留痕。`chosen_model` 记录最终生成答案的大模型真实全称；`arbiter` 记录仲裁卷宗；`judge4` 记录证据复核状态（`passed` 或 `repaired`）。 |

### 3. 模型 `output` 契约
- 格式必须以 `<|begin_of_thought|>` 开头，以 `<|end_of_thought|>` 分隔思维链与尾部扁平 JSON；
- 尾部 JSON 必须能被 `json.loads()` 解析，且解析字段集合与取值与 `gold` 字典 100% 严格一致（逐条门禁断言）。

---

## 六、复现流水线

```bash
# 执行完整重建流水线（自动完成合并 -> 转换 -> 内部去重 -> 跨库去重 -> 扫描剔除 -> 下采样 -> 断言门禁）
bash scripts/rebuild_gold500.sh
```
