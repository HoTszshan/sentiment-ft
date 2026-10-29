# 数据格式转换指南（Schema v1 → 训练/评测格式）

> 适用任何 Schema v1 标注批次（金标测试集、SFT 训练批次）。
> 转换器：`scripts/convert_sft_data.py`，纯规则、零 LLM 调用、可幂等重跑。

---

## 一、转换指令

所有命令都在 `tests/` 目录下执行：

```bash
cd /Users/tan/Desktop/sallyhe/resume/resume_v3/projects/sentiment/MVP/tests
```

### 用例 A · 500 条金标测试集（评测成绩单）

```bash
python3 -I scripts/convert_sft_data.py \
    --src ../../output/gold_eval_final/seed500_final.jsonl \
    --out data/gold500_test.jsonl \
    --report data/gold500_convert_report.json
```

预期输出：`"total": 500, "kept": 500, "dropped": 0`。

### 用例 B · round1 训练批次（1524 条，10 行业）

```bash
python3 -I scripts/convert_sft_data.py \
    --src ../../output/sft_round1/round1_train.jsonl \
    --out data/round1_train_aligned.jsonl \
    --report data/round1_convert_report.json
```

预期输出：`"total": 1524, "kept": 1524, "dropped": 0`。


### 参数说明

| 参数 | 说明 |
|---|---|
| `--src` | Schema v1 源文件路径（含 `gold` / `output` / `input` / `industry` / `target_subject` 字段） |
| `--out` | 转换产物路径（训练/评测链路直接消费） |
| `--report` | 质检报告落盘路径（`total / kept / dropped / drop_reasons`），留空则只打印到终端 |

---

## 二、转换做了什么（三件事）

1. **CoT 标签替换**：`<|begin_of_thought|>` → `<think>`、`<|end_of_thought|>` → `</think>`。
   契约解析器（`lib/schema_contract.parse_output`）只认 `<think>` 系标签；除这两个标签外其余字符逐字保留。

2. **补 instruction / 题面**：
   - system 侧 = `lib/task_template.build_instruction(industry)`（角色 + 任务目标 + 行业风险体系 + 输出协议，与训练数据同模板，同行业逐字一致）
   - user 侧 = `【目标主体】{subject}` + `【输入文本】{原文}`

3. **meta 白名单瘦身**：只留 `industry` / `hard_case_type` / `sentiment` / `relevance` / `provenance` / `sft_bucket`（有则透传）/ `event_id` / `input_raw`，丢弃 judges / arbiter / diff_fields 等打标过程字段。
   `provenance` 若是 dict（含 writer_model 等审计信息）会归一化为 `source` 字符串，完整信息挪 `provenance_detail`。

**质检**（不过即剔除并计数进 report）：gold 六字段过 `validate_schema` / evidence 是原文连续子串 4~80 字 / 标签替换后 output 可解析 / output 内嵌 JSON 与 gold 逐字段一致（subject 允许空白归一化后一致）/ 行业在词表内。

---

## 三、转换后的用途

### 3.1 金标测试集（`gold500_test.jsonl`）—— 只评不训

```bash
# 数据自检（模式 A）：四查全过才算数据可信
python3 -I scripts/evaluate_sft.py \
    --test_file data/gold500_test.jsonl \
    --out output/report_gold500_check.json

# 模型对照评测（模式 B）：训练完的模型 vs 金标
python3 -I scripts/evaluate_sft.py \
    --gold_file data/gold500_test.jsonl \
    --pred_file output/pred_gold500.jsonl \
    --out output/report_gold500.json

# 发布门（四道门独立判定；缺输入的门返回 N/A，不假装 PASS）
python3 -I scripts/release_gate.py \
    --gold data/gold500_test.jsonl \
    --pred output/pred_gold500.jsonl \
    --old_error_list output/old_errors.txt \
    --baseline_json output/report_prev.json \
    --out output/gate.json

# 四档控制变量对照（T1 基座 / T2 提示词加固 / T3 SFT / T4 GRPO）
python3 -I scripts/eval_four_tier.py \
    --gold data/gold500_test.jsonl \
    --pred_base output/pred_t1.jsonl \
    --pred_prompt output/pred_t2.jsonl \
    --pred_sft output/pred_t3.jsonl \
    --pred_grpo output/pred_t4.jsonl \
    --out output/tier_report.json
```

> ⚠️ **纪律：金标只评不训**。这 500 条是简历里「500 条人标金标」合法数字的唯一分母来源，绝不能混进 SFT 训练集。

### 3.2 训练批次（`round1_train_aligned.jsonl` / `sft750_train_aligned.jsonl`）—— 只训不评

```bash
# 9:1 混合 Anchor + 分层切分 train/val
# mix_sft_dataset 默认 domain_file 已指向 round1_train_aligned.jsonl
python3 -I scripts/mix_sft_dataset.py --out_dir data/mixed

# 换用 sft750 时显式传参
python3 -I scripts/mix_sft_dataset.py \
    --domain_file data/sft750_train_aligned.jsonl \
    --anchor_file data/coig_cqia_anchor.jsonl \
    --out_dir data/mixed \
    --anchor_ratio 0.1 --val_ratio 0.1 --seed 42

# 训练前干跑（无 GPU 验证掩码/加权逻辑，POC 三档）
SFT_POC_STAGE=POC1 python3 -I scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
SFT_POC_STAGE=POC2 python3 -I scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl
SFT_POC_STAGE=POC3 python3 -I scripts/train_sft_lora.py --dry_run --train_file data/mixed/sft_train.jsonl

# 真实训练（AutoDL 单卡）
SFT_POC_STAGE=POC1 ./scripts/run_probe_e2e.sh train
```

---

## 四、与另一条转换路（步骤 1.5 对齐器）的区别

| | `convert_sft_data.py`（本文件） | `align_sft_data.py`（步骤 1.5） |
|---|---|---|
| 吃什么 | **Schema v1 标注批次**（已有 `gold`/`spec`/`output`，CoT 标签是 `<\|begin_of_thought\|>`） | 「原文+meta」原始题面（没有 output，或 output 是 `<think>` 标签的成品行） |
| 典型源 | `seed500_final.jsonl` / `round1_train.jsonl` / `sft750_sample.jsonl` | 标注流水线早期的原文+meta 文件 |
| gold 来源 | 源行自带 `gold` 字段（逐字继承，与 output 内嵌 JSON 交叉校验） | 从 `output` JSON 直读，回退扁平 meta 字段 |

新数据是「标注流水线产出的 Schema v1 批次」就走本文件的转换；是「原始原文+meta」才走步骤 1.5 对齐器。

---

## 五、完整流水线中的位置

```
步骤 1   下载 Anchor + 基座            download_assets.py
步骤 1.5 对齐器（原文+meta 路）        align_sft_data.py
步骤 1.6 Schema v1 批次转换（本文件）  convert_sft_data.py   ← 你在这里
步骤 2   9:1 混合 + 分层切分           mix_sft_dataset.py
步骤 3   LoRA SFT 训练                 train_sft_lora.py
步骤 4   推理                          probe_infer.py
步骤 5   评测                          evaluate_sft.py
```

详细背景见 `tests/README.md` §2（步骤 1.6）与 §5（评测/发布门）。
