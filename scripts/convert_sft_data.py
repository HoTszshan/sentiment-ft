#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema v1 批次转换器 → 训练/评测格式 (零 LLM 调用)

适用任何 Schema v1 标注批次，两个已知用例：
  - 金标测试集: ../output/gold_eval_final/seed500_final.jsonl → data/gold500_test.jsonl
    （evaluate_sft / release_gate / eval_four_tier 直接消费）
  - 训练批次:   ../output/sft_round1/round1_train.jsonl → data/round1_train_aligned.jsonl
    （mix_sft_dataset 混合后喂 train_sft_lora）

只动三处，其余逐字继承：
  1. CoT 标签: <|begin_of_thought|>→<think>、<|end_of_thought|>→</think>
     （契约解析器 schema_contract.parse_output 只认 <think>；实测 500/500 兼容）
  2. 补 instruction / input: task_template.build_instruction + build_user_input
     （与训练数据同模板；评测模式 A 需要 instruction 字段）
  3. meta 白名单: 只留评测消费面字段（industry 分桶 / hard_case_type·sentiment·
     relevance 兜底 / provenance / event_id），另存 input_raw 裸原文；
     丢弃 judges/arbiter/diff_fields 等打标过程字段

质检（不过即剔除并计数）: 标签替换后 parse_output 可解析 + validate_schema +
evidence 是裸原文连续子串 4~80 字 + resolve_gold 零冲突

用法:
    python3 scripts/convert_sft_data.py \
        --src ../output/gold_eval_final/seed500_final.jsonl \
        --out data/gold500_test.jsonl \
        --report data/gold500_convert_report.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema_contract as sc          # noqa: E402
import task_template as tt            # noqa: E402
from corpus_schema import INDUSTRY_SCHEMAS  # noqa: E402

# Schema v1 的 CoT 标签 → 契约标签（唯一允许的字符串替换）
TAG_MAP = {
    "<|begin_of_thought|>": sc.THOUGHT_BEGIN,
    "<|end_of_thought|>": sc.THOUGHT_END,
}

# meta 白名单：评测/训练链路实际消费的字段（其余打标过程字段一律丢弃）
META_KEEP = ("provenance", "sft_bucket")


def convert_output(raw: str) -> str:
    """只替换 CoT 标签，其余字符逐字保留（含 </think> 后换行等用户格式）。"""
    out = raw
    for old, new in TAG_MAP.items():
        out = out.replace(old, new)
    return out


def convert_row(row: dict):
    """源行 → 评测行；返回 (评测行 dict, 剔除原因 or None)。"""
    industry = row.get("industry") or ""
    text = row.get("input") or ""
    subject = (row.get("target_subject")
               or (row.get("gold") or {}).get("subject") or "")

    # ---- 质检（转换前判定，不合格直接剔除）----
    if not text.strip():
        return None, "input 为空"
    if industry not in INDUSTRY_SCHEMAS:
        return None, f"industry 越界 {industry!r}"
    gold_src = row.get("gold") or {}
    err = sc.validate_schema(gold_src)
    if err:
        return None, f"gold schema: {err}"
    ev = gold_src.get("evidence") or ""
    if ev and ev not in text:
        return None, "evidence 非原文连续子串"

    # ---- 变换 ----
    output = convert_output(row.get("output") or "")
    cot, pred, flags = sc.parse_output(output)
    if pred is None:
        return None, "标签替换后 output 仍不可解析"
    if flags["schema_error"]:
        return None, f"output schema: {flags['schema_error']}"
    # 铁律「output JSON 为准」：与行内 gold 冲突即剔除（实测 0 条）。
    # subject 例外：spec 与 gold 存在空格级差异（如「极氪 001」vs「极氪001」），
    # 归一化空白后一致即放行（subject 以 output JSON 为准）。
    import re as _re
    for k in ("sentiment", "relevance", "hard_case_type", "risk_categories",
              "evidence", "subject"):
        pv, gv = pred.get(k), gold_src.get(k)
        if k == "subject" and pv is not None and gv is not None:
            if _re.sub(r"\s+", "", str(pv)) == _re.sub(r"\s+", "", str(gv)):
                continue
        if pv != gv:
            return None, f"output 与 gold 冲突: {k}"

    src_meta = row.get("meta") or {}
    meta = {}
    for k in META_KEEP:
        v = row.get(k) if row.get(k) is not None else src_meta.get(k)
        if v is not None:
            meta[k] = v
    # provenance 归一：Schema v1 的 provenance 可能是 dict（含 writer_model 等审计信息），
    # 下游 mix_sft_dataset 把它当字符串键统计 —— dict 展开取 source，完整信息挪 provenance_detail
    prov = row.get("provenance")
    if isinstance(prov, dict):
        meta["provenance_detail"] = prov
        meta["provenance"] = str(prov.get("source") or "synthetic")
    else:
        meta["provenance"] = str(prov) if prov is not None else "synthetic"
    meta["industry"] = industry
    meta["hard_case_type"] = gold_src["hard_case_type"]
    meta["sentiment"] = gold_src["sentiment"]
    meta["relevance"] = gold_src["relevance"]
    meta["event_id"] = row.get("event_id") or ""
    meta["input_raw"] = text          # 裸原文（input 是题面壳，grounding 用裸原文）

    return {
        "event_id": row.get("event_id") or "",
        "instruction": tt.build_instruction(industry),
        "input": tt.build_user_input(subject, text),
        "output": output,
        "meta": meta,
    }, None


def main():
    ap = argparse.ArgumentParser(description="500 金标 → 评测链路格式 (纯规则, 零 LLM)")
    ap.add_argument("--src", required=True, help="seed500_final.jsonl 路径")
    ap.add_argument("--out", default="data/gold500_test.jsonl")
    ap.add_argument("--report", default="", help="质检报告 json 路径 (留空则打印)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[FATAL] 找不到源文件: {src}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total, kept, drop_reasons = 0, 0, Counter()
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row, reason = convert_row(json.loads(line))
            if reason:
                drop_reasons[reason] += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    report = {"src": str(src), "total": total, "kept": kept,
              "dropped": total - kept, "drop_reasons": dict(drop_reasons)}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if kept == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
