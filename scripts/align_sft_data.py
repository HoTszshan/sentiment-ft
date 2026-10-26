#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式对齐器 · 纯规则 (零 LLM 调用)

把标注流水线产出的「原文+meta」jsonl (如 ../output/corpus/synth_300_stageA.dedup.jsonl)
对齐成 SFT 训练格式 {instruction, input, output, meta}：

  - instruction 由 tests/lib/task_template.build_instruction 生成（用户定稿模板：
    角色+任务目标+行业风险体系+输出协议；目标主体与原文在 user 侧）
  - gold JSON 从 meta 纯规则映射：subject=target_subject, risk_categories=risk_keys,
    evidence=evidence_span
  - CoT **只继承、不生成**：源行自带 output（含 <think>）则逐字沿用；否则只输出裸 JSON。
    对齐器永不编造 CoT —— 模板化 CoT 会被模型死记，并刷穿 GRPO 的 rubric 奖励
  - 逐条跑 schema_contract.validate_schema + evidence 子串 + subject_hit 质检，
    不合格剔除并按原因计数进 report

provenance 等全部 meta 字段透传 (mix_sft_dataset 靠 provenance 判 anchor、
靠 meta.hard_case_type × meta.sentiment 分层，丢失会退化)。

用法:
    python3 scripts/align_sft_data.py \
        --src ../output/corpus/synth_300_stageA.dedup.jsonl \
        --out data/train_synth_81_aligned.jsonl \
        --report data/align_report.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema_contract as sc          # noqa: E402
import task_template                   # noqa: E402
from corpus_schema import INDUSTRY_SCHEMAS, schema_to_inline  # noqa: E402

# 输入长度上限 (防 OOM 长尾，与 mix_sft_dataset --anchor_max_chars 逻辑呼应)
INPUT_MAX_CHARS = 2000

def flatten(row: dict) -> dict:
    """把 meta 子对象摊平到顶层 (兼容两种源：扁平「原文+meta」/ 带 meta 嵌套的 SFT 行)。"""
    flat = dict(row)
    meta = flat.pop("meta", None)
    if isinstance(meta, dict):
        for k, v in meta.items():
            flat.setdefault(k, v)
    return flat


def existing_gold(row: dict) -> dict:
    """源行自带 output 时，取 <think> 之后的 JSON 作为 gold (缺标签则整体解析)。"""
    raw = (row.get("output") or "").strip()
    if not raw:
        return {}
    tail = raw.split(sc.THOUGHT_END, 1)[1] if sc.THOUGHT_END in raw else raw
    try:
        obj = json.loads(tail.strip())
    except (json.JSONDecodeError, IndexError):
        return {}
    return obj if isinstance(obj, dict) else {}


def gold_of(row: dict) -> dict:
    """meta → gold JSON (键序 = schema_contract.REQUIRED_FIELDS)。

    字段优先级：源行已有 output 的 JSON > 扁平 meta 字段（题面元数据）> 缺失即空。
    """
    got = existing_gold(row)
    return {
        "subject": got.get("subject") or row.get("target_subject"),
        "relevance": got.get("relevance") or row.get("relevance"),
        "sentiment": got.get("sentiment") or row.get("sentiment"),
        "hard_case_type": got.get("hard_case_type") or row.get("hard_case_type"),
        "risk_categories": got.get("risk_categories") or list(row.get("risk_keys") or []),
        "evidence": got.get("evidence") or row.get("evidence_span"),
    }


def render_output(row: dict, gold: dict) -> Tuple[str, bool]:
    """assistant 侧输出：源行自带 output 则逐字沿用，否则用 gold 拼裸 JSON。

    返回 (output, has_cot)。**永不生成 CoT** —— 模板化 CoT 会被死记并刷穿 GRPO rubric。
    """
    src_output = (row.get("output") or "").strip()
    has_cot = sc.THOUGHT_BEGIN in src_output
    if src_output:
        return src_output, has_cot
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":")), False


def check_row(row: dict):
    """质检主判定式，返回剔除原因；通过返回 None。"""
    text = row.get("input") or ""
    if not text.strip():
        return "input 为空"
    # 注意: text 是【裸原文】(未包 user 侧壳)；evidence 子串判定以裸原文为准
    if len(text) > INPUT_MAX_CHARS:
        return f"input 超长 ({len(text)}>{INPUT_MAX_CHARS})"
    if row.get("industry") not in INDUSTRY_SCHEMAS:
        return f"industry 越界 {row.get('industry')!r}"
    if row.get("hard_case_type") not in sc.VALID_HARD_CASE:
        return f"hard_case_type 越界 {row.get('hard_case_type')!r}"
    bad_risk = [k for k in (row.get("risk_keys") or []) if k not in sc.VALID_RISK]
    if bad_risk:
        return f"risk_keys 越界 {bad_risk}"
    if row.get("sentiment") not in sc.VALID_SENTIMENT:
        return f"sentiment 越界 {row.get('sentiment')!r}"
    if row.get("relevance") not in sc.VALID_RELEVANCE:
        return f"relevance 越界 {row.get('relevance')!r}"
    # subject_hit 只在源行显式提供时校验（已带 output 的成品行没有该字段）
    if "subject_hit" in row and not row["subject_hit"]:
        return "subject_hit=false"
    ev = (existing_gold(row) or {}).get("evidence") or row.get("evidence_span") or ""
    if ev and ev not in text:
        return "evidence_span 非 input 连续子串"
    gold = gold_of(row)
    err = sc.validate_schema(gold)
    if err:
        return f"schema 校验: {err}"
    return None


def main():
    ap = argparse.ArgumentParser(description="SFT 数据格式对齐器 (纯规则, 零 LLM)")
    ap.add_argument("--src", required=True, help="原文+meta 源 jsonl")
    ap.add_argument("--out", default="data/train_synth_81_aligned.jsonl",
                    help="对齐产物路径（默认与 mix_sft_dataset --domain_file 的默认值衔接，"
                         "两处默认必须指向同一文件，否则流水线默认口径断链）")
    ap.add_argument("--report", default="", help="质检报告 json 路径 (留空则打印)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total, kept, drop_reasons = 0, 0, Counter()
    cot_source_counts = Counter({"inherited": 0, "none": 0})
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = flatten(json.loads(line))
            reason = check_row(row)
            if reason:
                drop_reasons[reason] += 1
                continue
            gold = gold_of(row)
            output, has_cot = render_output(row, gold)
            # 主体来源：gold JSON 的 subject（成品行与题面行统一走 gold_of）
            user_input = task_template.build_user_input(gold["subject"], row.get("input") or "")
            instruction = task_template.build_instruction(row["industry"])
            meta = {k: v for k, v in row.items()
                    if k not in ("instruction", "input", "output")}  # 大字段不进 meta（防重复膨胀）
            # 裸原文另存：input 已包 user 侧题面壳，grounding 断言必须用裸原文
            meta["input_raw"] = row.get("input") or ""
            meta["aligned_by"] = "align_sft_data"
            meta["cot_source"] = "inherited" if has_cot else "none"
            if not has_cot:
                cot_source_counts["none"] += 1
            else:
                cot_source_counts["inherited"] += 1
            fout.write(json.dumps(
                {"instruction": instruction, "input": user_input,
                 "output": output, "meta": meta},
                ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "src": str(src), "total": total,
        "kept": kept, "dropped": total - kept,
        "drop_reasons": dict(drop_reasons),
        "cot_source": dict(cot_source_counts),
        "empty_risk_injected": 0,  # 真值注入只针对源本就为空的样本; 本源 0 条, 不伪造空表
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
