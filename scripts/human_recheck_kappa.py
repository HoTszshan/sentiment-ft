#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工复标 κ 验收（评估手册 TODO 5.10 · 模块一 ①「人工复标」）

为什么单独成脚本:
  三方模型互评（build_eval_set.py）只能证明「三个模型口径一致」，
  不能证明「模型口径 = 人的口径」。所以还要抽一批样本由人独立重标，
  与模型标注算 Cohen's κ —— **人与模型 κ ≥ 0.74 才放行**，
  不达标退回重打标，分歧样本由专家按裁决链终审。

口径（与 GRPO 数据准备实战手册 §金标标注 一致）:
  - 抽样约 200 条，**背对背盲复标**（复标人看不到模型标签）；
  - 比对字段: sentiment / relevance / hard_case_type / risk_categories
    （与三方互评的 JUDGE_FIELDS 同一套，risk 多标签展开成集合比对）;
  - κ 计算复用 build_eval_set.cohen_kappa（**谁都不许再写一份 κ**）;
  - 分母不足（某对齐字段有效样本 < 5）→ 该字段 κ 记 None 并记账，不假装通过。

输入文件格式（复标结果 jsonl，每行一条）:
  {"id": "...",
   "model":  {"sentiment": "...", "relevance": "...", "hard_case_type": "...",
              "risk_categories": [...]},
   "human":  {"sentiment": "...", "relevance": "...", "hard_case_type": "...",
              "risk_categories": [...]}}
  human 侧缺字段 → 该字段不参与该条的 κ 对齐（自动跳过，不计错）。

用法:
  python scripts/human_recheck_kappa.py --recheck output/eval/human_recheck_200.jsonl \
      --out output/eval/recheck_kappa_report.json

判定输出:
  PASS     · 全字段最低 κ ≥ 0.74 → 放行
  BOUNDARY · 最低 κ ∈ [0.6, 0.74) → 分歧样本人工复核后再放行
  FAIL     · 最低 κ < 0.6 → 退回重打标
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from corpus_schema import load_prompts  # noqa: E402  (保持与 build_eval_set 同一加载路径; 未直接用)
from schema_contract import (  # noqa: E402
    VALID_HARD_CASE,
    VALID_RELEVANCE,
    VALID_SENTIMENT,
)

# κ 计算直接复用三方互评的同一份实现（单一判定源）
from build_eval_set import JUDGE_FIELDS, cohen_kappa, risk_key  # noqa: E402

# 人工复标放行门槛（评估手册模块一 ①: κ ≥ 0.74）
THRESH_HUMAN_KAPPA = 0.74
# BOUNDARY 下限（低于此值直接 FAIL 退回）
THRESH_BOUNDARY_KAPPA = 0.60
# 单字段最少有效样本（低于此值 κ 记 None，与 cohen_kappa 内部 n<5 一致）
MIN_KAPPA_N = 5


def _field_pairs(rows: List[Dict[str, Any]], field: str
                 ) -> List[Tuple[Any, Any]]:
    """取某字段 (human, model) 的有效对齐样本。缺字段 / 缺侧的样本跳过。"""
    pairs: List[Tuple[Any, Any]] = []
    for r in rows:
        h = (r.get("human") or {}).get(field)
        m = (r.get("model") or {}).get(field)
        if h is None or m is None:
            continue
        if field == "risk":
            pairs.append((risk_key(h), risk_key(m)))
        else:
            pairs.append((h, m))
    return pairs


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐字段算 κ + 一致率（复标口径: human 为参照侧）。"""
    report: Dict[str, Any] = {"n_input": len(rows)}
    fields = JUDGE_FIELDS + ["risk"]
    for f in fields:
        pairs = _field_pairs(rows, f)
        n = len(pairs)
        ok = sum(1 for h, m in pairs if h == m)
        entry: Dict[str, Any] = {
            "n": n,
            "agree": ok,
            "rate": round(ok / n, 4) if n else None,
            "kappa": cohen_kappa(pairs) if n >= MIN_KAPPA_N else None,
        }
        if n and n < MIN_KAPPA_N:
            entry["skipped_reason"] = f"有效样本 {n} < {MIN_KAPPA_N}，κ 不可判定"
        report[f] = entry
    return report


def verdict(report: Dict[str, Any]) -> str:
    fields = JUDGE_FIELDS + ["risk"]
    ks = [(f, report[f]["kappa"]) for f in fields
          if report[f].get("kappa") is not None]
    # 被跳过的字段（κ=None，样本不足）必须点名 —— 不许静默跳过后假装全字段通过
    skipped = [f for f in fields if report[f].get("kappa") is None]
    _suffix = f"（跳过: {', '.join(skipped)}）" if skipped else ""
    if not ks:
        return ("FAIL · 无任何字段可判 κ（样本量不足或字段全缺）—— "
                "先补齐复标样本再验")
    worst_field, m = min(ks, key=lambda kv: kv[1])
    if m >= THRESH_HUMAN_KAPPA:
        return (f"PASS · 最低 κ={m}（{worst_field}）≥ {THRESH_HUMAN_KAPPA}，"
                f"人与模型口径一致，放行{_suffix}")
    if m >= THRESH_BOUNDARY_KAPPA:
        return (f"BOUNDARY · 最低 κ={m}（{worst_field}）∈ "
                f"[{THRESH_BOUNDARY_KAPPA},{THRESH_HUMAN_KAPPA})，"
                f"分歧样本人工复核后再放行{_suffix}")
    return (f"FAIL · 最低 κ={m}（{worst_field}）< {THRESH_BOUNDARY_KAPPA} —— "
            f"模型标注与人的口径系统性分歧，退回重打标{_suffix}")


def _dump(report: Dict[str, Any], out_path: str) -> None:
    path = os.path.abspath(out_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] 复标 κ 报告已落盘: {path}")


def main():
    ap = argparse.ArgumentParser(description="人工复标 κ 验收（κ ≥ 0.74 才放行）")
    ap.add_argument("--recheck", required=True,
                    help="复标结果 jsonl（每行含 model / human 两侧标签）")
    ap.add_argument("--out", default="output/eval/recheck_kappa_report.json",
                    help="报告落盘路径")
    args = ap.parse_args()

    if not os.path.exists(args.recheck):
        raise SystemExit(f"[FATAL] 找不到复标文件: {args.recheck}")

    with open(args.recheck, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        raise SystemExit("[FATAL] 复标文件为空")

    report = build_report(rows)
    report["threshold"] = {"pass_kappa": THRESH_HUMAN_KAPPA,
                           "boundary_kappa": THRESH_BOUNDARY_KAPPA}
    report["verdict"] = verdict(report)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print(f"人工复标 κ 验收（背对背盲复标 · 放行线 κ ≥ {THRESH_HUMAN_KAPPA}）")
    print(f"  复标样本: {len(rows)} 条")
    print("-" * 70)
    for f in JUDGE_FIELDS + ["risk"]:
        e = report[f]
        k = e["kappa"]
        print(f"  {f:<16} n={e['n']:<5} 一致率="
              f"{'-' if e['rate'] is None else format(e['rate'], '.3f')}  "
              f"κ={'-' if k is None else k}"
              f"{'  (' + e['skipped_reason'] + ')' if 'skipped_reason' in e else ''}")
    print("-" * 70)
    print(f"  判定: {report['verdict']}")
    print("=" * 70)

    _dump(report, args.out)
    sys.exit(0 if report["verdict"].startswith("PASS") else 1)


if __name__ == "__main__":
    main()
