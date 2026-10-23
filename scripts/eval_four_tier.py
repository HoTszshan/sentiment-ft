#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四档阶梯一键对照（评估手册 TODO 5.2 · 模块二 §2「跑什么」）

为什么单独成脚本:
  只报「微调前 vs 微调后」两档，说不清提升是哪一步带来的。
  四档控制变量对照把「提升归因到某一步」变成可计算项:
    T1 裸基座       —— 模型出厂状态（线上模板 = 完整输出协议模板）
    T2 提示词加固   —— 同 T1 + few-shot / 自一致性等提示词工程加强（可选档）
    T3 SFT          —— + SFT LoRA（线上模板）
    T4 GRPO         —— + RL LoRA（线上模板）
  单变量差值才有归因资格: T3−T1 = SFT 净增益; T4−T3 = GRPO 净增益;
  T2−T1 = 提示词工程增益; T3−T2 / T4−T2 是合计量，**只展示、不进归因**。
  注: T1 已带完整输出协议（线上模板=训练模板），故「纯提示词上限」由 T1 本身承担。

用法:
  python scripts/eval_four_tier.py \
      --gold data/mixed/sft_val.jsonl \
      --pred_base  output/pred_base.jsonl \
      --pred_prompt output/pred_prompt.jsonl \
      --pred_sft   output/pred_sft.jsonl \
      --pred_grpo  output/pred_grpo.jsonl \
      --out output/tier_report.json

  每份 pred 文件行序必须与 gold 严格对齐（probe_infer.py 保证）。
  任一档缺文件 → 该档记 "missing"，其余档照常出数，不中断。

输出:
  tier_report.json —— 四档 × 指标矩阵 + 逐档增量 + 归因标注
                       （SFT 主功 / GRPO 主功 / SFT 下界·GRPO 上界）
  终端同步打印人可读摘要。

指标计算复用 lib/eval_metrics.py（与 evaluate_sft.py 同一套纯函数，
**谁都不许再各写一份判定式**）。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from eval_metrics import (  # noqa: E402
    macro_f1,
    pa_fp_pair,
    subset_accuracy,
)
from schema_contract import (  # noqa: E402
    VALID_HARD_CASE,
    VALID_RELEVANCE,
    VALID_RISK,
    VALID_SENTIMENT,
    load_pred_line,
    resolve_gold,
)

TIER_KEYS = ("base", "prompt", "sft", "grpo")
TIER_LABELS = {
    "base": "T1 裸基座",
    "prompt": "T2 提示词加固",
    "sft": "T3 SFT",
    "grpo": "T4 GRPO",
}

# 单变量差值（归因资格）与合计量（只展示）
SINGLE_VAR_PAIRS = (
    ("base", "sft", "SFT 净增益（单变量: 只差权重）"),
    ("sft", "grpo", "GRPO 净增益（单变量: 只差 RL 权重）"),
    ("base", "prompt", "提示词工程增益（单变量: 只差提示词加固）"),
)
CONFOUNDED_PAIRS = (
    ("prompt", "sft", "SFT vs 提示词天花板（合计量，含权重+提示词两变量）"),
    ("prompt", "grpo", "整套 vs prompt 天花板（合计量，含权重+提示词两变量）"),
)

REPORT_SCHEMA_VERSION = 1


# ============================================================
# 逐档算分（与 evaluate_sft.py 模式 B 同一套口径）
# ============================================================

def _pct(x: float) -> float:
    return round(float(x), 2)


def score_one_tier(gold_rows: List[Dict[str, Any]],
                   pred_raw_lines: List[str]) -> Dict[str, Any]:
    """对一档预测算全量指标。返回各指标的原始 dict（不取整，供差值计算）。"""
    golds: List[Dict[str, Any]] = []
    preds: List[Optional[Dict[str, Any]]] = []
    for grow, pline in zip(gold_rows, pred_raw_lines):
        gold, _ = resolve_gold(grow)
        golds.append(gold)
        pred, _form = load_pred_line(pline)
        preds.append(pred)

    g_sent = [g.get("sentiment") for g in golds]
    p_sent = [(p or {}).get("sentiment") for p in preds]
    g_hc = [g.get("hard_case_type") for g in golds]
    p_hc = [(p or {}).get("hard_case_type") for p in preds]
    g_rel = [g.get("relevance") for g in golds]
    p_rel = [(p or {}).get("relevance") for p in preds]
    g_rk = [g.get("risk_categories") or [] for g in golds]
    p_rk = [(p or {}).get("risk_categories") or [] for p in preds]

    sent_stat = macro_f1(g_sent, p_sent, sorted(VALID_SENTIMENT), VALID_SENTIMENT)
    hc_stat = macro_f1(g_hc, p_hc, sorted(VALID_HARD_CASE), VALID_HARD_CASE)
    rel_stat = macro_f1(g_rel, p_rel, sorted(VALID_RELEVANCE), VALID_RELEVANCE)
    rk_stat = subset_accuracy(g_rk, p_rk, VALID_RISK)
    pa = pa_fp_pair(g_sent, p_sent)
    irony = hc_stat["per_class"].get("irony", {})

    return {
        "n_samples": len(golds),
        "json_parse_ok": sum(1 for p in preds if p is not None),
        "sentiment_macro_f1": sent_stat["macro_f1"],
        "negative_recall": pa["negative_recall"],
        "false_alarm_rate": pa["false_alarm_rate"],
        "negative_recall_defined": pa["recall_defined"],
        "hard_case_macro_f1": hc_stat["macro_f1"],
        # support=0 → 金标无 irony 样本，召回不可判定；报 None 让差值跳过，
        # 不许伪装成「irony 召回 0%」
        "irony_recall": (irony.get("recall") if irony.get("support", 0) > 0 else None),
        "irony_support": irony.get("support", 0),
        "relevance_macro_f1": rel_stat["macro_f1"],
        "risk_subset_accuracy": rk_stat["subset_accuracy"],
        "risk_empty_precision": rk_stat["empty_precision"],
    }


def _delta(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]],
           metric_keys: tuple) -> Optional[Dict[str, float]]:
    """两档差值: b − a。任一档缺失 → None（不可计算，不编 0）。"""
    if a is None or b is None:
        return None
    out: Dict[str, float] = {}
    for k in metric_keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = _pct(vb - va)
    return out


def _attribution(metric: str) -> str:
    """主功归属（评估手册 §2「主功归因」表）。"""
    if metric == "json_parse_ok":
        return "SFT 主功 · GRPO 守门"
    if metric in ("risk_subset_accuracy", "hard_case_macro_f1"):
        return "GRPO 主功（R_accuracy 非对称罚抬升）"
    if metric in ("sentiment_macro_f1", "negative_recall", "false_alarm_rate"):
        return "SFT 定下界 · GRPO 提上界"
    return "SFT 主功"


def _dump(report: Dict[str, Any], out_path: str) -> None:
    path = os.path.abspath(out_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] 四档对照报告已落盘: {path}")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="四档阶梯一键对照（T1 基座 / T2 提示词加固 / T3 SFT / T4 GRPO）")
    ap.add_argument("--gold", required=True, help="金标文件（jsonl，四档共用同一份）")
    ap.add_argument("--pred_base", help="T1 裸基座预测文件")
    ap.add_argument("--pred_prompt", help="T2 提示词加固档预测文件（可选）")
    ap.add_argument("--pred_sft", help="T3 SFT 预测文件")
    ap.add_argument("--pred_grpo", help="T4 GRPO 预测文件")
    ap.add_argument("--out", default="output/tier_report.json",
                    help="报告落盘路径（默认 output/tier_report.json）")
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold_rows = [json.loads(l) for l in f if l.strip()]
    n_gold = len(gold_rows)

    pred_paths = {"base": args.pred_base, "prompt": args.pred_prompt,
                  "sft": args.pred_sft, "grpo": args.pred_grpo}

    tiers: Dict[str, Optional[Dict[str, Any]]] = {}
    for key in TIER_KEYS:
        path = pred_paths.get(key)
        if not path or not os.path.exists(path):
            tiers[key] = None
            continue
        with open(path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        if len(lines) != n_gold:
            print(f"[FATAL] {TIER_LABELS[key]} 行数不一致: pred={len(lines)} gold={n_gold}，"
                  f"该档记 missing（行序必须与 gold 严格对齐）")
            tiers[key] = None
            continue
        tiers[key] = score_one_tier(gold_rows, lines)

    metric_keys = ("json_parse_ok", "sentiment_macro_f1", "negative_recall",
                   "false_alarm_rate", "hard_case_macro_f1", "irony_recall",
                   "relevance_macro_f1", "risk_subset_accuracy",
                   "risk_empty_precision")
    # 百分比指标的 key（json_parse_ok 是条数，不做百分比差值展示）
    pct_keys = tuple(k for k in metric_keys if k != "json_parse_ok")

    deltas = {}
    for a_key, b_key, label in SINGLE_VAR_PAIRS + CONFOUNDED_PAIRS:
        deltas[f"{b_key}-{a_key}"] = {
            "label": label,
            "single_variable": (a_key, b_key) in [(x[0], x[1]) for x in SINGLE_VAR_PAIRS],
            "delta": _delta(tiers[a_key], tiers[b_key], metric_keys),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold": os.path.abspath(args.gold),
        "n_gold": n_gold,
        "tiers": {TIER_LABELS[k]: (tiers[k] if tiers[k] is not None else "missing")
                  for k in TIER_KEYS},
        "metric_attribution": {k: _attribution(k) for k in metric_keys},
        "deltas": deltas,
        "note": "只认单变量差值（single_variable=true）作归因；合计量仅展示。",
    }

    # ---- 终端摘要 ----
    print("=" * 76)
    print("四档阶梯对照")
    print(f"  gold: {args.gold}  ({n_gold} 条)")
    print("=" * 76)

    header = f"{'指标':<24}" + "".join(f"{TIER_LABELS[k]:>14}" for k in TIER_KEYS)
    print(header)
    print("-" * 76)
    for k in metric_keys:
        row = f"{k:<24}"
        for t in TIER_KEYS:
            v = tiers[t].get(k) if tiers[t] else None
            row += f"{v:>13.2f} " if isinstance(v, (int, float)) else f"{'missing':>13} "
        print(row)
    print("-" * 76)
    print("\n差值（只认单变量差值作归因; 合计量仅展示）:")
    for pair_key, d in deltas.items():
        mark = "✓ 单变量" if d["single_variable"] else "✗ 合计量"
        dv = d["delta"]
        if dv is None:
            print(f"  {pair_key:<14} {mark}  缺档不可算")
            continue
        print(f"  {pair_key:<14} {mark}  {d['label']}")
        for k in pct_keys:
            if dv.get(k) is None:
                # support=0 等不可判定项：差值跳过，明确记 N/A，不伪装 0.00pp
                print(f"      {k:<26} {'N/A':>8}     [{_attribution(k)}]")
                continue
            print(f"      {k:<26} {dv.get(k, 0.0):+8.2f}pp   [{_attribution(k)}]")

    print("=" * 76)
    _dump(report, args.out)
    sys.exit(0)


if __name__ == "__main__":
    main()
