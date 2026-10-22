#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发布门 CLI · 四道门独立判定 + 归因表落盘

为什么单独成脚本（评估手册缺口 5）:
  旧版 `evaluate_sft.py` 把四道门压成一个 `ok` 布尔 —— 门 1（历史错题修好了没）
  与门 3（接 L3 规则后安不安全）根本没有接口，等于这两道门永远假装通过；
  而且一个布尔把「哪道门没过」也吞了，修都不知道往哪修。

用法:
  # 门 1 + 门 2 + 门 4（门 3 需 L3 联动指标，不给则返回 N/A，绝不假装 PASS）
  python scripts/release_gate.py --gold gold.jsonl --pred pred.jsonl \
      --old_error_list old_errors.txt --out output/gate.json

  # 带在役基线（门 2 的退化对比）
  python scripts/release_gate.py --gold gold.jsonl --pred pred.jsonl \
      --baseline_json output/report_prev.json --out output/gate.json

  # 带 L3 联动指标（门 3）
  python scripts/release_gate.py --gold gold.jsonl --pred pred.jsonl \
      --l3_json output/l3_metrics.json --out output/gate.json

执行纪律: 任一门不过即中止发布；修复后**从门 1 全量重跑**，不补跑单门
（否则前置门的副作用被放过）。

纯标准库；判定逻辑全部在 lib/release_gate.py（本文件只做 I/O 与打印）。
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")

def _load_lib_module(name):
    """按文件路径加载 lib/ 模块 —— 避免与本 CLI 同名的 `release_gate` 撞名
    （unittest discover 时 scripts/ 在 sys.path 里，裸 import 会自我循环）。"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_LIB, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# lib/ 内部是裸包名互相 import，先把 lib/ 挂到 sys.path 保证它们能解析
sys.path.insert(0, _LIB)

release_gate = _load_lib_module("release_gate")
evaluate_gates = release_gate.evaluate_gates
overall_verdict = release_gate.overall_verdict

from schema_contract import (  # noqa: E402
    load_pred_line, parse_output, resolve_gold, unwrap_output_text,
    validate_schema,
)


def _load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_jsonl_text(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def _pred_text(raw: str) -> str:
    """委托 schema_contract.unwrap_output_text（剥壳逻辑唯一实现）。"""
    return unwrap_output_text(raw)


def _load_preds(path: str) -> list:
    """读预测文件 → 带解析标志的 6 字段 dict 列表（判定源仍是 schema_contract）。"""
    out = []
    for raw in _load_jsonl_text(path):
        pred, form = load_pred_line(raw)
        merged = dict(pred) if pred else {}
        if form == "text":
            text = _pred_text(raw)
            _, _, flags = parse_output(text)
            merged["valid_json"] = pred is not None
            merged["valid_tags"] = flags["valid_thought_tags"]
            merged["schema_error"] = flags["schema_error"]
            if pred is not None and merged["schema_error"] is None:
                merged["schema_error"] = validate_schema(pred)
        elif form == "dict":
            merged["valid_json"] = pred is not None
            merged["valid_tags"] = False          # 直接给 dict：无 CoT，闭合率不适用
            merged["schema_error"] = validate_schema(pred) if pred is not None else "JSON 解析失败"
        else:
            merged["valid_json"] = False
            merged["valid_tags"] = False
            merged["schema_error"] = "预测行无法解析"
        out.append(merged)
    return out


def _load_golds(path: str) -> list:
    """读金标文件 → 统一金标 dict 列表（output JSON 为准，meta 兜底）。"""
    golds = []
    for i, row in enumerate(_load_jsonl(path)):
        gold, _ = resolve_gold(row)
        meta = row.get("meta", {}) or {}
        gid = None
        for k in ("id", "event_id", "sample_id"):
            if row.get(k):
                gid = str(row[k])
                break
            if meta.get(k):
                gid = str(meta[k])
                break
        gold["id"] = gid if gid else f"#{i + 1}"
        golds.append(gold)
    return golds


def main():
    ap = argparse.ArgumentParser(description="舆情负面精判 · 发布门四道判定")
    ap.add_argument("--gold", required=True, help="金标文件（jsonl）")
    ap.add_argument("--pred", required=True, help="模型预测文件（jsonl）")
    ap.add_argument("--old_error_list", default=None,
                    help="门 1 用的上一版错题 id 列表（每行一个 id）")
    ap.add_argument("--baseline_json", default=None,
                    help="门 2 用的在役基线指标 JSON（可为 report.json 或纯指标 dict）")
    ap.add_argument("--l3_json", default=None,
                    help="门 3 用的 L3 规则联动指标 JSON（p0_recall / p0_precision + baseline_*）")
    ap.add_argument("--out", default="output/gate.json", help="门禁结果落盘路径")
    args = ap.parse_args()

    golds = _load_golds(args.gold)
    preds = _load_preds(args.pred)
    if len(golds) != len(preds):
        print(f"[FATAL] 样本数不一致: gold={len(golds)} pred={len(preds)}")
        sys.exit(2)

    baseline = None
    if args.baseline_json:
        baseline = _load_baseline(args.baseline_json)
    l3 = None
    if args.l3_json:
        with open(args.l3_json, encoding="utf-8") as f:
            l3 = json.load(f)

    old_error_ids = None
    if args.old_error_list:
        with open(args.old_error_list, encoding="utf-8") as f:
            old_error_ids = [l.strip() for l in f if l.strip()]

    gates = evaluate_gates(golds, preds, old_error_ids=old_error_ids,
                            baseline=baseline, l3=l3)
    verdict = overall_verdict(gates)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"gold": args.gold, "pred": args.pred,
                   "old_error_list": args.old_error_list,
                   "baseline_json": args.baseline_json,
                   "l3_json": args.l3_json},
        "n_samples": len(golds),
        "gates": {k: gates[k] for k in ("gate1", "gate2", "gate3", "gate4")},
        "attribution": gates["attribution"],
        "details": gates["details"],
        "verdict": verdict,
    }

    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("发布门四道判定（「与」关系：任一 FAIL 即中止发布）")
    print("-" * 70)
    print(f"  门 1 · 历史错题修复 : {gates['gate1']}")
    print(f"  门 2 · 金标不退化   : {gates['gate2']}")
    print(f"  门 3 · 全链路安全   : {gates['gate3']}")
    print(f"  门 4 · 格式合规     : {gates['gate4']}")
    print("-" * 70)
    print(f"  总判定: {'【PASS 可发布】' if verdict == 'PASS' else '【FAIL 中止发布】'}")
    if gates["attribution"]:
        print(f"\n  归因表（{len(gates['attribution'])} 条）:")
        for a in gates["attribution"][:20]:
            print(f"    [{a['gate']}] {a['sample_id']}: {a['reason']}")
    skipped = (gates["details"].get("gate2", {}) or {}).get("skipped") or []
    if skipped:
        print(f"\n  ⚠ 门 2 未参与判定的项（不可判定，不假装 PASS/FAIL）:")
        for s in skipped:
            print(f"    - {s}")
    print("=" * 70)
    print(f"[OK] 门禁结果已落盘: {path}")
    sys.exit(0 if verdict == "PASS" else 1)


def _load_baseline(path: str) -> dict:
    """从 report.json 或纯指标 dict 里取门 2 需要的四个基线指标。"""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if "sentiment" in obj or "hard_case" in obj or "risk" in obj:
        # evaluate_sft 的 report.json 结构
        sent = obj.get("sentiment", {}) or {}
        hc = obj.get("hard_case", {}) or {}
        rk = obj.get("risk", {}) or {}
        return {
            "negative_recall": sent.get("negative_recall"),
            "irony_recall": hc.get("irony_recall"),
            "macro_f1": hc.get("macro_f1"),
            "subset_accuracy": rk.get("subset_accuracy"),
        }
    return {k: obj[k] for k in ("negative_recall", "irony_recall",
                                "macro_f1", "subset_accuracy") if k in obj}


if __name__ == "__main__":
    main()
