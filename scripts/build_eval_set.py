#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金标评测集构建 · 三方异族一致性打标 (摸底 → 铺量)

唯一真理源: materials/DataPipeline_SOP_v4.md §3.2/§3.3 (IAA 校准流程, κ ≥ 0.7)

三族 (异族铁律):
  A 深度求索 deepseek-v4-pro   ← 语料池里现成的打标 (meta / output)
  B 第三方 1 (如 claude-opus-5.5)
  C 第三方 2 (如 gpt-6-luna)
  Qwen-Max 是写手, **不参与裁决** (先排除生成端自洽)

处置三档:
  1. A=B=C 全字段一致         → 入金标 (eval_gold_probe.jsonl)
  2. 仅 hard_case 分歧, 其余一致 → 边界样本 (eval_boundary.jsonl) —— 不当金标, 留给 DPO / 难例复盘
  3. sentiment 或 risk 分歧     → 丢弃 (真歧义, 进训练只会教错)

口径: 三方分歧率本身是可写指标; **κ < 0.6 说明是法典歧义而非模型差异, 必须回炉标注规范**。

用法 (先只摸底 20 条):
  python3 scripts/build_eval_set.py \
      --src output/corpus/eval_probe_pool.jsonl \
      --out_dir output/eval --limit 20 --concurrency 4 \
      --third1_provider openai --third1_model claude-opus-5.5 \
      --third2_provider openai --third2_model gpt-6-luna
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from llm_client import LLMClient                                   # noqa: E402
from corpus_schema import load_prompts, schema_to_inline           # noqa: E402
from reward_fns import split_completion, validate_schema            # noqa: E402


# ============================================================
# 打标
# ============================================================

def parse_annotation(raw: str) -> Optional[Dict[str, Any]]:
    """从 (可能带 CoT 的) 打标输出里取出 5 字段 JSON。"""
    _, pred, _ = split_completion(raw if "<|begin_of_thought|>" in (raw or "")
                                  else f"<|begin_of_thought|>{raw}<|end_of_thought|>{raw}")
    if pred is None:
        # 退路: 直接是 JSON
        t = (raw or "").strip()
        if t.startswith("{") and t.rfind("}") != -1:
            try:
                pred = json.loads(t[: t.rfind("}") + 1])
            except Exception:
                pred = None
    if not isinstance(pred, dict):
        return None
    if validate_schema(pred):
        return None
    return pred


def anno_one(client: LLMClient, sys_prompt: str, text: str,
             subject: str, industry: str) -> Tuple[Optional[Dict], str]:
    """一次独立打标，返回 (pred, err)。"""
    user = (
        f"【目标主体 target_subject】{subject}\n"
        f"【所属行业 industry】{industry}\n"
        f"【行业风险二级词表 custom_schema】\n{schema_to_inline(industry)}\n\n"
        f"【待标注原文】\n{text}"
    )
    try:
        raw = client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user}],
            temperature=0, max_tokens=3000,
        )
    except Exception as e:
        return None, f"API 失败: {e}"
    pred = parse_annotation(raw)
    if pred is None:
        return None, "输出不满足闭集 schema (缺字段/越界/非法 JSON)"
    return pred, ""


# ============================================================
# 一致性判定
# ============================================================

JUDGE_FIELDS = ["sentiment", "relevance", "hard_case_type"]


def risk_key(r: Any) -> Tuple[str, ...]:
    return tuple(sorted(r if isinstance(r, (list, tuple)) else []))


def classify(trio: Tuple[Dict, Optional[Dict], Optional[Dict]]) -> str:
    """
    A=deepseek, B/C=第三方。返回 'gold' / 'boundary' / 'drop'。
    """
    a, b, c = trio
    if b is None or c is None:
        return "drop"
    if any(a.get(f) != b.get(f) or a.get(f) != c.get(f) for f in ["sentiment", "relevance"]):
        return "drop"
    if risk_key(a.get("risk_categories")) != risk_key(b.get("risk_categories")) or \
            risk_key(a.get("risk_categories")) != risk_key(c.get("risk_categories")):
        return "drop"
    if a.get("hard_case_type") == b.get("hard_case_type") == c.get("hard_case_type"):
        return "gold"
    return "boundary"      # 仅难点类型分歧


def pair_agreement(pairs: List[Tuple[Any, Any]]) -> Tuple[int, int, float]:
    """两两一致的条数/总数/一致率。"""
    if not pairs:
        return 0, 0, 0.0
    ok = sum(1 for x, y in pairs if x == y)
    return ok, len(pairs), ok / len(pairs)


def cohen_kappa(pairs: List[Tuple[Any, Any]]) -> Optional[float]:
    """Cohen's Kappa。样本过少或 p_e=1 时返回 None。"""
    n = len(pairs)
    if n < 5:
        return None
    po, _, _ = pair_agreement(pairs)
    po /= n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    if abs(1 - pe) < 1e-9:
        return None
    return round((po - pe) / (1 - pe), 4)


def agreement_report(rows: List[Dict]) -> Dict[str, Any]:
    """逐字段、逐模型对的 κ 与一致率。"""
    report: Dict[str, Any] = {"n": len(rows)}
    pairs = [("A-DeepSeek", "B"), ("A-DeepSeek", "C"), ("B", "C")]
    for f in JUDGE_FIELDS + ["risk"]:
        per = {}
        for ka, kb in pairs:
            vals = []
            for r in rows:
                va = r["gold"][f] if f != "risk" else risk_key(r["gold"].get("risk_categories"))
                vb = r[ka][f] if f != "risk" else risk_key(r[ka].get("risk_categories"))
                vc = r[kb][f] if f != "risk" else risk_key(r[kb].get("risk_categories"))
                if vb is None or vc is None:
                    continue
                vals.append((vb, vc))
            ok, n, rate = pair_agreement(vals)
            per[f"{ka}/{kb}"] = {"n": n, "agree": ok, "rate": round(rate, 4),
                                 "kappa": cohen_kappa(vals)}
        report[f] = per
    return report


def verdict(report: Dict[str, Any]) -> str:
    ks = [report[f][k]["kappa"] for f in JUDGE_FIELDS
          for k, v in report[f].items() if v["kappa"] is not None]
    if not ks:
        return "样本量不足，无法计算 κ（先跑 80 条再看）"
    m = min(ks)
    if m >= 0.7:
        return f"PASS · 最低 κ={m} ≥ 0.7，法典口径一致，金标可上岗"
    if m >= 0.6:
        return f"BOUNDARY · 最低 κ={m} ∈ [0.6,0.7)，边界 case 需人工复核后放行"
    return f"FAIL · 最低 κ={m} < 0.6 —— 多半是**标注法典本身有歧义**（而非模型差异），必须回炉重写标注规范"


# ============================================================
# 主流程
# ============================================================

def subject_of(instruction: str) -> str:
    import re
    m = re.search(r"目标主体[：:]\s*【(.+?)】", instruction or "")
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser(description="三方异族一致性 → 金标评测集")
    ap.add_argument("--src", required=True, help="评测语料池 jsonl (corpus 格式)")
    ap.add_argument("--out_dir", default="output/eval")
    ap.add_argument("--limit", type=int, default=20, help="摸底 20；铺量 80")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--third1_provider", default="openai")
    ap.add_argument("--third1_model", default="claude-opus-5.5")
    ap.add_argument("--third1_base_url", default=None, help="中转站地址，留空用 provider 默认")
    ap.add_argument("--third2_provider", default="openai")
    ap.add_argument("--third2_model", default="gpt-6-luna")
    ap.add_argument("--third2_base_url", default=None)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[FATAL] 找不到评测语料池: {src}")

    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = rows[: args.limit]
    if not rows:
        raise SystemExit("[FATAL] 语料池为空")

    _, anno_sys = load_prompts()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"金标评测集 · 三方异族一致性打标 (摸底 {len(rows)} 条)")
    print(f"  A 语料池打标 : dashscope/deepseek-v4-pro  (现成)")
    print(f"  B 第三方     : {args.third1_provider}/{args.third1_model}"
          f"{(' @ ' + args.third1_base_url) if args.third1_base_url else ''}")
    print(f"  C 第三方     : {args.third2_provider}/{args.third2_model}"
          f"{(' @ ' + args.third2_base_url) if args.third2_base_url else ''}")
    print(f"  Qwen-Max     : 写手，不参与裁决 (排除生成端自洽)")
    print("=" * 74)

    c1 = LLMClient(provider=args.third1_provider, model=args.third1_model,
                    base_url=args.third1_base_url)
    c2 = LLMClient(provider=args.third2_provider, model=args.third2_model,
                    base_url=args.third2_base_url)

    gold, boundary, dropped, records = [], [], [], []
    import concurrent.futures as cf

    def work(item):
        row, subject, industry, a = item
        b, eb = anno_one(c1, anno_sys, row["input"], subject, industry)
        c, ec = anno_one(c2, anno_sys, row["input"], subject, industry)
        return row, a, b, c, (eb, ec)

    items = []
    for row in rows:
        # A: 语料池现成打标 (output 里的 JSON)
        _, pred_a, _ = split_completion(row.get("output", ""))
        if pred_a is None:
            dropped.append((row.get("event_id", "?"), "A(DeepSeek) 解析失败"))
            continue
        subject = subject_of(row.get("instruction", ""))
        industry = row.get("meta", {}).get("industry", "")
        if not subject or not industry:
            dropped.append((row.get("event_id", "?"), "缺主体/行业"))
            continue
        items.append((row, subject, industry, pred_a))

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        results = list(ex.map(work, items))

    for row, a, b, c, errs in results:
        eid = row.get("event_id", "?")
        if any(errs):
            dropped.append((eid, f"第三方打标失败: {[e for e in errs if e]}"))
            continue
        cls = classify((a, b, c))
        rec = {"event_id": eid, "input": row["input"],
               "industry": row.get("meta", {}).get("industry"),
               "gold": a, "B": b, "C": c, "verdict": cls}
        records.append(rec)
        if cls == "gold":
            # 沿用 SFT 输出格式，便于 evaluate_sft 模式A/B 直接读
            gold.append({
                "instruction": row["instruction"], "input": row["input"],
                "output": row["output"], "meta": {**row.get("meta", {}),
                                                  "eval_gold": True,
                                                  "iaa": "triple_agree"}},
            )
        elif cls == "boundary":
            boundary.append(rec)
        else:
            reasons = []
            for f in JUDGE_FIELDS:
                if not (a.get(f) == b.get(f) == c.get(f)):
                    reasons.append(f"{f}: A={a.get(f)} B={b.get(f)} C={c.get(f)}")
            if risk_key(a["risk_categories"]) != risk_key(b["risk_categories"]) or \
                    risk_key(a["risk_categories"]) != risk_key(c["risk_categories"]):
                reasons.append(f"risk: A={a['risk_categories']} B={b['risk_categories']} C={c['risk_categories']}")
            dropped.append((eid, "; ".join(reasons)[:160]))

    # ---- 落盘 ----
    def dump(path, data):
        p = out_dir / path
        with open(p, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return p

    gp = dump("eval_gold_probe.jsonl", gold)
    bp = dump("eval_boundary.jsonl", boundary)

    report = agreement_report(records) if records else {"n": 0}
    report["counts"] = {"total": len(rows), "gold": len(gold),
                        "boundary": len(boundary), "dropped": len(dropped)}
    report["verdict"] = verdict(report) if records else "无有效三方记录"
    rp = out_dir / "eval_agreement_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 终端报告 ----
    print("\n--- 逐字段两两一致率 / κ ---")
    for f in JUDGE_FIELDS + ["risk"]:
        print(f"  {f}:")
        for k, v in (report.get(f) or {}).items():
            ks = v["kappa"]
            print(f"    {k:<18} n={v['n']:<4} 一致率={v['rate']:.3f}  "
                  f"κ={'-' if ks is None else ks}")

    print(f"\n--- 分流 ---")
    print(f"  语料池          : {len(rows)} 条")
    print(f"  金标 (三方全一致): {len(gold)} 条 -> {gp}")
    print(f"  边界 (仅难点分歧): {len(boundary)} 条 -> {bp}   ← 不当金标，留给 DPO/难例复盘")
    print(f"  丢弃            : {len(dropped)} 条")
    for eid, why in dropped[:6]:
        print(f"      {eid}: {why}")
    print(f"\n  报告: {rp}")
    print(f"\n  判定: {report['verdict']}")
    print("\n⚠️ 这是「第三方一致性金标」，不是人标金标；简历只能写一致性口径，")
    print("   plan 里锁死的 500 条人标金标仍是 [待回填]。")
    print("=" * 74)
    print(f"用量 B({args.third1_model}): {c1.usage.summary()}")
    print(f"用量 C({args.third2_model}): {c2.usage.summary()}")


if __name__ == "__main__":
    main()
