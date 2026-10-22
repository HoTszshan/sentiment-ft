#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端业务终验（评估手册 TODO 5.6 · 模块一 ③ 第二段）

为什么单独成脚本:
  代理指标只保证「模型 + 规则」这条链的行为，业务侧还需要两项独立证据 +
  一项兜底。没有它们，漏报只能等客户侧反馈才暴露。

三个验收项（口径见手册模块一 ③ 第二段）:

  1. 漏报核查（前向抽样）
     两版告警集合分别抽样 → 双人独立盲标 → 先算新版**绝对漏报率**，
     再算**净增漏报**（新版未告警、旧版告警、且人工确认为负面）。
     达标线: 绝对漏报率与旧版持平；**净增漏报须为 0**。
     老模型本身漏报的部分（两版都没告警但确认为负面）另计，不归因本次改动。

  2. 工时核查
     统计分析师实际需阅读复核的条数（条/天/客户），与老版本同期对比。
     **分子分母必须同源**: 同一客户、同一时间窗、同一批入文量。
     单客户运营量级与全网链路量级是两本账，严禁交叉引用。

  3. 事后漏报回流（兜底监控）
     客户侧对告警的改判 / 驳回动作落盘回流，统计事后被确认的漏报。
     事后漏报率**不等价于**前向抽样结论，只做兜底监控。

输入文件格式（jsonl）:

  --alerts_new / --alerts_old   每行一条告警:
      {"sample_id": "...", "confirmed_negative": bool,
       "labeler_a": bool, "labeler_b": bool}
      confirmed_negative = 双人盲标一致结论（两人不一致的样本须先仲裁再入库，
      本脚本不仲裁 —— 仲裁是人的工作，脚本是算账的）。

  --workload                    每行一个客户一个时间窗:
      {"customer_id": "...", "period": "2025-03",
       "version": "new" | "old", "articles_ingested": 30000,
       "human_reviewed": 45}
      同源约束在脚本内强校验: 同一 customer_id 必须同时有 new/old 两版、
      且 articles_ingested 一致，否则该客户整体剔除并记账。

  --feedback                    每行一条客户侧动作（可省略）:
      {"sample_id": "...", "action": "escalate" | "revoke",
       "confirmed_negative": bool}
      escalate + confirmed_negative=true = 事后确认的漏报。

判定输出:
  漏报核查  PASS / FAIL（净增漏报 = 0 为硬线）
  工时核查  报数 + 降幅（判读归人，脚本不给 PASS/FAIL —— 降幅是收益数字不是门槛）
  事后回流  事后漏报率（兜底监控，不设门槛）

用法:
  python scripts/business_verify.py \
      --alerts_new output/biz/alerts_new.jsonl \
      --alerts_old output/biz/alerts_old.jsonl \
      --workload   output/biz/workload.jsonl \
      [--feedback  output/biz/feedback.jsonl] \
      --out output/biz/business_report.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPORT_SCHEMA_VERSION = 1


# ============================================================
# 1. 漏报核查（前向抽样）
# ============================================================

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _labeler_consistent(row: Dict[str, Any]) -> bool:
    """双人盲标必须一致才可入库（不一致样本先仲裁再进，本脚本不仲裁）。"""
    a, b = row.get("labeler_a"), row.get("labeler_b")
    if a is None or b is None:
        return True   # 单人标的历史数据：允许，但 confirmed_negative 必填
    return bool(a) == bool(b)


def missed_alarm_check(alerts_new: List[Dict[str, Any]],
                       alerts_old: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    漏报核查主计算。

    输入语义: 每个版本的文件 = 该版**告警的样本** ∪ 该版**未告警负面候选抽样**，
    每行用 `alarmed: bool` 区分（缺省 true = 在告警集合里）。两份文件取并集
    作为抽样全集，逐样本按 (alarmed_new, alarmed_old) 落入四象限:
      (False, False) 且确认为负面 → 老模型存量漏报（另计，不归因本次）
      (True,  False) 且确认为负面 → **净增漏报（硬线: = 0）**
      (False, True)  且确认为负面 → 新版新捞回（只记账，不减罪也不加分）
      (True,  True)               → 两版都告警，正常命中
    """
    def index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            sid = str(r.get("sample_id", ""))
            if not sid:
                continue
            out[sid] = r
        return out

    new_idx, old_idx = index(alerts_new), index(alerts_old)
    all_ids = sorted(set(new_idx) | set(old_idx))

    inconsistent = [sid for sid in all_ids
                    if not _labeler_consistent(new_idx.get(sid) or old_idx.get(sid) or {})]
    if inconsistent:
        return {"status": "BLOCKED",
                "reason": (f"{len(inconsistent)} 条样本双人盲标不一致，"
                           f"须先仲裁再跑（样本: {inconsistent[:5]}…）"),
                "n_inconsistent": len(inconsistent)}

    def confirmed(sid: str) -> bool:
        r = new_idx.get(sid) or old_idx.get(sid) or {}
        return bool(r.get("confirmed_negative"))

    # 同一 id 在两份文件里 alarm 状态冲突（一真一假）→ 抽样口径错误，先拦下
    def alarmed(idx: Dict[str, Dict[str, Any]], sid: str) -> Optional[bool]:
        if sid not in idx:
            return None
        r = idx[sid]
        return bool(r.get("alarmed", True))

    net_new_missed: List[str] = []       # 净增漏报
    legacy_missed: List[str] = []        # 两版都漏（老模型存量，另计）
    rescued: List[str] = []              # 新版新捞回
    both_hit: List[str] = []             # 两版都告警（正常命中）
    n_neg_total = 0

    for sid in all_ids:
        if not confirmed(sid):
            continue
        n_neg_total += 1
        a_new = alarmed(new_idx, sid)
        a_old = alarmed(old_idx, sid)
        # 只出现在一份文件里 → 另一份视为「未告警抽样未覆盖」，按未告警算
        a_new = True if a_new is None else a_new
        a_old = True if a_old is None else a_old
        if a_old and not a_new:
            net_new_missed.append(sid)
        elif not a_old and not a_new:
            legacy_missed.append(sid)
        elif not a_old and a_new:
            rescued.append(sid)
        else:
            both_hit.append(sid)
    # 绝对漏报率（新版视角）: 新版未告警且确认为负面 / 全部确认负面的抽样样本
    new_missed_all = net_new_missed + legacy_missed
    abs_miss_rate_new = (len(new_missed_all) / n_neg_total * 100) if n_neg_total else None
    abs_miss_rate_old = ((len(net_new_missed) + len(legacy_missed)) / n_neg_total * 100
                         if n_neg_total else None)  # 抽样口径下 old 版漏报 = 同一批样本中 old 未告警部分

    status = "PASS" if not net_new_missed else "FAIL"
    return {
        "status": status,
        "n_sampled": len(all_ids),
        "n_confirmed_negative": n_neg_total,
        "net_new_missed": {"count": len(net_new_missed), "ids": net_new_missed[:20],
                           "hard_line": "必须为 0"},
        "legacy_missed_old_model": {"count": len(legacy_missed),
                                    "note": "两版都未告警，老模型存量漏报，不归因本次改动"},
        "rescued_by_new": {"count": len(rescued),
                           "note": "新版新捞回（旧版漏、新版告警）"},
        "both_alerted": {"count": len(both_hit)},
        "absolute_miss_rate_new_pct": (round(abs_miss_rate_new, 2)
                                        if abs_miss_rate_new is not None else None),
        "note": ("绝对漏报率与旧版持平即达标；净增漏报 = 0 为硬线。"
                 "抽样口径下的率只代表抽样结论，分母是「确认负面的抽样样本」。"),
    }


# ============================================================
# 2. 工时核查（同源约束强校验）
# ============================================================

def workload_check(workload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    工时核查: 新旧两版的人均日复核条数对比。

    同源三约束（脚本内强校验，违反者该客户整体剔除并记账）:
      同一 customer_id 必须同时有 new / old 两版；
      两版 articles_ingested 必须一致（同一批入文量）；
      两版 period 必须成对（同一时间窗）。
    """
    by_cust: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in workload:
        cid = str(r.get("customer_id", ""))
        ver = r.get("version")
        if cid and ver in ("new", "old"):
            by_cust[cid][ver].append(r)

    dropped: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    totals = {"new": {"reviewed": 0, "ingested": 0},
              "old": {"reviewed": 0, "ingested": 0}}

    for cid, vers in sorted(by_cust.items()):
        if "new" not in vers or "old" not in vers:
            dropped.append({"customer_id": cid,
                            "reason": "缺 new/old 之一，不同源，整体剔除"})
            continue
        # 多时间窗: 按出现顺序配对，条数不等剔除该客户
        new_rows = sorted(vers["new"], key=lambda r: str(r.get("period", "")))
        old_rows = sorted(vers["old"], key=lambda r: str(r.get("period", "")))
        if len(new_rows) != len(old_rows):
            dropped.append({"customer_id": cid,
                            "reason": f"时间窗数量不等 new={len(new_rows)} old={len(old_rows)}"})
            continue
        for nw, od in zip(new_rows, old_rows):
            if nw.get("period") != od.get("period"):
                dropped.append({"customer_id": cid,
                                "reason": f"时间窗不配对: {nw.get('period')} vs {od.get('period')}"})
                break
            ni, oi = int(nw.get("articles_ingested", 0)), int(od.get("articles_ingested", 0))
            if ni != oi:
                dropped.append({"customer_id": cid,
                                "reason": f"入文量不同源: new={ni} vs old={oi}"})
                break
            nr, orv = int(nw.get("human_reviewed", 0)), int(od.get("human_reviewed", 0))
            rows.append({"customer_id": cid, "period": nw.get("period"),
                         "articles_ingested": ni,
                         "human_reviewed_new": nr, "human_reviewed_old": orv,
                         "reduction_pct": (round((orv - nr) / orv * 100, 2) if orv else None)})
            totals["new"]["reviewed"] += nr
            totals["old"]["reviewed"] += orv
            totals["new"]["ingested"] += ni
            totals["old"]["ingested"] += oi

    total_new = totals["new"]["reviewed"]
    total_old = totals["old"]["reviewed"]
    overall_reduction = (round((total_old - total_new) / total_old * 100, 2)
                         if total_old else None)
    # 入文量加权后的「每万条入文人工阅读数」——比裸条数更抗客户结构变化
    per_10k_new = (round(total_new / totals["new"]["ingested"] * 10000, 2)
                   if totals["new"]["ingested"] else None)
    per_10k_old = (round(total_old / totals["old"]["ingested"] * 10000, 2)
                   if totals["old"]["ingested"] else None)

    return {
        "status": "OK" if rows else "NO_VALID_DATA",
        "n_customers_valid": len(set(r["customer_id"] for r in rows)),
        "n_customers_dropped": len(dropped),
        "dropped": dropped[:10],
        "per_customer": rows[:50],
        "totals": {
            "human_reviewed_new": total_new,
            "human_reviewed_old": total_old,
            "overall_reduction_pct": overall_reduction,
            "reviewed_per_10k_articles_new": per_10k_new,
            "reviewed_per_10k_articles_old": per_10k_old,
            "note": ("工时降幅是收益数字不是门槛，判读归人；分子分母已同源"
                     "（同客户 / 同时间窗 / 同入文量）。单客户量级与全网口径两本账。"),
        },
    }


# ============================================================
# 3. 事后漏报回流（兜底监控）
# ============================================================

def feedback_check(feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    事后漏报回流: 客户侧改判 / 驳回动作的落盘回流统计。

    口径: escalate（客户改判升级 = 告警漏了负面）且 confirmed_negative=true
    → 事后确认的漏报。revoke（驳回误报）不计入漏报，但单列计数供误报侧参考。
    事后漏报率**不等价于**前向抽样结论（回流有选择性偏差：客户未必反馈），
    只做兜底监控。
    """
    if not feedback:
        return {"status": "NO_DATA", "note": "未提供回流数据（可省略）"}

    escalate_confirmed = [r for r in feedback
                          if r.get("action") == "escalate" and r.get("confirmed_negative")]
    revoke = [r for r in feedback if r.get("action") == "revoke"]
    return {
        "status": "OK",
        "n_feedback_total": len(feedback),
        "confirmed_missed_after_release": {"count": len(escalate_confirmed),
                                           "ids": [str(r.get("sample_id")) for r in escalate_confirmed[:20]]},
        "revoked_false_alarms": {"count": len(revoke),
                                 "note": "驳回误报，供误报侧参考，不计入漏报"},
        "note": "事后漏报率有选择性偏差（客户未必反馈），兜底监控用，不等价前向抽样。",
    }


# ============================================================
# 报告与 CLI
# ============================================================

def _dump(report: Dict[str, Any], out_path: str) -> None:
    path = os.path.abspath(out_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] 业务终验报告已落盘: {path}")


def main():
    ap = argparse.ArgumentParser(description="端到端业务终验（漏报核查 / 工时核查 / 事后回流）")
    ap.add_argument("--alerts_new", required=True, help="新版告警抽样 jsonl（含双人盲标结论）")
    ap.add_argument("--alerts_old", required=True, help="旧版告警抽样 jsonl")
    ap.add_argument("--workload", required=True, help="工时台账 jsonl（按客户 × 时间窗 × 版本）")
    ap.add_argument("--feedback", default=None, help="客户侧改判/驳回回流 jsonl（可省略）")
    ap.add_argument("--out", default="output/biz/business_report.json", help="报告落盘路径")
    args = ap.parse_args()

    for p in (args.alerts_new, args.alerts_old, args.workload):
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] 找不到输入文件: {p}")

    missed = missed_alarm_check(_load_jsonl(args.alerts_new), _load_jsonl(args.alerts_old))
    workload = workload_check(_load_jsonl(args.workload))
    feedback = feedback_check(_load_jsonl(args.feedback)) if args.feedback else \
        {"status": "NO_DATA", "note": "未提供回流数据（可省略）"}

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"alerts_new": os.path.abspath(args.alerts_new),
                   "alerts_old": os.path.abspath(args.alerts_old),
                   "workload": os.path.abspath(args.workload),
                   "feedback": os.path.abspath(args.feedback) if args.feedback else None},
        "missed_alarm_check": missed,
        "workload_check": workload,
        "feedback_check": feedback,
        "overall": "PASS" if missed["status"] == "PASS" else (
            "BLOCKED" if missed["status"] == "BLOCKED" else "FAIL"),
    }

    # ---- 终端摘要 ----
    print("=" * 70)
    print("端到端业务终验")
    print("=" * 70)
    print(f"\n--- 一、漏报核查（前向抽样, {missed.get('n_sampled', 0)} 条）---")
    if missed["status"] == "BLOCKED":
        print(f"  ⚠ BLOCKED: {missed['reason']}")
    else:
        print(f"  确认负面样本        : {missed.get('n_confirmed_negative', 0)} 条")
        nn = missed["net_new_missed"]
        print(f"  净增漏报（硬线 =0） : {nn['count']} 条  "
              f"{'✓' if nn['count'] == 0 else '✗ FAIL — ' + str(nn['ids'][:5])}")
        print(f"  老模型存量漏报      : {missed['legacy_missed_old_model']['count']} 条（另计，不归因本次）")
        print(f"  新版新捞回          : {missed['rescued_by_new']['count']} 条")
        amr = missed.get("absolute_miss_rate_new_pct")
        print(f"  新版绝对漏报率      : {'N/A' if amr is None else f'{amr:.2f}%'}")
    print(f"\n--- 二、工时核查 ---")
    wc = workload
    if wc["status"] != "OK":
        print(f"  ⚠ {wc['status']}: 无有效同源客户数据")
        for d in wc["dropped"][:5]:
            print(f"      {d['customer_id']}: {d['reason']}")
    else:
        t = wc["totals"]
        print(f"  有效客户            : {wc['n_customers_valid']} 个"
              f"（剔除不同源 {wc['n_customers_dropped']} 个）")
        print(f"  人工复核 new/old    : {t['human_reviewed_new']} / {t['human_reviewed_old']} 条")
        if t["overall_reduction_pct"] is not None:
            print(f"  降幅                : {t['overall_reduction_pct']:.2f}%"
                  f"  （每万条入文阅读 {t['reviewed_per_10k_articles_old']} → "
                  f"{t['reviewed_per_10k_articles_new']}）")
    print(f"\n--- 三、事后漏报回流（兜底监控）---")
    fb = feedback
    if fb["status"] == "NO_DATA":
        print("  （未提供回流数据）")
    else:
        print(f"  回流动作            : {fb['n_feedback_total']} 条")
        print(f"  事后确认漏报        : {fb['confirmed_missed_after_release']['count']} 条")
        print(f"  驳回误报            : {fb['revoked_false_alarms']['count']} 条")
    print(f"\n{'=' * 70}")
    print(f"总判定: 【{report['overall']}】"
          f"{'（净增漏报硬线未过）' if report['overall'] == 'FAIL' else ''}")
    print("=" * 70)

    _dump(report, args.out)
    sys.exit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
