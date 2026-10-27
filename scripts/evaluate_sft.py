#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情负面精判 · SFT 评测与验收主脚本 (《评测验收SOP_对齐版》+《评估手册》对齐)

两种运行模式 (语义保持不变, 只新增参数):
  [A] 单文件质检模式 (默认) —— 校验一个已冻结的数据集自身合规性
      python scripts/evaluate_sft.py --test_file data/gold500_test.jsonl

  [B] 双文件对照模式 —— 真实模型推理结果 vs 人工金标，出全量指标
      python scripts/evaluate_sft.py --gold_file gold.jsonl --pred_file pred.jsonl
      (pred_file 每行三选一: ① 裸模型输出字符串  ② {"output": "..."}  ③ {"prediction": {...}})

新增:
  --out PATH  落结构化报告 (默认 output/report.json)，供跨批次自动 diff
  --old_error_list PATH  门 1 用的上一版错题 id 列表 (可选)

五段式结构 (解析 → 指标 → 分桶 → 门禁 → 落盘):
  1. 解析段   —— 判定源统一到 lib/schema_contract.py（训练/评测同一份契约）
  2. 指标段   —— 纯函数在 lib/eval_metrics.py，非法标签单列并从集合指标剔除
  3. 分桶段   —— 四维热力 (hard_case / industry / relevance / sentiment)
  4. 门禁段   —— 四道发布门在 lib/release_gate.py，任一 FAIL 即中止
  5. 落盘段   —— report.json (schema_version=1)，终端同步打印人可读摘要

核心口径 (《评测验收SOP》卡点 + 评估手册缺口 11~18):
  1. 协议层: 思考标签闭合率 100% / JSON 合法率 ≥99.8% / evidence 严格子串率 100%
             (分母一律**总样本数**) / 闭集契约零违规
  2. 分类层: hard_case 7类 Macro-F1 + irony 单项召回 ≥90%
  3. 集合层: risk_categories Subset Accuracy ≥90% (漏标多标对称双罚)
  4. 业务层: sentiment 负向召回与误报率**成对**输出 (禁止单报其一)
  5. 归因层: 四维分桶 + 错判样例 Top5

纯标准库 (无 torch / transformers / requests)，可离线单测。
"""

import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from eval_metrics import (  # noqa: E402
    confusion_slices,
    illegal_label_counts,
    macro_f1,
    pa_fp_pair,
    subset_accuracy,
)
from release_gate import evaluate_gates, overall_verdict  # noqa: E402
from schema_contract import (  # noqa: E402
    VALID_HARD_CASE,
    VALID_RELEVANCE,
    VALID_RISK,
    VALID_SENTIMENT,
    extract_baseline_pred,
    load_pred_line as _load_pred_line,
    parse_lenient,
    parse_output,
    resolve_gold as _resolve_gold,
    unwrap_output_text,
)

# ============================================================
# 验收门槛 (《评测验收SOP》定稿 + 评估手册 5.8)
# ============================================================

THRESH_JSON_VALID = 99.8        # %
THRESH_EVIDENCE_SUBSTR = 100.0  # % 一票否决
THRESH_SUBSET_ACC = 90.0        # %
THRESH_IRONY_RECALL = 90.0      # %
THRESH_MACRO_F1 = 88.0          # %
# 负向召回门槛**不填设计值**（手册 5.8: 门槛一律取在役实测基线），
# 未提供基线时只报数、不进 gate2 判定（release_gate 会记入 skipped）。

# 报告 schema 版本
REPORT_SCHEMA_VERSION = 1


# ============================================================
# 1. 解析段
# ============================================================

def parse_model_output(output_str: Any) -> Dict[str, Any]:
    """
    解析模型 output（兼容旧接口：返回 dict 而不是元组）。

    判定源统一到 `schema_contract.parse_output` —— 与训练侧 `R_format` 门禁
    **同一份判定式**（评估手册缺口 11）。旧的宽松降级（裸 JSON 也算解析成功）
    保留在 `parse_lenient` 里，但降级**不改变协议判定**：闭合率照常掉。

    基线降级路径：目标格式全失败时（未微调模型原生输出对不齐 <think> +
    裸 JSON 协议），再走 `extract_baseline_pred` 正则捕获 JSON 片段、
    只留有效字段 —— 基线评测不做强解析，否则解析失败会把指标全量清零，
    对基线不公平。降级不改变协议判定（valid_json 仍按强解析口径记 False）。
    """
    cot, pred, flags = parse_output(output_str)
    res = {
        "valid_thought_tags": flags["valid_thought_tags"],
        "valid_json": flags["valid_json"],
        "trailing_garbage": flags["trailing_garbage"],
        "schema_error": flags["schema_error"],
        "cot": cot or "",
        "pred": pred,
        "error": flags["schema_error"],
    }
    if not res["valid_json"] and not flags["valid_thought_tags"]:
        # 诊断用宽松降级：裸 JSON 也能读出结构，便于定位「格式漂移」类缺陷
        _, lenient_pred, lenient_flags = parse_lenient(output_str)
        if lenient_pred is not None:
            res["pred"] = lenient_pred
            res["valid_json"] = True
            res["schema_error"] = lenient_flags["schema_error"]
    if res["pred"] is None:
        # 基线宽松提取：正则捕获 JSON 片段，只取协议内有效字段做对比
        # （字段对比以原始提取结果为准；协议判定不放宽，valid_json 不翻真）
        baseline_pred = extract_baseline_pred(output_str)
        if baseline_pred:
            res["pred"] = baseline_pred
            res["baseline_extracted"] = True
    return res


def load_pred_line(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    把 pred_file 的一行规范化。返回 (pred_dict, form)。

    form 取值:
      "text"      —— 该行提供了模型输出文本（可判闭合率）
      "dict"      —— 该行直接给了结构化预测（无 CoT，闭合率不适用）
      "invalid"   —— 该行既不是文本也不是合法 dict

    实现委托 `schema_contract.load_pred_line`，评测侧不再各写一份解析。
    """
    return _load_pred_line(raw)


def resolve_gold(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    金标字段来源统一（评估手册缺口 15）。

    铁律: **金标一律以 output JSON 为准**；`meta` 只用于
      ① 分桶维度（industry）；
      ② output JSON 缺字段时的兜底；
    两者冲突 → 记录冲突清单（不静默），由上层打印前 5 条。

    实现委托 `schema_contract.resolve_gold`。
    """
    return _resolve_gold(row)


# ============================================================
# 2. 指标段（薄封装：真正的计算在 lib/eval_metrics.py）
# ============================================================

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """兼容旧接口：基础 P/R/F1（0~1 口径）。"""
    from eval_metrics import prf as _prf
    return _prf(tp, fp, fn)


def macro_f1_single_label(golds: List[str], preds: List[str],
                          labels: List[str]) -> Dict[str, Any]:
    """兼容旧接口：单值字段逐类 P/R/F1 + 宏均 F1（% 口径）。"""
    return macro_f1(golds, preds, labels)


def subset_accuracy_pair(golds: List[List[str]],
                          preds: List[List[str]]) -> Dict[str, Any]:
    """兼容旧接口：risk_categories 的 Subset Accuracy。"""
    return subset_accuracy(golds, preds)


# ============================================================
# 3. 分桶段
# ============================================================

SLICE_DIMS = ("hard_case_type", "industry", "relevance", "sentiment")


def build_slice_dims(rows: List[Dict[str, Any]],
                      golds: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    构造四维分桶的桶标签序列。

    维度取值来源:
      hard_case_type / sentiment / relevance —— 金标（output JSON 为准）
      industry                              —— meta（分桶维度专属来源）
    """
    dims: Dict[str, List[Any]] = {d: [] for d in SLICE_DIMS}
    for row, gold in zip(rows, golds):
        meta = row.get("meta", {}) or {}
        dims["hard_case_type"].append(gold.get("hard_case_type"))
        dims["sentiment"].append(gold.get("sentiment"))
        dims["relevance"].append(gold.get("relevance"))
        dims["industry"].append(meta.get("industry") or "unknown")
    return dims


# ============================================================
# 报告组装
# ============================================================

def _pct(x: float) -> float:
    return round(float(x), 2)


def build_protocol_block(n_total: int, tag_ok: int, text_lines: int,
                         json_ok: int, ev_ok: int, ev_present: int,
                         violations: int, trailing: int) -> Dict[str, Any]:
    """
    协议层指标。

    分母口径统一（缺口 13）: `evidence_substr` 一律用**总样本数**做分母
    （缺 evidence 即不通过），另报 `evidence_field_coverage` 报字段覆盖率。
    思考标签闭合率的分母是**提供了模型输出文本的行数**（`text_lines`）——
    直接给结构化 dict 的行没有 CoT 可判，记 N/A 而不是假装闭合。
    """
    n = max(n_total, 1)
    return {
        "thought_tag_closure": _pct(tag_ok / text_lines * 100) if text_lines else None,
        "thought_tag_closure_applicable": text_lines,
        "json_valid": _pct(json_ok / n * 100),
        "evidence_substr": _pct(ev_ok / n * 100),
        "evidence_field_coverage": _pct(ev_present / n * 100),
        "contract_violations": violations,
        "trailing_garbage": trailing,
        "n_samples": n_total,
    }


def build_empty_report(mode: str, gold_path: str, pred_path: str) -> Dict[str, Any]:
    """空报告骨架（字段名与《评估手册》§1 固定结构一致）。"""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "inputs": {"gold": gold_path, "pred": pred_path},
        "n_samples": 0,
        "protocol": {},
        "illegal_labels": {"sentiment": {}, "hard_case_type": {}, "risk_categories": {}},
        "sentiment": {"macro_f1": 0.0, "per_class": {}, "negative_recall": 0.0,
                      "false_alarm_rate": 0.0},
        "hard_case": {"macro_f1": 0.0, "per_class": {}, "irony_recall": 0.0},
        "risk": {"subset_accuracy": 0.0, "empty_support": 0, "empty_precision": 0.0,
                 "per_key": {}},
        "slices": {d: {} for d in SLICE_DIMS},
        "errors_top5": [],
        "gates": {"gate1": "N/A", "gate2": "N/A", "gate3": "N/A", "gate4": "N/A"},
        "verdict": "FAIL",
        "fail_reasons": [],
    }


# ============================================================
# 模式 A · 单文件质检
# ============================================================

def run_single_file_check(file_path: str, out_path: Optional[str] = None) -> bool:
    print("=" * 70)
    print(f"模式 A · 数据集自检: {file_path}")
    print("=" * 70)

    with open(file_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # ---- 分账：Anchor(公开通用指令) 不走舆情四查 ----
    # Anchor 是防遗忘的通用指令数据 (provenance=public_anchor / hard_case_type=general_anchor)，
    # 其「output」是自然语言答案，不含 CoT+JSON 协议——按舆情协议查必然
    # JSON 合法率 0%、evidence 子串率 0%，四查 FAIL 是口径错配而非数据问题。
    # 处置：锚点样本分账跳过舆情协议四查，只做行级 JSONL 结构合法性检查，
    # 统计入 report["anchor_partition"]；混入 val 的锚点另行报错（两账分离铁律）。
    def _is_anchor(r):
        m = r.get("meta") or {}
        return (m.get("provenance") == "public_anchor"
                or m.get("hard_case_type") == "general_anchor")

    anchor_rows = [r for r in rows if _is_anchor(r)]
    domain_rows = [r for r in rows if not _is_anchor(r)]
    anchor_partition = {
        "n_anchor": len(anchor_rows),
        "n_domain": len(domain_rows),
        "skipped_reason": "公开通用 Anchor 无舆情输出协议，不适用 CoT/JSON 四查",
        "jsonl_structural_ok": None,
        "note": "Anchor 只查 JSONL 行级结构合法；协议四查仅对领域样本执行",
    }
    # Anchor 行级结构检查：字段齐全 + 非空 output
    struct_bad = [i for i, r in enumerate(anchor_rows, 1)
                  if not r.get("instruction") or not r.get("output")]
    anchor_partition["jsonl_structural_ok"] = (len(struct_bad) == 0)
    if struct_bad:
        anchor_partition["bad_rows"] = struct_bad[:10]

    rows = domain_rows  # 舆情四查只跑领域分账

    report = build_empty_report("single_file", file_path, "")
    total = len(rows)

    golds: List[Dict[str, Any]] = []
    preds: List[Optional[Dict[str, Any]]] = []
    dims = {d: [] for d in SLICE_DIMS}
    all_conflicts: List[Dict[str, Any]] = []

    tag_ok = text_lines = json_ok = ev_ok = ev_present = 0
    violations = trailing = 0
    agree_hc = agree_sent = agree_risk = 0
    agreement_n = 0
    risk_agreement_n = 0

    for i, row in enumerate(rows, 1):
        meta = row.get("meta", {}) or {}
        text = row.get("output", "")
        parsed = parse_model_output(text)
        gold, conflicts = resolve_gold(row)
        golds.append(gold)
        preds.append(parsed["pred"])
        if conflicts:
            all_conflicts.append({"idx": i, "conflicts": conflicts})

        # 该行是否提供了模型输出文本（裸串 / {"output": str}）
        is_text = isinstance(text, str) and bool(text.strip())
        if is_text:
            text_lines += 1
            if parsed["valid_thought_tags"]:
                tag_ok += 1
        if parsed["valid_json"]:
            json_ok += 1
        if parsed["trailing_garbage"]:
            trailing += 1
        if parsed["schema_error"]:
            violations += 1

        ev = (parsed["pred"] or {}).get("evidence") if parsed["pred"] else None
        if isinstance(ev, str) and ev:
            ev_present += 1
            if ev in (row.get("input", "") or ""):
                ev_ok += 1

        # meta 与 output 的一致率（数据集自洽性诊断）
        if parsed["pred"]:
            agreement_n += 1
            if meta.get("hard_case_type") and \
                    meta["hard_case_type"] == parsed["pred"].get("hard_case_type"):
                agree_hc += 1
            if meta.get("sentiment") and \
                    meta["sentiment"] == parsed["pred"].get("sentiment"):
                agree_sent += 1
            if isinstance(meta.get("risk_categories"), list) and \
                    sorted(map(str, meta["risk_categories"])) == \
                    sorted(map(str, parsed["pred"].get("risk_categories") or [])):
                agree_risk += 1
                risk_agreement_n += 1

        dims["hard_case_type"].append(gold.get("hard_case_type"))
        dims["sentiment"].append(gold.get("sentiment"))
        dims["relevance"].append(gold.get("relevance"))
        dims["industry"].append(meta.get("industry") or "unknown")

    # ---- 协议层 ----
    report["n_samples"] = total
    report["protocol"] = build_protocol_block(total, tag_ok, text_lines, json_ok,
                                              ev_ok, ev_present, violations, trailing)

    # ---- 指标层（金标与预测同源 → 语义指标恒为满分，实意义在 meta 一致率）----
    g_sent = [g.get("sentiment") for g in golds]
    p_sent = [(p or {}).get("sentiment") for p in preds]
    g_hc = [g.get("hard_case_type") for g in golds]
    p_hc = [(p or {}).get("hard_case_type") for p in preds]
    g_rk = [g.get("risk_categories") or [] for g in golds]
    p_rk = [(p or {}).get("risk_categories") or [] for p in preds]

    sent_stat = macro_f1(g_sent, p_sent, sorted(VALID_SENTIMENT), VALID_SENTIMENT)
    hc_stat = macro_f1(g_hc, p_hc, sorted(VALID_HARD_CASE), VALID_HARD_CASE)
    rk_stat = subset_accuracy(g_rk, p_rk, VALID_RISK)
    pa = pa_fp_pair(g_sent, p_sent)

    report["sentiment"] = {
        "macro_f1": _pct(sent_stat["macro_f1"]),
        "per_class": sent_stat["per_class"],
        "negative_recall": _pct(pa["negative_recall"]),
        "false_alarm_rate": _pct(pa["false_alarm_rate"]),
        "negative_precision": _pct(pa["negative_precision"]),
    }
    report["hard_case"] = {
        "macro_f1": _pct(hc_stat["macro_f1"]),
        "per_class": hc_stat["per_class"],
        "irony_recall": _pct(hc_stat["per_class"]["irony"]["recall"]),
    }
    report["risk"] = {
        "subset_accuracy": _pct(rk_stat["subset_accuracy"]),
        "empty_support": rk_stat["empty_support"],
        "empty_precision": _pct(rk_stat["empty_precision"]),
        "per_key": rk_stat["per_key"],
    }

    # ---- 非法标签 ----
    report["illegal_labels"] = {
        "sentiment": illegal_label_counts(preds, "sentiment", VALID_SENTIMENT),
        "hard_case_type": illegal_label_counts(preds, "hard_case_type", VALID_HARD_CASE),
        "risk_categories": illegal_label_counts(preds, "risk_categories", VALID_RISK),
    }

    # ---- 分桶 ----
    report["slices"] = confusion_slices(dims, [p or {} for p in preds], SLICE_DIMS,
                                        gold={"sentiment": g_sent,
                                              "hard_case_type": g_hc,
                                              "risk_categories": g_rk})

    # ---- meta 一致率（自洽性诊断）----
    agreement = {
        "n": agreement_n,
        "hard_case_type_agree": _pct(agree_hc / agreement_n * 100) if agreement_n else None,
        "sentiment_agree": _pct(agree_sent / agreement_n * 100) if agreement_n else None,
        "risk_categories_agree": (_pct(agree_risk / risk_agreement_n * 100)
                                  if risk_agreement_n else None),
        "risk_agreement_n": risk_agreement_n,
    }
    report["meta_agreement"] = agreement
    report["gold_source_conflicts"] = {"count": len(all_conflicts),
                                       "examples": all_conflicts[:5]}
    # 模式 A 的语义指标是**自洽性指标**（金标与预测同源 → 恒 100%），
    # 真正的信号在 protocol 与 meta_agreement。不写清楚会被误读成「模型成绩」。
    report["note"] = ("模式 A 为数据集自检：金标与预测同源（output JSON），"
                      "sentiment/hard_case/risk 恒为 100% 属构造性结果，"
                      "不构成模型成绩；有效信号是 protocol 与 meta_agreement。")

    # ---- 门禁 ----
    gates = evaluate_gates(golds, [dict(p or {}, valid_json=bool(p)) for p in preds])
    report["gates"] = {k: gates[k] for k in ("gate1", "gate2", "gate3", "gate4")}
    report["gates"]["gate2"] = "N/A"  # 单文件自检无「上一版基线」，不假装判过
    report["gates"]["gate3"] = "N/A"

    fail_reasons: List[str] = []
    proto = report["protocol"]
    if proto["thought_tag_closure_applicable"] and proto["thought_tag_closure"] != 100.0:
        fail_reasons.append(f"思考标签闭合率 {proto['thought_tag_closure']}% < 100%")
    if proto["json_valid"] < THRESH_JSON_VALID:
        fail_reasons.append(f"JSON 合法率 {proto['json_valid']}% < {THRESH_JSON_VALID}%")
    if proto["evidence_substr"] < THRESH_EVIDENCE_SUBSTR:
        fail_reasons.append(f"evidence 严格子串率 {proto['evidence_substr']}% "
                            f"< {THRESH_EVIDENCE_SUBSTR}%")
    if proto["contract_violations"] > 0:
        fail_reasons.append(f"闭集契约违规 {proto['contract_violations']} 条")
    if report["gates"]["gate4"] == "FAIL":
        fail_reasons.append("门 4 格式合规未过")
    report["fail_reasons"] = fail_reasons
    report["verdict"] = "PASS" if not fail_reasons else "FAIL"

    # ---- Anchor 分账结果并入报告与 fail_reasons ----
    report["anchor_partition"] = anchor_partition
    if not anchor_partition["jsonl_structural_ok"]:
        fail_reasons.append(f"Anchor 分账行级结构非法 {len(struct_bad)} 条")
        report["verdict"] = "FAIL"

    # ---- 终端输出 ----
    print(f"\n--- 〇、Anchor 分账 (不走舆情四查) ---")
    print(f"  Anchor 样本         : {anchor_partition['n_anchor']} 条 (跳过协议四查)")
    print(f"  领域样本            : {anchor_partition['n_domain']} 条 (执行四查)")
    print(f"  Anchor 行级结构     : {'OK' if anchor_partition['jsonl_structural_ok'] else 'FAIL'}")

    print(f"\n--- 一、协议硬约束 ---")
    print(f"  样本总数            : {total}")
    if proto["thought_tag_closure_applicable"]:
        print(f"  思考标签闭合率      : {proto['thought_tag_closure']:.2f}%  (要求 100%)")
    else:
        print("  思考标签闭合率      : N/A (无模型输出文本行)")
    print(f"  JSON 格式合法率     : {proto['json_valid']:.2f}%  (门槛 ≥{THRESH_JSON_VALID}%)")
    print(f"  evidence 严格子串率 : {proto['evidence_substr']:.2f}%  "
          f"(分母=总样本数 {total}, 一票否决)")
    print(f"  evidence 字段覆盖率 : {proto['evidence_field_coverage']:.2f}%")
    print(f"  闭集契约违规        : {proto['contract_violations']} 条")

    print(f"\n--- 二、标注自洽（meta 与 output JSON 一致率） ---")
    print(f"  hard_case 一致率    : {agreement['hard_case_type_agree']}%")
    print(f"  sentiment 一致率    : {agreement['sentiment_agree']}%")
    print(f"  risk 一致率         : {agreement['risk_categories_agree']}%  "
          f"(n={agreement['risk_agreement_n']}, meta 未带 risk 标签的样本不计入)")
    if all_conflicts:
        print(f"\n  [WARN] 金标来源冲突 {len(all_conflicts)} 条（前 5 条）:")
        for c in all_conflicts[:5]:
            print(f"    #{c['idx']}: {'; '.join(c['conflicts'])}")

    print(f"\n--- 三、四维切片 ---")
    for dim in SLICE_DIMS:
        buckets = report["slices"].get(dim, {})
        if not buckets:
            continue
        print(f"  [{dim}]")
        for key, st in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
            if st.get("insufficient"):
                # n<5 只报样本数不报率（读数抖 ±6~7pp 是抽样噪声不是能力）
                print(f"    {key:<22}n={st['n']:<4}  (样本不足, 不报率)")
            else:
                print(f"    {key:<22}n={st['n']:<4}风险完全匹配率={st.get('risk_exact', 0):>6.1f}%")

    print(f"\n{'=' * 70}")
    print(f"自检结论: {'【PASS】' if report['verdict'] == 'PASS' else '【FAIL】'}")
    for r in fail_reasons:
        print(f"  - {r}")
    print(f"{'=' * 70}")

    if out_path:
        _dump_report(report, out_path)
    return report["verdict"] == "PASS"


# ============================================================
# 模式 B · 双文件对照
# ============================================================

def run_dual_file_eval(gold_file: str, pred_file: str,
                       out_path: Optional[str] = None,
                       old_error_list: Optional[str] = None) -> bool:
    print("=" * 70)
    print("模式 B · 模型对照评测")
    print(f"  gold: {gold_file}")
    print(f"  pred: {pred_file}")
    print("=" * 70)

    with open(gold_file, encoding="utf-8") as f:
        golds_rows = [json.loads(l) for l in f if l.strip()]
    with open(pred_file, encoding="utf-8") as f:
        pred_lines = [l for l in f if l.strip()]

    report = build_empty_report("dual_file", gold_file, pred_file)

    if len(golds_rows) != len(pred_lines):
        print(f"[FATAL] 样本数不一致: gold={len(golds_rows)} pred={len(pred_lines)}")
        report["verdict"] = "FAIL"
        report["fail_reasons"] = [f"样本数不一致: gold={len(golds_rows)} pred={len(pred_lines)}"]
        if out_path:
            _dump_report(report, out_path)
        return False

    total = len(golds_rows)
    report["n_samples"] = total

    golds: List[Dict[str, Any]] = []
    preds: List[Optional[Dict[str, Any]]] = []
    pred_flags: List[Dict[str, Any]] = []
    dims = {d: [] for d in SLICE_DIMS}
    all_conflicts: List[Dict[str, Any]] = []

    tag_ok = text_lines = json_ok = ev_ok = ev_present = 0
    violations = trailing = 0
    errors: List[Dict[str, Any]] = []
    form_counts: collections.Counter = collections.Counter()

    for i, (grow, pline) in enumerate(zip(golds_rows, pred_lines), 1):
        meta = grow.get("meta", {}) or {}
        gold, conflicts = resolve_gold(grow)
        if conflicts:
            all_conflicts.append({"idx": i, "conflicts": conflicts})
        # 注入样本 id（与 scripts/release_gate.py 的 _load_golds 同一规则），
        # 否则门 1 的 _gold_id 退化为 0 基行号，真实 id 的旧错题全部对不上。
        gid = None
        for k in ("id", "event_id", "sample_id"):
            if grow.get(k):
                gid = str(grow[k])
                break
            if isinstance(meta, dict) and meta.get(k):
                gid = str(meta[k])
                break
        gold["id"] = gid if gid else f"#{i}"
        golds.append(gold)

        pred, form = load_pred_line(pline)
        form_counts[form] += 1
        # 基线宽松提取：强解析失败时正则捕获输出里的 JSON 片段，只留协议内
        # 有效字段参与字段对比。基线未微调，强解析会全量失败把指标清零，
        # 对基线不公平；但 valid_json 协议口径不放宽（仍按强解析记 False），
        # 格式门槛照实反映「基线格式不合规」这一事实。
        baseline_extracted = False
        if pred is None:
            bp_text = unwrap_output_text(pline) if form == "text" else pline
            bp = extract_baseline_pred(bp_text)
            if bp:
                pred = bp
                form_counts["baseline"] += 1
                baseline_extracted = True
        preds.append(pred)

        # 协议标志：文本行才有闭合率口径
        is_text = form == "text"
        raw_text = ""
        if is_text:
            # 统一走 schema_contract.unwrap_output_text（剥壳逻辑唯一实现）
            raw_text = unwrap_output_text(pline)
        parsed = parse_model_output(raw_text) if is_text else \
            {"valid_thought_tags": False, "valid_json": pred is not None,
             "trailing_garbage": False, "schema_error": None,
             "cot": "", "pred": pred}
        if is_text:
            text_lines += 1
            if parsed["valid_thought_tags"]:
                tag_ok += 1
        if pred is not None:
            json_ok += 1
        if parsed["trailing_garbage"]:
            trailing += 1
        # 契约违规：闭集越界 / 缺字段 / 嵌套对象（与训练侧同一判定式）
        schema_error = parsed["schema_error"]
        if schema_error is None and pred is not None and form == "dict":
            from schema_contract import validate_schema
            schema_error = validate_schema(pred)
        if schema_error:
            violations += 1

        pred_flags.append({
            "valid_json": pred is not None,
            "valid_tags": bool(parsed["valid_thought_tags"]),
            "schema_error": schema_error,
        })

        ev = pred.get("evidence") if pred else None
        if isinstance(ev, str) and ev:
            ev_present += 1
            if ev in (grow.get("input", "") or ""):
                ev_ok += 1

        # 错判样例（归因用）
        if pred is not None:
            risk_exact = sorted(map(str, gold.get("risk_categories") or [])) == \
                sorted(map(str, pred.get("risk_categories") or []))
            sent_ok = gold.get("sentiment") == pred.get("sentiment")
            if not (risk_exact and sent_ok):
                errors.append({
                    "idx": i,
                    "industry": meta.get("industry") or "unknown",
                    "gold_sentiment": gold.get("sentiment"),
                    "pred_sentiment": pred.get("sentiment"),
                    "gold_risk": sorted(map(str, gold.get("risk_categories") or [])),
                    "pred_risk": sorted(map(str, pred.get("risk_categories") or [])),
                    "gold_hc": gold.get("hard_case_type"),
                    "pred_hc": pred.get("hard_case_type"),
                    "input_head": (grow.get("input", "") or "")[:50],
                })

        dims["hard_case_type"].append(gold.get("hard_case_type"))
        dims["sentiment"].append(gold.get("sentiment"))
        dims["relevance"].append(gold.get("relevance"))
        dims["industry"].append(meta.get("industry") or "unknown")

    # ---- 协议层 ----
    report["protocol"] = build_protocol_block(total, tag_ok, text_lines, json_ok,
                                              ev_ok, ev_present, violations, trailing)
    report["protocol"]["n_pred_forms"] = dict(form_counts)

    # ---- 指标层 ----
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

    report["sentiment"] = {
        "macro_f1": _pct(sent_stat["macro_f1"]),
        "per_class": sent_stat["per_class"],
        "negative_recall": _pct(pa["negative_recall"]),
        "false_alarm_rate": _pct(pa["false_alarm_rate"]),
        "negative_precision": _pct(pa["negative_precision"]),
        "n_scored": sent_stat["n_scored"],
        "n_skipped_illegal": sent_stat["n_skipped_illegal"],
    }
    report["hard_case"] = {
        "macro_f1": _pct(hc_stat["macro_f1"]),
        "per_class": hc_stat["per_class"],
        "irony_recall": _pct(hc_stat["per_class"]["irony"]["recall"]),
        "n_scored": hc_stat["n_scored"],
        "n_skipped_illegal": hc_stat["n_skipped_illegal"],
    }
    report["risk"] = {
        "subset_accuracy": _pct(rk_stat["subset_accuracy"]),
        "empty_support": rk_stat["empty_support"],
        "empty_precision": _pct(rk_stat["empty_precision"]),
        "per_key": rk_stat["per_key"],
        "n_scored": rk_stat["n_scored"],
        "n_skipped_illegal": rk_stat["n_skipped_illegal"],
    }
    report["relevance"] = {"macro_f1": _pct(rel_stat["macro_f1"]),
                           "per_class": rel_stat["per_class"]}

    # ---- 非法标签（闭集外单列，集合类指标里已剔除）----
    report["illegal_labels"] = {
        "sentiment": illegal_label_counts(preds, "sentiment", VALID_SENTIMENT),
        "hard_case_type": illegal_label_counts(preds, "hard_case_type", VALID_HARD_CASE),
        "risk_categories": illegal_label_counts(preds, "risk_categories", VALID_RISK),
    }

    # ---- 分桶 ----
    report["slices"] = confusion_slices(dims, [p or {} for p in preds], SLICE_DIMS,
                                        gold={"sentiment": g_sent,
                                              "hard_case_type": g_hc,
                                              "risk_categories": g_rk})

    # ---- 错判样例 ----
    report["errors_top5"] = errors[:5]
    report["gold_source_conflicts"] = {"count": len(all_conflicts),
                                       "examples": all_conflicts[:5]}

    # ---- 门禁 ----
    old_error_ids = None
    if old_error_list:
        with open(old_error_list, encoding="utf-8") as f:
            old_error_ids = [l.strip() for l in f if l.strip()]
    # 门 2/门 4 需要**预测内容 + 解析标志**两样信息，
    # 合成一份带标志的预测列表（不污染原 preds）。
    pred_for_gate: List[Dict[str, Any]] = []
    for p, fl in zip(preds, pred_flags):
        merged = dict(p) if p else {}
        merged["valid_json"] = fl["valid_json"]
        merged["valid_tags"] = fl["valid_tags"]
        pred_for_gate.append(merged)
    gates = evaluate_gates(golds, pred_for_gate, old_error_ids=old_error_ids,
                            thresholds={"irony_recall": THRESH_IRONY_RECALL,
                                        "macro_f1": THRESH_MACRO_F1,
                                        "subset_accuracy": THRESH_SUBSET_ACC,
                                        "json_valid": THRESH_JSON_VALID})
    report["gates"] = {k: gates[k] for k in ("gate1", "gate2", "gate3", "gate4")}
    report["gate_details"] = gates["details"]
    report["attribution"] = gates["attribution"][:50]

    # ---- verdict（协议层 + 门禁层一起判）----
    proto = report["protocol"]
    fail_reasons: List[str] = []
    if proto["thought_tag_closure_applicable"] and proto["thought_tag_closure"] != 100.0:
        fail_reasons.append(f"思考标签闭合率 {proto['thought_tag_closure']}% < 100%")
    if proto["json_valid"] < THRESH_JSON_VALID:
        fail_reasons.append(f"JSON 合法率 {proto['json_valid']}% < {THRESH_JSON_VALID}%")
    if proto["evidence_substr"] < THRESH_EVIDENCE_SUBSTR:
        fail_reasons.append(f"evidence 严格子串率 {proto['evidence_substr']}% "
                            f"< {THRESH_EVIDENCE_SUBSTR}%")
    if proto["contract_violations"] > 0:
        fail_reasons.append(f"闭集契约违规 {proto['contract_violations']} 条")
    if report["gates"]["gate2"] == "FAIL":
        fail_reasons.append("门 2 金标不退化未过")
    if report["gates"]["gate4"] == "FAIL":
        fail_reasons.append("门 4 格式合规未过")
    if report["gates"]["gate1"] == "FAIL":
        fail_reasons.append("门 1 历史错题修复未过")
    if report["gates"]["gate3"] == "FAIL":
        fail_reasons.append("门 3 全链路安全底线未过")
    report["fail_reasons"] = fail_reasons
    report["verdict"] = "PASS" if not fail_reasons else "FAIL"

    # ---- 终端输出 ----
    print(f"\n--- 一、协议与格式门槛 ---")
    if proto["thought_tag_closure_applicable"]:
        print(f"  思考标签闭合率      : {proto['thought_tag_closure']:.2f}%  (要求 100%)")
    else:
        print("  思考标签闭合率      : N/A (pred 全为结构化 dict，无 CoT 可判)")
    print(f"  JSON 合法率         : {proto['json_valid']:.2f}%  "
          f"(门槛 ≥{THRESH_JSON_VALID}%)")
    print(f"  evidence 严格子串率 : {proto['evidence_substr']:.2f}%  "
          f"(分母=总样本数 {total}, 一票否决)")
    print(f"  evidence 字段覆盖率 : {proto['evidence_field_coverage']:.2f}%")
    print(f"  闭集契约违规        : {proto['contract_violations']} 条")
    print(f"  尾赘文本条数        : {proto['trailing_garbage']}")

    print(f"\n--- 二、hard_case_type (7 类) ---")
    print(f"  Macro-F1: {report['hard_case']['macro_f1']:.2f}%")
    print(f"  {'类别':<18}{'support':>8}{'P':>9}{'R':>9}{'F1':>9}")
    for lab in sorted(VALID_HARD_CASE):
        s = hc_stat["per_class"][lab]
        print(f"  {lab:<18}{s['support']:>8}{s['precision']:>8.1f}%{s['recall']:>8.1f}%"
              f"{s['f1']:>8.1f}%")
    irony_r = report["hard_case"]["irony_recall"]
    print(f"\n  ★ irony 单项召回: {irony_r:.2f}%  "
          f"({'PASS' if irony_r >= THRESH_IRONY_RECALL else 'FAIL'}, 门槛 ≥{THRESH_IRONY_RECALL}%)")

    print(f"\n--- 三、sentiment (3 类) ---")
    print(f"  Macro-F1: {report['sentiment']['macro_f1']:.2f}%")
    print(f"  {'类别':<12}{'support':>8}{'P':>9}{'R':>9}{'F1':>9}")
    for lab in sorted(VALID_SENTIMENT):
        s = sent_stat["per_class"][lab]
        print(f"  {lab:<12}{s['support']:>8}{s['precision']:>8.1f}%{s['recall']:>8.1f}%"
              f"{s['f1']:>8.1f}%")
    print(f"\n  ★ 负向召回 (成对口径): {report['sentiment']['negative_recall']:.2f}%  |  "
          f"误报率: {report['sentiment']['false_alarm_rate']:.2f}%  "
          f"(负向精确率 {report['sentiment']['negative_precision']:.2f}%)")

    print(f"\n--- 四、risk_categories (对称双罚) ---")
    sa = report["risk"]["subset_accuracy"]
    print(f"  ★ Subset Accuracy: {sa:.2f}%  "
          f"({'PASS' if sa >= THRESH_SUBSET_ACC else 'FAIL'}, 门槛 ≥{THRESH_SUBSET_ACC}%)")
    print(f"  空表支撑集: {report['risk']['empty_support']} 条 | "
          f"空表精确率: {report['risk']['empty_precision']:.2f}%")
    print(f"\n  per-key 诊断线 (不进准入门槛):")
    print(f"  {'Key':<20}{'support':>8}{'P':>9}{'R':>9}{'F1':>9}")
    for k in sorted(VALID_RISK):
        s = rk_stat["per_key"][k]
        print(f"  {k:<20}{s['support']:>8}{s['precision']:>8.1f}%{s['recall']:>8.1f}%"
              f"{s['f1']:>8.1f}%")

    illegal_total = sum(len(v) for v in report["illegal_labels"].values())
    if illegal_total:
        print(f"\n  ⚠ 非法标签（闭集外，已从集合类指标剔除）:")
        for field, cnt in report["illegal_labels"].items():
            for val, c in cnt.items():
                print(f"     {field} = {val!r}: {c} 条")

    print(f"\n--- 五、四维切片热力 ---")
    for dim in SLICE_DIMS:
        buckets = report["slices"].get(dim, {})
        if not buckets:
            continue
        print(f"  [{dim}] {'风险完全匹配率'}")
        for key, st in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
            if st.get("insufficient"):
                print(f"    {key:<22}n={st['n']:<4}  (样本不足, 不报率)")
            else:
                print(f"    {key:<22}n={st['n']:<4}{st.get('risk_exact', 0):>7.1f}%")

    if all_conflicts:
        print(f"\n  [WARN] 金标来源冲突 {len(all_conflicts)} 条（前 5 条）:")
        for c in all_conflicts[:5]:
            print(f"    #{c['idx']}: {'; '.join(c['conflicts'])}")

    if report["errors_top5"]:
        print(f"\n--- 六、错判样例 (前 5 条，供 Shadow Pool 归因) ---")
        for ex in report["errors_top5"]:
            print(f"  #{ex['idx']} [{ex['industry']}] gold={ex['gold_risk']} "
                  f"pred={ex['pred_risk']} sent:{ex['gold_sentiment']}→{ex['pred_sentiment']} "
                  f"hc:{ex['gold_hc']}→{ex['pred_hc']}")
            print(f"      原文: {ex['input_head']}…")

    print(f"\n--- 七、发布门 ---")
    for g in ("gate1", "gate2", "gate3", "gate4"):
        print(f"  {g}: {report['gates'][g]}")
    if report["attribution"]:
        print(f"  归因（前 5 条）:")
        for a in report["attribution"][:5]:
            print(f"    [{a['gate']}] {a['sample_id']}: {a['reason']}")

    print(f"\n{'=' * 70}")
    print(f"验收结论: {'【PASS 达到发布门槛】' if report['verdict'] == 'PASS' else '【FAIL 未达发布门槛】'}")
    for r in fail_reasons:
        print(f"  - {r}")
    print(f"{'=' * 70}")

    if out_path:
        _dump_report(report, out_path)
    return report["verdict"] == "PASS"


# ============================================================
# 落盘
# ============================================================

def _dump_report(report: Dict[str, Any], out_path: str) -> None:
    """把 report.json 落盘（跨批次自动 diff 用）。"""
    path = os.path.abspath(out_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] 结构化报告已落盘: {path}")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="舆情负面精判 SFT 评测验收")
    ap.add_argument("--test_file", default="data/gold500_test.jsonl",
                    help="[模式A] 单文件数据集自检路径")
    ap.add_argument("--gold_file", help="[模式B] 金标文件")
    ap.add_argument("--pred_file", help="[模式B] 模型预测文件")
    ap.add_argument("--out", default="output/report.json",
                    help="结构化报告落盘路径（默认 output/report.json）")
    ap.add_argument("--old_error_list", default=None,
                    help="[模式B] 门 1 用的上一版错题 id 列表（每行一个 id）")
    args = ap.parse_args()

    if args.gold_file and args.pred_file:
        ok = run_dual_file_eval(args.gold_file, args.pred_file,
                                out_path=args.out,
                                old_error_list=args.old_error_list)
    else:
        ok = run_single_file_check(args.test_file, out_path=args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
