#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发布门四道独立判定 · 纯逻辑（评估手册缺口 5）

背景: 旧版 `evaluate_sft.py` 把四道门压成**一个 `ok` 布尔** —— 门 1（历史错题修好了没）
与门 3（接 L3 规则后安不安全）根本没有接口，等于这两道门「永远假装通过」。
更糟的是：一个布尔把「哪道门没过」也吞了，修都不知道往哪修。

四道门是**「与」关系，不求总分**：任一门不过即中止发布；修复后**从门 1 全量重跑**，
不补跑单门（否则前置门的副作用被放过）。

| 门 | 判什么 | 门槛 | 无输入时 |
|---|---|---|---|
| 门 1 · 历史错题修复 | 上次错题这次修好没 | ≥ 95% | `"N/A"`（不假装 PASS） |
| 门 2 · 金标不退化 | 金标有没有回退 | 安全字段 0 容差；普通字段退化 ≤ 1% | 仍判定（退化项无基线时跳过并记账） |
| 门 3 · 全链路安全底线 | 接 L3 规则后是否安全 | P0 召回/查准不低于在役版 | `"N/A"`（不假装 PASS） |
| 门 4 · 格式合规 | 下游能否解析 | JSON 合法率 ≥ 99.8% | 仍判定（从 pred 的解析标志算） |

**门 1 与门 2 不重复**：门 2 看的是同一批金标会不会回退（守成），门 1 看的是
已知错误有没有被真正修掉（进攻）。只守门 2，模型可以靠什么都不改变而全门通过；
只守门 1，又会为了修旧错把整体能力拉花。两道门必须同时成立。

**为什么门 1 不是 100%**：100% 修复率等价于要求模型把旧错题背下来，代价是泛化能力；
95% 是「绝大部分修掉、允许少量顽固样本」，与门 2 的 0 容差互补。

本模块是纯逻辑：只吃已经解析好的 dict 列表，不读文件、不打网络，可直接单测。
"""

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_metrics import _as_key_set, macro_f1, pa_fp_pair, subset_accuracy  # noqa: E402
from schema_contract import VALID_HARD_CASE, VALID_RISK, VALID_SENTIMENT  # noqa: E402

# ============================================================
# 门槛常量
# ============================================================

THRESH_GATE1_FIX_RATE = 95.0     # % 历史错题修复率
THRESH_GATE2_SAFE_DROP = 0.0      # pp 安全字段退化容差（0 容差）
THRESH_GATE2_ORDINARY_DROP = 1.0  # pp 普通字段退化容差
THRESH_GATE4_JSON_VALID = 99.8    # % JSON 合法率

# 门 2 的「安全字段」：极性方向错了代价最大（漏报危机 / 误报刷人工）
SAFETY_FIELDS = ("negative_recall", "irony_recall")
# 门 2 的「普通字段」：允许小幅退化（1pp）
ORDINARY_FIELDS = ("macro_f1", "subset_accuracy")


# ============================================================
# 辅助
# ============================================================

def _item_correct(gold: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> bool:
    """
    单条「判别主线全对」判定：sentiment + hard_case_type + risk_categories 三项全中。

    GRPO `accuracy_pass` 已阶梯式放宽为「情感对即算有效」，
    本评测侧保持**严格口径不变**——发布门是验收关卡，放宽会放水；两侧行为有意分叉，
    不再声称对齐。若需在评测中复现 GRPO 的宽松口径，另加参数，不动默认值。
    """
    if not pred:
        return False
    if gold.get("sentiment") != pred.get("sentiment"):
        return False
    if gold.get("hard_case_type") != pred.get("hard_case_type"):
        return False
    if _as_key_set(gold.get("risk_categories")) != _as_key_set(pred.get("risk_categories")):
        return False
    return True


def _gold_id(gold: Dict[str, Any], index: int) -> str:
    """样本标识：优先取 id/event_id，否则用行号。"""
    for k in ("id", "event_id", "sample_id"):
        v = gold.get(k)
        if v:
            return str(v)
    return f"#{index}"


def _compute_metrics(gold: Sequence[Dict[str, Any]],
                     pred: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    从金标/预测对算出门 2 要用的四个指标。

    **支撑集为 0 的指标返回 None**（不可判定），而不是 0 ——
    「金标里根本没有 irony 样本」不等于「irony 召回 0%」，
    拿 0 去和安全底线比会误杀。None 的指标在门 2 里被跳过并记账。
    """
    golds = [g.get("sentiment") for g in gold]
    preds = [p.get("sentiment") if p else None for p in pred]
    pa = pa_fp_pair(golds, preds)
    # 闭集外的预测整条剔除（与评测报告同口径；非法标签由 illegal_label_counts 单列）
    hc = macro_f1([g.get("hard_case_type") for g in gold],
                  [p.get("hard_case_type") if p else None for p in pred],
                  sorted(VALID_HARD_CASE), VALID_HARD_CASE)
    rk = subset_accuracy([g.get("risk_categories") for g in gold],
                         [p.get("risk_categories") if p else None for p in pred],
                         VALID_RISK)
    irony = hc["per_class"].get("irony", {})
    return {
        "negative_recall": pa["negative_recall"] if pa["support_negative"] else None,
        "irony_recall": (irony.get("recall") if irony.get("support", 0) > 0 else None),
        "macro_f1": hc["macro_f1"] if hc["n_scored"] else None,
        "subset_accuracy": (rk["subset_accuracy"] if rk["n_scored"] else None),
    }


# ============================================================
# 四门判定
# ============================================================

def evaluate_gates(gold: Sequence[Dict[str, Any]],
                   pred: Sequence[Dict[str, Any]],
                   old_error_ids: Optional[Sequence[Any]] = None,
                   baseline: Optional[Dict[str, float]] = None,
                   l3: Optional[Dict[str, Any]] = None,
                   thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    四道发布门独立判定。

    参数:
      gold          —— 金标列表，每条含 sentiment / hard_case_type / risk_categories
                       （可选 id / event_id 供门 1 对齐）；
      pred          —— 预测列表，每条含同样字段 + 解析标志
                       （`valid_json` / `valid_tags` / `evidence_ok`，缺省视为 True）；
      old_error_ids —— 上一版的错题 id 列表（门 1 用）；None → 门 1 返回 "N/A"；
      baseline      —— 在役版/上一版指标字典（门 2 的退化对比基线）；
                       形如 {"negative_recall": 90.0, "irony_recall": 88.0,
                             "macro_f1": 87.0, "subset_accuracy": 90.0}；
      l3            —— 端到端 L3 规则联动指标（门 3 用），形如
                       {"p0_recall": .., "p0_precision": ..,
                        "baseline_p0_recall": .., "baseline_p0_precision": ..}；
      thresholds    —— 门槛覆盖，形如 {"negative_recall": 90.0}（门 2 的绝对安全底线）。

    返回: {"gate1": "PASS|FAIL|N/A", "gate2": ..., "gate3": ..., "gate4": ...,
           "attribution": [{"gate": .., "sample_id": .., "reason": ..}, ...],
           "details": {...}}
    """
    thresholds = thresholds or {}
    attribution: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    gold = list(gold or [])
    pred = list(pred or [])

    # ---------------- 门 1 · 历史错题修复 ----------------
    if not old_error_ids:
        details["gate1"] = {"status": "N/A", "reason": "未提供旧错题集（--old_error_list）"}
        gate1 = "N/A"
    else:
        gold_ids = [_gold_id(g, i) for i, g in enumerate(gold)]
        id_to_idx = {gid: i for i, gid in enumerate(gold_ids)}
        total, fixed = 0, 0
        for eid in old_error_ids:
            key = str(eid)
            idx = id_to_idx.get(key)
            if idx is None:
                # 旧错题在本次金标里找不到 → 视为未修复，记账
                total += 1
                attribution.append({"gate": "gate1", "sample_id": key,
                                    "reason": "旧错题不在本次金标中，无法验证修复"})
                continue
            total += 1
            if _item_correct(gold[idx], pred[idx] if idx < len(pred) else None):
                fixed += 1
            else:
                attribution.append({"gate": "gate1", "sample_id": key,
                                    "reason": "上一版错题本次仍未判对"})
        rate = fixed / total * 100 if total else 0.0
        gate1 = "PASS" if rate >= THRESH_GATE1_FIX_RATE else "FAIL"
        details["gate1"] = {"status": gate1, "fix_rate": rate,
                            "n_old_errors": total, "n_fixed": fixed,
                            "threshold": THRESH_GATE1_FIX_RATE}

    # ---------------- 门 2 · 金标不退化 ----------------
    cur = _compute_metrics(gold, pred)
    details["gate2"] = {"current": cur, "baseline": baseline or {},
                        "safety_fields": list(SAFETY_FIELDS),
                        "ordinary_fields": list(ORDINARY_FIELDS),
                        "skipped": []}
    gate2_ok = True
    # 基线值缺失 / 为 None → 按「缺基线」跳过并记账，不许 float(None) 崩门禁
    def _baseline_value(field: str) -> Optional[float]:
        if not baseline or field not in baseline:
            return None
        v = baseline[field]
        return float(v) if isinstance(v, (int, float)) else None

    # 2a. 绝对安全底线（0 容差：有基线时以基线为准，无基线时用门槛常量）
    for field in SAFETY_FIELDS:
        cur_v = cur.get(field)
        if cur_v is None:
            details["gate2"]["skipped"].append(
                f"{field}（本次金标无支撑样本，不可判定）")
            continue
        base_v = _baseline_value(field)
        if base_v is not None:
            if cur_v < base_v:
                gate2_ok = False
                attribution.append({"gate": "gate2", "sample_id": field,
                                    "reason": f"安全字段退化（0 容差）：{cur_v:.2f} < 在役 {base_v:.2f}"})
        else:
            thr = thresholds.get(field)
            if field == "irony_recall" and thr is None:
                thr = 90.0
            if thr is None:
                details["gate2"]["skipped"].append(
                    f"{field}（缺在役基线与门槛值，未参与判定）")
                continue
            if cur_v < float(thr):
                gate2_ok = False
                attribution.append({"gate": "gate2", "sample_id": field,
                                    "reason": f"安全字段未达底线：{cur_v:.2f} < {float(thr):.2f}"})
    # 2b. 普通字段退化容差 ≤ 1pp（无基线时跳过并记账）
    for field in ORDINARY_FIELDS:
        cur_v = cur.get(field)
        if cur_v is None:
            details["gate2"]["skipped"].append(f"{field}（本次金标无支撑样本，不可判定）")
            continue
        base_v = _baseline_value(field)
        if base_v is None:
            details["gate2"]["skipped"].append(f"{field}（缺在役基线，未参与退化判定）")
            continue
        if base_v - cur_v > THRESH_GATE2_ORDINARY_DROP:
            gate2_ok = False
            attribution.append({"gate": "gate2", "sample_id": field,
                                "reason": f"普通字段退化超容差：{cur_v:.2f} vs 在役 {base_v:.2f}"
                                          f"（容差 {THRESH_GATE2_ORDINARY_DROP}pp）"})
    gate2 = "PASS" if gate2_ok else "FAIL"
    # 预测全部不可解析时，门 2 的四项指标虽能算出数字（如 recall=0.0），但那是
    # "没有输出"的产物，不是"判错了"。此时报 PASS 会让格式全崩的版本看起来安全字段过关。
    # 与门 1 / 门 3 同语义：无可判定输入即 N/A，不假装 PASS。
    n_usable = sum(1 for p in pred if p and p.get("valid_json", True))
    if pred and n_usable == 0:
        gate2 = "N/A"
        details["gate2"]["reason"] = "预测全部不可解析，安全字段无从判定"
        details["gate2"]["skipped"].append("预测可用数为 0 → 门 2 记 N/A，不参与发布判定")
    details["gate2"]["n_usable_pred"] = n_usable
    details["gate2"]["status"] = gate2

    # ---------------- 门 3 · 全链路安全底线 ----------------
    if not l3:
        details["gate3"] = {"status": "N/A", "reason": "未接 L3 规则引擎联动指标"}
        gate3 = "N/A"
    else:
        l3_ok = True
        for key, base_key in (("p0_recall", "baseline_p0_recall"),
                              ("p0_precision", "baseline_p0_precision")):
            if key not in l3:
                continue
            cur_v = float(l3[key])
            base_v = l3.get(base_key)
            if base_v is None:
                details["gate3"].setdefault("skipped", []).append(f"{key}（缺在役基线）")
                continue
            if cur_v < float(base_v):
                l3_ok = False
                attribution.append({"gate": "gate3", "sample_id": key,
                                    "reason": f"接 L3 后 {key} 低于在役：{cur_v:.2f} < {float(base_v):.2f}"})
        gate3 = "PASS" if l3_ok else "FAIL"
        details["gate3"] = {"status": gate3, "metrics": l3}

    # ---------------- 门 4 · 格式合规 ----------------
    total = len(pred)
    if total == 0:
        details["gate4"] = {"status": "N/A", "reason": "无预测样本"}
        gate4 = "N/A"
    else:
        ok_json = sum(1 for p in pred if p.get("valid_json", True))
        rate = ok_json / total * 100
        thr = float(thresholds.get("json_valid", THRESH_GATE4_JSON_VALID))
        gate4 = "PASS" if rate >= thr else "FAIL"
        if gate4 == "FAIL":
            attribution.append({"gate": "gate4", "sample_id": "-",
                                "reason": f"JSON 合法率 {rate:.2f}% < {thr}%"})
        details["gate4"] = {"status": gate4, "json_valid_rate": rate,
                            "n_samples": total, "threshold": thr}

    return {"gate1": gate1, "gate2": gate2, "gate3": gate3, "gate4": gate4,
            "attribution": attribution, "details": details}


def overall_verdict(gates: Dict[str, Any]) -> str:
    """
    四门是「与」关系：任一 FAIL 即中止发布；N/A 不阻断（无输入的门本就没法判）。
    """
    for g in ("gate1", "gate2", "gate3", "gate4"):
        if gates.get(g) == "FAIL":
            return "FAIL"
    return "PASS"
