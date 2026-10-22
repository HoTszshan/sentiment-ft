#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标纯函数层 · 零外部依赖（只吃标准库）

为什么单独抽一层（评估手册 §1「评分代码必须配单元测试」）:
  评分代码改错了**不会报错** —— 求和写反、罚分挪档、集合比较用错算子，
  跑起来照样绿，只是分数偏了。而分数是拿来当结论用的，所以指标必须是
  **可单测的纯函数**：无 I/O、无打印、无全局状态。

本文件提供的六组口径:

  1. `prf(tp, fp, fn)`                 —— 基础 P/R/F1
  2. `macro_f1(golds, preds, labels)`   —— 单值字段逐类 P/R/F1 + 宏均（support=0 不入宏均）
  3. `subset_accuracy(golds, preds)`    —— 多标签集合精确匹配（漏标多标对称双罚）+ 空表专项 + per-key 诊断
  4. `confusion_slices(...)`            —— 通用分桶（任意维度，桶内样本数 <5 标 insufficient）
  5. `illegal_label_counts(...)`        —— 闭集外的非法标签单列计数（缺口 12）
  6. `pa_fp_pair(gold_sent, pred_sent)` —— 负向召回与误报率**成对**返回（缺口 9）

**最容易被写错的一条（务必背下来）**:
  误报率 ≠ FP/(FP+TN)。本项目口径 = `误报条数 / 模型判为负面的条数`
  （业务关心的是「推给人看的预警里有多少是假的」），
  也就是 negative 类的 precision 补数。故 `false_alarm_rate = 1 - negative_precision`。
  负向召回与误报率**必须成对出现**，单报其一没有意义。
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 闭集枚举统一从 schema_contract 取 (训练/评测/质检三侧同源)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_contract import (  # noqa: E402
    VALID_HARD_CASE,
    VALID_RELEVANCE,
    VALID_RISK,
    VALID_SENTIMENT,
)

# 分桶最小样本数：低于此值只报数不报率 (长尾桶读数抖动 ±6~7pp，是抽样噪声不是能力)
MIN_BUCKET_N = 5


# ============================================================
# 1. 基础 P/R/F1
# ============================================================

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """标准 P/R/F1（0~1 口径）。分母为 0 时该分量取 0，不抛异常。"""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ============================================================
# 2. 单值字段宏均 F1
# ============================================================

def macro_f1(golds: Sequence[Any], preds: Sequence[Any],
             labels: Sequence[str],
             valid_set: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    单值字段（sentiment / hard_case_type / relevance）的逐类 P/R/F1 + 宏均 F1。

    口径要点:
      - 单位一律 **%**（与 SOP 门槛同量纲，可直接比）；
      - `support == 0` 的类**不入宏均**（否则没出现的类会把宏均往下拽）；
      - 传了 `valid_set` 时：**闭集外的预测值整条样本剔除**（既不记对也不记错），
        闭集外的值由 `illegal_label_counts` 单列计数（缺口 12）——
        堵住「崩掉 JSON = 空表正确」这条反向优化通道。

    返回: {"per_class": {label: {precision, recall, f1, support}}, "macro_f1": float,
           "n_scored": int, "n_skipped_illegal": int}
    """
    assert len(golds) == len(preds), "golds / preds 长度必须一致"
    valid = set(valid_set) if valid_set is not None else None
    keep = []
    skipped = 0
    parse_failed = 0
    for g, p in zip(golds, preds):
        # 解析失败（pred 为 None）整条剔除 —— 崩 JSON 不得进语义指标
        # （否则既不记对也不记错，等于白送一条；与 subset_accuracy 同口径）
        if p is None:
            parse_failed += 1
            continue
        if valid is not None and str(p) not in valid:
            skipped += 1
            continue
        keep.append((g, p))

    kg = [a for a, _ in keep]
    kp = [b for _, b in keep]

    per: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for g, p in zip(kg, kp) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(kg, kp) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(kg, kp) if g == lab and p != lab)
        p_, r_, f_ = prf(tp, fp, fn)
        support = sum(1 for g in kg if g == lab)
        per[lab] = {"precision": p_ * 100, "recall": r_ * 100,
                    "f1": f_ * 100, "support": support}
        if support > 0:
            f1s.append(f_ * 100)
    return {"per_class": per,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "n_scored": len(kg),
            "n_skipped_illegal": skipped,
            "n_parse_failed": parse_failed}


# ============================================================
# 3. 多标签集合精确匹配
# ============================================================

def _as_key_set(x: Any) -> frozenset:
    """风险标签集合归一：非数组当空表，元素统一转字符串。"""
    if not isinstance(x, (list, tuple)):
        return frozenset()
    return frozenset(str(v) for v in x)


def subset_accuracy(golds: Sequence[Any], preds: Sequence[Any],
                    valid_set: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    risk_categories 的 Subset Accuracy —— 漏标算错、多标也算错，整条集合完全相等才得分。

    同时产出 per-key 的 P/R/F1 作为**诊断线**（SOP 明确：诊断用，不进准入门槛）。

    口径要点:
      - 单位 %；
      - **解析失败（pred 为 None）整条剔除**，不参与比对 —— 否则金标为空表时
        「崩 JSON」经 `_as_key_set(None) → 空集` 白拿分（正是要堵的反向优化通道）；
        解析失败进格式失败口径，不进语义指标；
      - 空表专项：金标为空表的样本数 + 空表精确率（空表错成有标签 = 强行打标幻觉）；
      - 金标为空表而预测非空 → 记为 FP 侧的误报（业务上比漏报更刺眼：凭空造风险）；
      - 传了 `valid_set` 时：预测含**闭集外标签**的样本整条剔除（既不记对也不记错），
        闭集外标签由 `illegal_label_counts` 单列计数（缺口 12）。

    返回: {"subset_accuracy", "empty_support", "empty_precision", "per_key",
           "n_scored", "n_skipped_illegal", "n_parse_failed"}
    """
    assert len(golds) == len(preds), "golds / preds 长度必须一致"
    valid = set(valid_set) if valid_set is not None else None

    gsets_all = [_as_key_set(g) for g in golds]
    psets_all = [_as_key_set(p) for p in preds]

    gsets, psets = [], []
    skipped = 0
    parse_failed = 0
    for i, (g, p) in enumerate(zip(gsets_all, psets_all)):
        # 解析失败（pred 为 None）→ 整条剔除，进格式失败口径，不给空表白拿分
        if preds[i] is None:
            parse_failed += 1
            continue
        if valid is not None and not p.issubset(valid):
            skipped += 1
            continue
        gsets.append(g)
        psets.append(p)

    n = len(gsets)
    exact = sum(1 for g, p in zip(gsets, psets) if g == p)

    per_key: Dict[str, Dict[str, float]] = {}
    for k in sorted(VALID_RISK):
        tp = sum(1 for g, p in zip(gsets, psets) if k in g and k in p)
        fp = sum(1 for g, p in zip(gsets, psets) if k not in g and k in p)
        fn = sum(1 for g, p in zip(gsets, psets) if k in g and k not in p)
        p_, r_, f_ = prf(tp, fp, fn)
        per_key[k] = {"precision": p_ * 100, "recall": r_ * 100, "f1": f_ * 100,
                      "support": sum(1 for g in gsets if k in g)}

    empty_idx = [i for i, g in enumerate(gsets) if len(g) == 0]
    empty_correct = sum(1 for i in empty_idx if len(psets[i]) == 0)

    return {
        "subset_accuracy": exact / n * 100 if n else 0.0,
        "empty_support": len(empty_idx),
        "empty_precision": empty_correct / len(empty_idx) * 100 if empty_idx else 100.0,
        "per_key": per_key,
        "n_scored": n,
        "n_skipped_illegal": skipped,
        "n_parse_failed": parse_failed,
    }


# ============================================================
# 4. 通用分桶
# ============================================================

def confusion_slices(gold_meta: Dict[str, Sequence[Any]],
                      preds: Sequence[Dict[str, Any]],
                      dims: Sequence[str],
                      gold: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """
    通用分桶：按任意维度切，逐桶算 n / risk_exact / sentiment_acc / hard_case_acc。

    参数:
      gold_meta —— {dim: [bucket_value_per_sample]}，长度须与 preds 一致；
      preds     —— [pred_dict_per_sample]（含 sentiment / hard_case_type / risk_categories）；
      dims      —— 要切分的维度名列表（gold_meta 的子集键）；
      gold      —— {field: [value_per_sample]}，可选；给了才算准确率，
                   不给则只报 n（诊断「分布是否失衡」用）。

    返回: {dim: {bucket: {"n", "risk_exact", "sentiment_acc", "hard_case_acc",
                          "insufficient"(bool)}}}

    **小样本保护**: 桶内 n < MIN_BUCKET_N 时置 `insufficient=True`，
    上层只报样本数不报率 —— 长尾桶读数抖动 ±6~7pp，报出来是抽样噪声不是能力。
    """
    gold = gold or {}
    g_rk_all = gold.get("risk_categories")
    g_sent_all = gold.get("sentiment")
    g_hc_all = gold.get("hard_case_type")

    out: Dict[str, Any] = {}
    for dim in dims:
        buckets = list(gold_meta.get(dim, []))
        per_bucket: Dict[str, Dict[str, Any]] = {}
        for i, b in enumerate(buckets):
            key = str(b) if b is not None else "unknown"
            slot = per_bucket.setdefault(key, {"n": 0, "risk_exact": 0,
                                               "sentiment_ok": 0, "hard_case_ok": 0})
            slot["n"] += 1
            p = preds[i] if i < len(preds) else None
            if not p:
                continue
            if g_rk_all is not None:
                slot["risk_exact"] += int(_as_key_set(g_rk_all[i]) == _as_key_set(p.get("risk_categories")))
            if g_sent_all is not None:
                slot["sentiment_ok"] += int(g_sent_all[i] == p.get("sentiment"))
            if g_hc_all is not None:
                slot["hard_case_ok"] += int(g_hc_all[i] == p.get("hard_case_type"))

        for key, slot in per_bucket.items():
            n = slot["n"]
            entry: Dict[str, Any] = {"n": n}
            entry["insufficient"] = n < MIN_BUCKET_N
            if gold:
                entry["risk_exact"] = round(slot["risk_exact"] / n * 100, 2) if n else 0.0
                entry["sentiment_acc"] = round(slot["sentiment_ok"] / n * 100, 2) if n else 0.0
                entry["hard_case_acc"] = round(slot["hard_case_ok"] / n * 100, 2) if n else 0.0
            per_bucket[key] = entry
        out[dim] = per_bucket
    return out


# ============================================================
# 5. 非法标签计数
# ============================================================

def illegal_label_counts(preds: Sequence[Dict[str, Any]], field: str,
                         valid_set: Iterable[str]) -> Dict[str, int]:
    """
    闭集外的预测值单列计数（缺口 12）。

    口径:
      - 只统计**闭集外**的值；闭集内的值不计数；
      - 解析失败（pred 为 None）的样本不计在这里，它们进格式失败口径；
      - 非法标签**从集合类指标中剔除**（既不记对也不记错），
        防止「崩掉 JSON = 空表正确」这条反向优化通道。

    返回: {非法值: 次数}
    """
    valid = set(valid_set)
    counts: Dict[str, int] = {}
    for p in preds:
        if not p:
            continue
        v = p.get(field)
        if isinstance(v, (list, tuple)):
            for item in v:
                if str(item) not in valid:
                    counts[str(item)] = counts.get(str(item), 0) + 1
            continue
        if v is None:
            continue
        if str(v) not in valid:
            counts[str(v)] = counts.get(str(v), 0) + 1
    return counts


def illegal_risk_counts(preds: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """risk_categories 专用版本：闭集外的标签逐个计数（多标签逐项计）。"""
    return illegal_label_counts(preds, "risk_categories", VALID_RISK)


# ============================================================
# 6. 负向召回 × 误报率（成对口径）
# ============================================================

def pa_fp_pair(gold_sent: Sequence[Any], pred_sent: Sequence[Any]) -> Dict[str, Any]:
    """
    负向召回与误报率**成对**返回（缺口 9：禁止单报其一）。

    口径（务必背下来）:
      - 负向召回 = TP / (TP + FN)，分母是**金标为 negative 的全部样本**；
      - 误报率   = FP / (模型判为 negative 的条数) = 1 - negative_precision
                   （**不是** FP/(FP+TN)）——
                   业务关心的是「推给人看的预警里有多少是假的」；
      - 两者必须同时出现在报告里，单报任一都是半截数字。

    返回: {"negative_recall", "false_alarm_rate", "negative_precision",
           "tp", "fp", "fn", "tn", "support_negative", "n_pred_negative",
           "recall_defined", "false_alarm_defined"}

    分母为 0 的两种退化情形（都必须显式标注，不许静默给 0 或 100）:
      - `support_negative == 0`：金标里没有负面 → 召回不可判定，`recall_defined=False`；
      - `n_pred_negative == 0`：模型一条负面都没报 → 没有预警推给人看，
        误报率**空真**为 0（`false_alarm_defined=False`），此时真正的信号是召回=0，
        两个数必须一起读。
    """
    assert len(gold_sent) == len(pred_sent), "golds / preds 长度必须一致"
    tp = fp = fn = tn = 0
    for g, p in zip(gold_sent, pred_sent):
        g_neg = (g == "negative")
        p_neg = (p == "negative")
        if g_neg and p_neg:
            tp += 1
        elif g_neg and not p_neg:
            fn += 1
        elif (not g_neg) and p_neg:
            fp += 1
        else:
            tn += 1
    support_neg = tp + fn
    n_pred_neg = tp + fp
    neg_recall = tp / support_neg * 100 if support_neg else 0.0
    if n_pred_neg:
        neg_prec = tp / n_pred_neg * 100
        far = 100.0 - neg_prec
    else:
        # 一条负面都没报：没有预警可误，误报率空真为 0（精确率同样空真 100）
        neg_prec = 100.0
        far = 0.0
    return {
        "negative_recall": neg_recall,
        "false_alarm_rate": far,
        "negative_precision": neg_prec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support_negative": support_neg,
        "n_pred_negative": n_pred_neg,
        "recall_defined": support_neg > 0,
        "false_alarm_defined": n_pred_neg > 0,
    }
