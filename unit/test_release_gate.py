#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发布门四道判定 · 单元测试（零外部依赖）

覆盖四门各自 PASS / FAIL / N/A 三态与归因表（TODO 5.5）:
  门 1 · 历史错题修复: ≥95% / 无旧错题集 → N/A / 顽固样本记归因
  门 2 · 金标不退化:   安全字段 0 容差 / 普通字段 ≤1pp / 无基线记账不假装
  门 3 · 全链路安全:   P0 召回与查准不低于在役 / 无 L3 指标 → N/A
  门 4 · 格式合规:     JSON 合法率 ≥99.8%

运行:
    python3 -m unittest unit.test_release_gate -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from release_gate import (  # noqa: E402
    THRESH_GATE1_FIX_RATE, THRESH_GATE2_ORDINARY_DROP, THRESH_GATE4_JSON_VALID,
    evaluate_gates, overall_verdict,
)


def _gold(i, sent="negative", hc="irony", risk=("product", "marketing")):
    return {"id": f"e{i}", "sentiment": sent, "hard_case_type": hc,
            "risk_categories": list(risk)}


def _pred_ok(i, sent="negative", hc="irony", risk=("product", "marketing")):
    return {"id": f"e{i}", "sentiment": sent, "hard_case_type": hc,
            "risk_categories": list(risk), "valid_json": True}


# 20 条全对的基线场景
GOLD_OK = [_gold(i) for i in range(20)]
PRED_OK = [_pred_ok(i) for i in range(20)]


class TestGate1OldErrorFix(unittest.TestCase):
    def test_no_old_errors_is_na(self):
        """缺旧错题集 → N/A，绝不假装 PASS（评估手册缺口 5）。"""
        g = evaluate_gates(GOLD_OK, PRED_OK)
        self.assertEqual(g["gate1"], "N/A")

    def test_all_fixed_passes(self):
        g = evaluate_gates(GOLD_OK, PRED_OK, old_error_ids=[f"e{i}" for i in range(20)])
        self.assertEqual(g["gate1"], "PASS")
        self.assertEqual(g["details"]["gate1"]["fix_rate"], 100.0)

    def test_below_95_fails(self):
        # 20 条旧错题修好 18 条 = 90% < 95%
        pred = [(_pred_ok(i) if i < 18 else
                 dict(_pred_ok(i), sentiment="positive")) for i in range(20)]
        g = evaluate_gates(GOLD_OK, pred, old_error_ids=[f"e{i}" for i in range(20)])
        self.assertEqual(g["gate1"], "FAIL")
        self.assertAlmostEqual(g["details"]["gate1"]["fix_rate"], 90.0)

    def test_unverifiable_old_error_attributed(self):
        """旧错题不在本次金标 → 记归因，按未修复计（不悄悄丢掉）。"""
        g = evaluate_gates(GOLD_OK, PRED_OK, old_error_ids=["e0", "ghost_id"])
        self.assertEqual(g["gate1"], "FAIL")
        self.assertEqual(g["details"]["gate1"]["fix_rate"], 50.0)
        self.assertTrue(any(a["gate"] == "gate1" and a["sample_id"] == "ghost_id"
                            for a in g["attribution"]))

    def test_threshold_is_95_not_100(self):
        """门槛 95%：允许少量顽固样本（100% = 背死旧错牺牲泛化）。"""
        self.assertEqual(THRESH_GATE1_FIX_RATE, 95.0)


class TestGate2NoRegression(unittest.TestCase):
    def test_perfect_passes(self):
        g = evaluate_gates(GOLD_OK, PRED_OK)
        self.assertEqual(g["gate2"], "PASS")

    def test_safety_field_zero_tolerance_with_baseline(self):
        """安全字段（负向召回）0 容差：掉 0.01pp 也 FAIL。"""
        pred = [dict(_pred_ok(i), sentiment="positive") for i in range(20)]
        baseline = {"negative_recall": 100.0, "irony_recall": 100.0,
                    "macro_f1": 100.0, "subset_accuracy": 100.0}
        g = evaluate_gates(GOLD_OK, pred, baseline=baseline)
        self.assertEqual(g["gate2"], "FAIL")
        self.assertTrue(any(a["gate"] == "gate2" and "negative_recall" in a["sample_id"]
                            for a in g["attribution"]))

    def test_ordinary_field_within_tolerance_passes(self):
        """普通字段退化 ≤1pp 容差内不 FAIL。"""
        # 安全字段必须保持不动（基线也是 100），只动普通字段：
        # 400 条全 neutral / irony / 空风险，其中 2 条 sentiment 误判成 negative；
        # macro_f1 掉约 0.5pp（容差内），negative_recall 与 irony_recall 不受影响。
        n = 400
        gold = [_gold(i, sent="neutral", hc="irony", risk=()) for i in range(n)]
        pred = [(_pred_ok(i, sent="neutral", hc="irony", risk=()) if i >= 2 else
                 _pred_ok(i, sent="negative", hc="irony", risk=()))
                for i in range(n)]
        baseline = {"irony_recall": 100.0, "macro_f1": 100.0, "subset_accuracy": 100.0}
        g = evaluate_gates(gold, pred, baseline=baseline)
        self.assertEqual(g["gate2"], "PASS")

    def test_ordinary_field_beyond_tolerance_fails(self):
        """普通字段退化 >1pp FAIL（容差 1pp，不冲 0）。"""
        pred = [dict(_pred_ok(i), hard_case_type="explicit") for i in range(20)]
        baseline = {"negative_recall": 100.0, "irony_recall": 100.0,
                    "macro_f1": 100.0, "subset_accuracy": 100.0}
        g = evaluate_gates(GOLD_OK, pred, baseline=baseline)
        self.assertEqual(g["gate2"], "FAIL")
        self.assertTrue(any("macro_f1" in a["sample_id"] for a in g["attribution"]))

    def test_no_baseline_skips_and_records(self):
        """无在役基线 → 该项不参与退化判定，但必须记账（不假装 PASS/FAIL）。"""
        g = evaluate_gates(GOLD_OK, PRED_OK)
        skipped = g["details"]["gate2"]["skipped"]
        self.assertTrue(any("macro_f1" in s for s in skipped))

    def test_support_zero_field_not_judged_as_zero(self):
        """金标没有 irony 样本 → irony 召回不可判定，不得当 0% 误杀。"""
        gold = [_gold(i, hc="explicit") for i in range(20)]
        pred = [_pred_ok(i, hc="explicit") for i in range(20)]
        g = evaluate_gates(gold, pred)
        self.assertEqual(g["gate2"], "PASS")
        self.assertTrue(any("irony" in s
                            for s in g["details"]["gate2"]["skipped"]))

    def test_safety_floor_without_baseline_uses_threshold(self):
        """无基线但有门槛值 → 用门槛当绝对安全底线。"""
        pred = [dict(_pred_ok(i), sentiment="positive") for i in range(20)]
        g = evaluate_gates(GOLD_OK, pred, thresholds={"negative_recall": 90.0})
        self.assertEqual(g["gate2"], "FAIL")


class TestGate3L3Safety(unittest.TestCase):
    def test_no_l3_is_na(self):
        """缺 L3 联动指标 → N/A，绝不假装 PASS。"""
        g = evaluate_gates(GOLD_OK, PRED_OK)
        self.assertEqual(g["gate3"], "N/A")

    def test_l3_not_below_baseline_passes(self):
        g = evaluate_gates(GOLD_OK, PRED_OK,
                           l3={"p0_recall": 0.99, "p0_precision": 0.97,
                               "baseline_p0_recall": 0.99, "baseline_p0_precision": 0.96})
        self.assertEqual(g["gate3"], "PASS")

    def test_l3_recall_drop_fails(self):
        g = evaluate_gates(GOLD_OK, PRED_OK,
                           l3={"p0_recall": 0.95, "p0_precision": 0.97,
                               "baseline_p0_recall": 0.99, "baseline_p0_precision": 0.96})
        self.assertEqual(g["gate3"], "FAIL")
        self.assertTrue(any("p0_recall" in a["sample_id"] for a in g["attribution"]))

    def test_l3_precision_drop_fails(self):
        g = evaluate_gates(GOLD_OK, PRED_OK,
                           l3={"p0_recall": 0.99, "p0_precision": 0.90,
                               "baseline_p0_recall": 0.99, "baseline_p0_precision": 0.96})
        self.assertEqual(g["gate3"], "FAIL")


class TestGate4Format(unittest.TestCase):
    def test_all_valid_passes(self):
        g = evaluate_gates(GOLD_OK, PRED_OK)
        self.assertEqual(g["gate4"], "PASS")

    def test_below_threshold_fails(self):
        # 20 条里 1 条崩格式 = 95% < 99.8%
        pred = [(_pred_ok(i) if i < 19 else {"valid_json": False}) for i in range(20)]
        g = evaluate_gates(GOLD_OK, pred)
        self.assertEqual(g["gate4"], "FAIL")
        self.assertAlmostEqual(g["details"]["gate4"]["json_valid_rate"], 95.0)

    def test_default_flag_counts_as_valid(self):
        """预测没带 valid_json 标志 → 视为合法（由上游解析层保证）。"""
        pred = [{k: v for k, v in _pred_ok(i).items() if k != "valid_json"}
                for i in range(20)]
        g = evaluate_gates(GOLD_OK, pred)
        self.assertEqual(g["gate4"], "PASS")

    def test_no_samples_is_na(self):
        g = evaluate_gates([], [])
        self.assertEqual(g["gate4"], "N/A")

    def test_threshold_constant(self):
        self.assertEqual(THRESH_GATE4_JSON_VALID, 99.8)


class TestOverallVerdict(unittest.TestCase):
    def test_and_relation_any_fail_stops(self):
        """四门是「与」关系：任一 FAIL 即中止。"""
        gates = {"gate1": "PASS", "gate2": "FAIL", "gate3": "PASS", "gate4": "PASS"}
        self.assertEqual(overall_verdict(gates), "FAIL")

    def test_na_does_not_block(self):
        gates = {"gate1": "N/A", "gate2": "PASS", "gate3": "N/A", "gate4": "PASS"}
        self.assertEqual(overall_verdict(gates), "PASS")

    def test_all_pass(self):
        gates = {"gate1": "PASS", "gate2": "PASS", "gate3": "PASS", "gate4": "PASS"}
        self.assertEqual(overall_verdict(gates), "PASS")

    def test_tolerance_constant(self):
        self.assertEqual(THRESH_GATE2_ORDINARY_DROP, 1.0)


class TestGate2UnparseablePreds(unittest.TestCase):
    """预测全部不可解析时，门 2 必须记 N/A —— 不许因初始值 True 报 PASS。"""

    def test_all_unparseable_is_na_not_pass(self):
        gold = [_gold(i) for i in range(6)]
        pred = [{"valid_json": False} for _ in range(6)]
        g = evaluate_gates(gold, pred)
        self.assertEqual(g["gate2"], "N/A")
        self.assertEqual(g["details"]["gate2"]["n_usable_pred"], 0)
        self.assertTrue(g["details"]["gate2"].get("reason"))

    def test_partial_unparseable_still_judged(self):
        """只要有可用预测就必须实际判定，不能退化成 N/A。"""
        gold = [_gold(i) for i in range(6)]
        pred = [_pred_ok(i) for i in range(6)]
        pred[0] = {"valid_json": False}
        g = evaluate_gates(gold, pred)
        self.assertIn(g["gate2"], ("PASS", "FAIL"))
        self.assertEqual(g["details"]["gate2"]["n_usable_pred"], 5)

    def test_na_gate2_does_not_block_release(self):
        """N/A 不阻断（与门 1/门 3 同语义）；真正拦截的是门 4。"""
        gold = [_gold(i) for i in range(6)]
        pred = [{"valid_json": False} for _ in range(6)]
        g = evaluate_gates(gold, pred)
        self.assertEqual(g["gate4"], "FAIL")
        self.assertEqual(overall_verdict(g), "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
