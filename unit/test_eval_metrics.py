#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标纯函数层 · 单元测试（零外部依赖）

运行:
    python3 -m unittest unit.test_eval_metrics -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from eval_metrics import (  # noqa: E402
    MIN_BUCKET_N, confusion_slices, illegal_label_counts, macro_f1,
    pa_fp_pair, prf, subset_accuracy,
)
from schema_contract import VALID_HARD_CASE, VALID_RISK, VALID_SENTIMENT  # noqa: E402


class TestPrf(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(prf(5, 0, 0), (1.0, 1.0, 1.0))

    def test_all_zero_denominators(self):
        self.assertEqual(prf(0, 0, 0), (0.0, 0.0, 0.0))

    def test_precision_recall_f1(self):
        p, r, f = prf(3, 1, 2)
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 0.6)
        self.assertAlmostEqual(f, 2 * 0.75 * 0.6 / 1.35)


class TestMacroF1(unittest.TestCase):
    def test_support_zero_excluded_from_macro(self):
        """没出现的类不得把宏均往下拽（support=0 不入宏均）。"""
        golds = ["explicit", "explicit", "irony"]
        preds = ["explicit", "explicit", "irony"]
        st = macro_f1(golds, preds, sorted(VALID_HARD_CASE))
        self.assertEqual(st["macro_f1"], 100.0)
        self.assertEqual(st["per_class"]["metaphor"]["support"], 0)
        self.assertEqual(st["per_class"]["metaphor"]["f1"], 0.0)

    def test_partial_correctness(self):
        # explicit 3 条全对但多报了 2 条 → P=3/5 的代价；irony 2 条全错 → F1=0
        # 宏均 = (explicit F1 75 + irony F1 0) / 2 = 37.5
        golds = ["explicit"] * 3 + ["irony"] * 2
        preds = ["explicit"] * 3 + ["explicit"] * 2
        st = macro_f1(golds, preds, sorted(VALID_HARD_CASE))
        self.assertAlmostEqual(st["macro_f1"], 37.5)
        self.assertAlmostEqual(st["per_class"]["irony"]["recall"], 0.0)
        self.assertAlmostEqual(st["per_class"]["explicit"]["precision"], 60.0)
        self.assertEqual(st["per_class"]["explicit"]["support"], 3)

    def test_illegal_predictions_excluded(self):
        """闭集外的预测整条剔除（既不记对也不记错），不让它污染分母。"""
        golds = ["negative", "negative", "neutral"]
        preds = ["negative", "hate", "neutral"]
        st = macro_f1(golds, preds, sorted(VALID_SENTIMENT), VALID_SENTIMENT)
        self.assertEqual(st["n_skipped_illegal"], 1)
        self.assertEqual(st["n_scored"], 2)
        self.assertAlmostEqual(st["per_class"]["negative"]["recall"], 100.0)

    def test_empty_input(self):
        st = macro_f1([], [], sorted(VALID_SENTIMENT), VALID_SENTIMENT)
        self.assertEqual(st["macro_f1"], 0.0)
        self.assertEqual(st["n_scored"], 0)

    def test_parse_failed_dropped_from_denominator(self):
        """解析失败（pred=None）整条剔除，不计入 n_scored 分母——
        统一「丢弃」口径：崩 JSON 不得既不记对也不记错地白送，也不得改变分母语义。"""
        golds = ["negative", "negative", "neutral", "positive"]
        preds = ["negative", None, "neutral", "positive"]
        st = macro_f1(golds, preds, sorted(VALID_SENTIMENT), VALID_SENTIMENT)
        self.assertEqual(st["n_parse_failed"], 1)
        self.assertEqual(st["n_scored"], 3)
        self.assertAlmostEqual(st["per_class"]["negative"]["recall"], 100.0)
        self.assertAlmostEqual(st["macro_f1"], 100.0)


class TestSubsetAccuracy(unittest.TestCase):
    def test_exact_match_only(self):
        golds = [["product"], ["product", "service"]]
        preds = [["product"], ["product"]]
        st = subset_accuracy(golds, preds)
        self.assertAlmostEqual(st["subset_accuracy"], 50.0)

    def test_over_label_and_under_label_both_penalized(self):
        golds = [["product"], ["product"]]
        preds = [["product", "service"], ["product"]]
        st = subset_accuracy(golds, preds)
        self.assertAlmostEqual(st["subset_accuracy"], 50.0)

    def test_order_insensitive(self):
        st = subset_accuracy([["product", "service"]], [["service", "product"]])
        self.assertAlmostEqual(st["subset_accuracy"], 100.0)

    def test_empty_support_precision(self):
        # 金标空表 3 条：2 条判空对、1 条强行打标 → 空表精确率 2/3
        golds = [[], [], [], ["product"]]
        preds = [[], [], ["product"], ["product"]]
        st = subset_accuracy(golds, preds)
        self.assertEqual(st["empty_support"], 3)
        self.assertAlmostEqual(st["empty_precision"], 200 / 3)

    def test_no_empty_support_gives_100(self):
        st = subset_accuracy([["product"]], [["product"]])
        self.assertEqual(st["empty_support"], 0)
        self.assertEqual(st["empty_precision"], 100.0)

    def test_illegal_key_excluded(self):
        golds = [["product"], ["product"]]
        preds = [["product"], ["hack_key"]]
        st = subset_accuracy(golds, preds, VALID_RISK)
        self.assertEqual(st["n_skipped_illegal"], 1)
        self.assertAlmostEqual(st["subset_accuracy"], 100.0)

    def test_per_key_diagnostic_line(self):
        golds = [["product"], ["service"]]
        preds = [["product"], ["service"]]
        st = subset_accuracy(golds, preds)
        self.assertEqual(st["per_key"]["product"]["f1"], 100.0)
        self.assertEqual(st["per_key"]["finance"]["support"], 0)


class TestPaFpPair(unittest.TestCase):
    def test_false_alarm_denominator_is_pred_negative(self):
        # 金标 2 负 2 正；预测 2 负（1 对 1 错）→ 误报率 = 1/2 = 50%，不是 1/(1+1)
        pair = pa_fp_pair(["negative", "negative", "neutral", "positive"],
                          ["negative", "neutral", "negative", "positive"])
        self.assertEqual(pair["negative_recall"], 50.0)
        self.assertEqual(pair["false_alarm_rate"], 50.0)
        self.assertEqual(pair["negative_precision"], 50.0)

    def test_recall_and_false_alarm_always_paired(self):
        pair = pa_fp_pair(["negative"], ["negative"])
        self.assertIn("negative_recall", pair)
        self.assertIn("false_alarm_rate", pair)
        self.assertEqual(pair["negative_recall"], 100.0)
        self.assertEqual(pair["false_alarm_rate"], 0.0)

    def test_no_predicted_negative(self):
        pair = pa_fp_pair(["negative", "positive"], ["neutral", "positive"])
        self.assertEqual(pair["negative_recall"], 0.0)
        self.assertEqual(pair["n_pred_negative"], 0)
        # 一条负面都没报 → 没有预警可误 → 误报率空真为 0，且必须打标「不可判定分母」
        self.assertEqual(pair["false_alarm_rate"], 0.0)
        self.assertFalse(pair["false_alarm_defined"])
        self.assertTrue(pair["recall_defined"])

    def test_no_negative_support(self):
        pair = pa_fp_pair(["positive", "neutral"], ["positive", "negative"])
        self.assertFalse(pair["recall_defined"])
        self.assertTrue(pair["false_alarm_defined"])
        self.assertEqual(pair["false_alarm_rate"], 100.0)

    def test_counts(self):
        pair = pa_fp_pair(["negative", "negative", "positive", "positive"],
                          ["negative", "positive", "negative", "positive"])
        self.assertEqual((pair["tp"], pair["fp"], pair["fn"], pair["tn"]), (1, 1, 1, 1))


class TestIllegalLabelCounts(unittest.TestCase):
    def test_counts_only_out_of_set(self):
        preds = [{"sentiment": "hate"}, {"sentiment": "negative"}, {"sentiment": "hate"}]
        cnt = illegal_label_counts(preds, "sentiment", VALID_SENTIMENT)
        self.assertEqual(cnt, {"hate": 2})

    def test_list_field_counts_items(self):
        preds = [{"risk_categories": ["product", "bogus"]}, {"risk_categories": ["bogus"]}]
        cnt = illegal_label_counts(preds, "risk_categories", VALID_RISK)
        self.assertEqual(cnt, {"bogus": 2})

    def test_none_preds_ignored(self):
        self.assertEqual(illegal_label_counts([None, {}], "sentiment", VALID_SENTIMENT), {})


class TestConfusionSlices(unittest.TestCase):
    def test_bucket_counts_and_small_bucket_flag(self):
        dims = {"hc": ["irony", "irony", "explicit", "explicit", "explicit",
                       "explicit", "explicit"]}
        preds = [{"sentiment": "negative", "hard_case_type": "irony",
                  "risk_categories": []} for _ in range(7)]
        gold = {"sentiment": ["negative"] * 7, "hard_case_type": ["irony"] * 7,
                "risk_categories": [[] for _ in range(7)]}
        out = confusion_slices(dims, preds, ["hc"], gold)
        self.assertEqual(out["hc"]["irony"]["n"], 2)
        self.assertTrue(out["hc"]["irony"]["insufficient"])   # n < MIN_BUCKET_N
        self.assertFalse(out["hc"]["explicit"]["insufficient"])
        self.assertEqual(out["hc"]["explicit"]["risk_exact"], 100.0)

    def test_without_gold_only_reports_n(self):
        out = confusion_slices({"d": ["a", "a"]}, [{}, {}], ["d"])
        self.assertEqual(out["d"]["a"]["n"], 2)
        self.assertNotIn("risk_exact", out["d"]["a"])

    def test_min_bucket_constant(self):
        self.assertEqual(MIN_BUCKET_N, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
