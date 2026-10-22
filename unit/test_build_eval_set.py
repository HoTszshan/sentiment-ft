#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金标评测集构建器 · 纯逻辑单测 (不联网、不打 API)

运行:
    python3 -m unittest unit.test_build_eval_set -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from build_eval_set import (  # noqa: E402
    agreement_report, classify, cohen_kappa, parse_annotation,
    pair_agreement, risk_key, verdict,
)


def _a(**kw):
    d = {"subject": "广发证券", "relevance": "core", "sentiment": "negative",
         "hard_case_type": "irony", "risk_categories": ["product", "marketing"],
         "evidence": "才一天就亏了3000多"}
    d.update(kw)
    return d


class TestParseAnnotation(unittest.TestCase):
    def test_coT_wrapped_json(self):
        raw = ("<think>Step1 判 core。Step2 判 irony。Step3 判负。</think>"
               + json.dumps(_a(), ensure_ascii=False))
        self.assertIsNotNone(parse_annotation(raw))

    def test_bare_json(self):
        self.assertIsNotNone(parse_annotation(json.dumps(_a(), ensure_ascii=False)))

    def test_out_of_set_enum_rejected(self):
        raw = json.dumps(_a(sentiment="hate"), ensure_ascii=False)
        self.assertIsNone(parse_annotation(raw))

    def test_garbage_rejected(self):
        self.assertIsNone(parse_annotation("完全不是标注的乱话"))


class TestClassify(unittest.TestCase):
    def test_all_agree_is_gold(self):
        self.assertEqual(classify((_a(), _a(), _a())), "gold")

    def test_only_hard_case_differs_is_boundary(self):
        b, c = _a(), _a()
        b["hard_case_type"] = "explicit"
        c["hard_case_type"] = "metaphor"
        self.assertEqual(classify((_a(), b, c)), "boundary")

    def test_sentiment_differs_is_drop(self):
        b = _a(sentiment="positive")
        self.assertEqual(classify((_a(), b, _a())), "drop")

    def test_risk_order_insensitive_still_gold(self):
        b = _a(risk_categories=["marketing", "product"])   # 顺序不同 = 同一集合
        self.assertEqual(classify((_a(), b, _a())), "gold")

    def test_risk_set_differs_is_drop(self):
        b = _a(risk_categories=["product"])
        self.assertEqual(classify((_a(), b, _a())), "drop")

    def test_relevance_differs_is_drop(self):
        b = _a(relevance="passing")
        self.assertEqual(classify((_a(), b, _a())), "drop")

    def test_missing_third_party_is_drop(self):
        self.assertEqual(classify((_a(), None, _a())), "drop")

    def test_risk_key_handles_non_list(self):
        self.assertEqual(risk_key("product"), ())
        self.assertEqual(risk_key(["b", "a"]), ("a", "b"))


class TestKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        # 两类交替出现，保证 p_e < 1，κ 才有定义；两边完全一致 → κ = 1.0
        pairs = [((("negative", "positive")[i % 2]), (("negative", "positive")[i % 2]))
                 for i in range(10)]
        self.assertAlmostEqual(cohen_kappa(pairs), 1.0, places=4)

    def test_pair_agreement_rate(self):
        pairs = [("n", "n"), ("n", "p"), ("n", "n"), ("n", "n")]
        ok, n, rate = pair_agreement(pairs)
        self.assertEqual((ok, n), (3, 4))
        self.assertAlmostEqual(rate, 0.75)

    def test_all_identical_labels_kappa_none(self):
        # 全票同一取值 → p_e=1，κ 无定义 (返回 None 而非 0)
        self.assertIsNone(cohen_kappa([("n", "n")] * 10))

    def test_too_few_samples_kappa_none(self):
        self.assertIsNone(cohen_kappa([("n", "n"), ("n", "p")]))

    def test_low_agreement_low_kappa(self):
        pairs = [("a", "b"), ("b", "a"), ("a", "b"), ("b", "a"),
                 ("c", "c"), ("c", "c"), ("a", "a"), ("b", "b")]
        k = cohen_kappa(pairs)
        self.assertIsNotNone(k)
        self.assertLess(k, 0.7)


class TestReportAndVerdict(unittest.TestCase):
    @staticmethod
    def _rows(k):
        rows = []
        for i in range(k):
            g = _a(sentiment="negative" if i % 2 == 0 else "positive")
            rows.append({"gold": g, "A-DeepSeek": g, "B": g, "C": g})
        return rows

    def test_all_agree_report(self):
        rep = agreement_report(self._rows(8))
        self.assertEqual(rep["n"], 8)
        rate = rep["sentiment"]["A-DeepSeek/B"]["rate"]
        self.assertEqual(rate, 1.0)

    def test_verdict_pass(self):
        rep = agreement_report(self._rows(8))
        v = verdict(rep)
        self.assertIn("PASS", v)
        self.assertIn("≥ 0.7", v)

    def test_verdict_insufficient_samples(self):
        rep = agreement_report(self._rows(3))
        self.assertIn("样本量不足", verdict(rep))

    def test_verdict_fail_when_kappa_low(self):
        rows = []
        for i in range(12):
            g = _a(sentiment="negative")
            b = dict(g, sentiment="positive" if i % 2 == 0 else "negative")
            rows.append({"gold": g, "A-DeepSeek": g, "B": b, "C": g})
        rep = agreement_report(rows)
        v = verdict(rep)
        self.assertTrue(v.startswith("FAIL") or v.startswith("BOUNDARY"), v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
