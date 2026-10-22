#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 Rubric 三判定 · 单元测试（零外部依赖）

运行:
    python3 -m unittest unit.test_rubric_check -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from rubric_check import (  # noqa: E402
    _check_contradiction, rubric_check,
)

INPUT_TEXT = "今天去广发证券体验了一把所谓的专业理财，呵呵，真的是“稳赚不赔”啊！才一天就亏了3000多。"
EVIDENCE = "才一天就亏了3000多"


def _pred(**kw):
    body = {"subject": "广发证券", "relevance": "core", "sentiment": "negative",
            "hard_case_type": "irony", "risk_categories": ["product", "marketing"],
            "evidence": EVIDENCE}
    body.update(kw)
    return body


GOOD_COT = ("Step1 上游锚定主体为广发证券，语篇地位判 core。"
            "Step2 字面褒真实贬判 irony。Step3 亏损归 product，判 negative。")


class TestSubjectNamed(unittest.TestCase):
    def test_subject_in_cot_passes(self):
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT)
        self.assertTrue(r["items"]["subject_named"])

    def test_subject_missing_in_cot_fails(self):
        r = rubric_check("Step1 判 core。Step2 判 irony。Step3 判 negative。",
                         _pred(), INPUT_TEXT)
        self.assertFalse(r["items"]["subject_named"])
        self.assertFalse(r["pass"])
        self.assertIn("主体", r["reason"])

    def test_no_subject_no_cot_gives_none(self):
        """缺输入 → 不可判定（None），不误杀。"""
        r = rubric_check("", {"sentiment": "negative"}, INPUT_TEXT)
        self.assertIsNone(r["items"]["subject_named"])


class TestNoContradiction(unittest.TestCase):
    def test_consistent_cot_passes(self):
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT)
        self.assertTrue(r["items"]["no_contradiction"])

    def test_contradictory_sentiment_fails(self):
        cot = "Step1 主体广发证券。Step2 判 positive。Step3 判负面。"
        r = rubric_check(cot, _pred(sentiment="negative"), INPUT_TEXT)
        self.assertFalse(r["items"]["no_contradiction"])
        self.assertFalse(r["pass"])

    def test_contradictory_hard_case_fails(self):
        cot = "Step1 主体广发证券。Step2 判 metaphor。"
        r = rubric_check(cot, _pred(hard_case_type="irony"), INPUT_TEXT)
        self.assertFalse(r["items"]["no_contradiction"])

    def test_fact_implicit_not_confused_with_explicit(self):
        """子串陷阱回归：CoT 写 fact_implicit，标签也是 fact_implicit → 不得误判矛盾。"""
        cot = "Step1 主体广发证券。Step2 隐含事实判 fact_implicit。"
        r = rubric_check(cot, _pred(hard_case_type="fact_implicit"), INPUT_TEXT)
        self.assertTrue(r["items"]["no_contradiction"])

    def test_cot_without_enum_gives_none(self):
        """CoT 没提任何极性/难点枚举 → 不可判定，不误杀。"""
        r = rubric_check("Step1 主体广发证券。Step2 综合判断。Step3 结论给出。",
                         _pred(), INPUT_TEXT)
        self.assertIsNone(r["items"]["no_contradiction"])


class TestEvidenceSupports(unittest.TestCase):
    def test_substring_evidence_passes(self):
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT)
        self.assertTrue(r["items"]["evidence_supports"])

    def test_fabricated_evidence_fails(self):
        r = rubric_check(GOOD_COT, _pred(evidence="这句话是我编的原文里根本没有"),
                         INPUT_TEXT)
        self.assertFalse(r["items"]["evidence_supports"])
        self.assertFalse(r["pass"])

    def test_missing_evidence_fails(self):
        r = rubric_check(GOOD_COT, _pred(evidence=""), INPUT_TEXT)
        self.assertFalse(r["items"]["evidence_supports"])

    def test_no_input_gives_none(self):
        r = rubric_check(GOOD_COT, _pred(), "")
        self.assertIsNone(r["items"]["evidence_supports"])


class TestJudgeInjection(unittest.TestCase):
    def test_judge_pass_keeps_result(self):
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT,
                         judge=lambda p: "PASS")
        self.assertTrue(r["pass"])
        self.assertTrue(r["judge_verdict"])

    def test_judge_fail_overrides(self):
        """离线裁判判不通过 → 即使规则子集全过也 FAIL。"""
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT,
                         judge=lambda p: "FAIL: 推理跳步")
        self.assertFalse(r["pass"])
        self.assertFalse(r["judge_verdict"])

    def test_judge_exception_does_not_crash(self):
        def boom(_):
            raise RuntimeError("judge down")
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT, judge=boom)
        self.assertFalse(r["pass"])
        self.assertIn("judge 调用失败", r["reason"])

    def test_judge_garbage_output(self):
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT, judge=lambda p: "不知道")
        self.assertIsNone(r["judge_verdict"])
        self.assertIn("未给出可解析结论", r["reason"])


class TestOverallVerdict(unittest.TestCase):
    def test_all_pass(self):
        r = rubric_check(GOOD_COT, _pred(), INPUT_TEXT)
        self.assertTrue(r["pass"])
        # 无裁判时第③项只跑机械前提，reason 里要点名「语义项跳过」
        self.assertIn("语义项跳过", r["reason"])

    def test_nothing_judgeable_is_fail_not_pass(self):
        """三项全不可判定 → pass=False（不因「测不了」而放行）。"""
        r = rubric_check("", {}, "")
        self.assertFalse(r["pass"])


class TestContradictionHelper(unittest.TestCase):
    def test_none_when_no_enum(self):
        self.assertIsNone(_check_contradiction("纯中文推理没有枚举词。",
                                               _pred(), {"negative"}, {"irony"}))

    def test_none_when_empty_cot(self):
        self.assertIsNone(_check_contradiction("", _pred(), {"negative"}, {"irony"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
