#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证据支撑性公共判定 · 单元测试（评估手册 TODO 5.4「共调」项）

覆盖:
  - mechanical_precheck: 子串 / 长度窗 / 缺 evidence / 原文缺失
  - parse_support_reply: JSON 路径 / 自由文本兜底 / 无法解析 → 判不支撑
  - judge_evidence_support: 注入裁判 / 裁判异常 / 未注入裁判
  - 与 rubric_check 第③项口径一致（同一实现，非两份）

运行:
    python3 -m unittest unit.test_evidence_support -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from evidence_support import (  # noqa: E402
    build_support_prompt, judge_evidence_support, mechanical_precheck,
    parse_support_reply,
)
from rubric_check import rubric_check  # noqa: E402

TEXT = "今天去广发证券体验了一把所谓的专业理财，呵呵，真的是“稳赚不赔”啊！才一天就亏了3000多。"
EV = "才一天就亏了3000多"
PRED = {"subject": "广发证券", "relevance": "core", "sentiment": "negative",
        "hard_case_type": "irony", "risk_categories": ["product"],
        "evidence": EV}


class TestMechanicalPrecheck(unittest.TestCase):
    def test_substring_passes(self):
        ok, why = mechanical_precheck(EV, TEXT)
        self.assertTrue(ok)
        self.assertEqual(why, "")

    def test_fabricated_fails(self):
        ok, why = mechanical_precheck("这句话原文里根本没有", TEXT)
        self.assertFalse(ok)
        self.assertIn("子串", why)

    def test_missing_evidence_fails(self):
        ok, why = mechanical_precheck("", TEXT)
        self.assertFalse(ok)
        self.assertIn("缺 evidence", why)

    def test_too_short_fails(self):
        ok, why = mechanical_precheck("亏了", TEXT)
        self.assertFalse(ok)
        self.assertIn("长度", why)

    def test_too_long_fails(self):
        ok, why = mechanical_precheck("字" * 100, TEXT)
        self.assertFalse(ok)
        self.assertIn("长度", why)

    def test_no_input_text_skips(self):
        """原文缺失 → 机械前提放行（跳过），交由上层决定。"""
        ok, why = mechanical_precheck(EV, "")
        self.assertTrue(ok)
        self.assertIn("跳过", why)


class TestParseSupportReply(unittest.TestCase):
    def test_json_yes(self):
        ok, why = parse_support_reply('{"support":"yes","reason":"证据明确"}')
        self.assertTrue(ok)
        self.assertIn("证据明确", why)

    def test_json_no(self):
        ok, why = parse_support_reply('{"support":"no","reason":"撑不住"}')
        self.assertFalse(ok)
        self.assertIn("撑不住", why)

    def test_free_text_pass(self):
        ok, _ = parse_support_reply("PASS")
        self.assertTrue(ok)

    def test_free_text_fail(self):
        ok, _ = parse_support_reply("FAIL: 推理跳步")
        self.assertFalse(ok)

    def test_unparseable_is_not_support(self):
        """不可判定 ≠ 放行。"""
        ok, why = parse_support_reply("不知道")
        self.assertFalse(ok)
        self.assertIn("无法解析", why)

    def test_empty_is_not_support(self):
        ok, _ = parse_support_reply("")
        self.assertFalse(ok)


class TestJudgeEvidenceSupport(unittest.TestCase):
    def test_injected_judge_yes(self):
        ok, _ = judge_evidence_support(lambda p: '{"support":"yes"}',
                                        TEXT, "广发证券", PRED)
        self.assertTrue(ok)

    def test_injected_judge_no(self):
        ok, _ = judge_evidence_support(lambda p: '{"support":"no"}',
                                        TEXT, "广发证券", PRED)
        self.assertFalse(ok)

    def test_judge_exception_is_not_support(self):
        def boom(_):
            raise RuntimeError("judge down")
        ok, why = judge_evidence_support(boom, TEXT, "广发证券", PRED)
        self.assertFalse(ok)
        self.assertIn("异常", why)

    def test_no_judge_skips_semantic(self):
        ok, why = judge_evidence_support(None, TEXT, "广发证券", PRED)
        self.assertTrue(ok)
        self.assertIn("跳过", why)

    def test_prompt_contains_key_fields(self):
        p = build_support_prompt(TEXT, "广发证券", PRED)
        self.assertIn("广发证券", p)
        self.assertIn(EV, p)
        self.assertIn("negative", p)


class TestSharedWithRubric(unittest.TestCase):
    """Rubric 第③项与查④ 必须走同一份实现（口径漂移防护）。"""

    def test_rubric_uses_same_judgement(self):
        """同一裁判回复，Rubric 第③项与公共函数结论必须一致。"""
        cot = "Step1 主体广发证券。Step2 判 irony。Step3 判 negative。"
        r = rubric_check(cot=cot, pred=PRED, input_text=TEXT,
                         judge=lambda p: '{"support":"yes","reason":"ok"}')
        ok, _ = judge_evidence_support(lambda p: '{"support":"yes","reason":"ok"}',
                                       TEXT, PRED["subject"], PRED)
        self.assertEqual(r["items"]["evidence_supports"], ok)

    def test_rubric_rejects_when_judge_says_no(self):
        cot = "Step1 主体广发证券。Step2 判 irony。Step3 判 negative。"
        r = rubric_check(cot=cot, pred=PRED, input_text=TEXT,
                         judge=lambda p: '{"support":"no","reason":"撑不住"}')
        self.assertFalse(r["items"]["evidence_supports"])
        self.assertFalse(r["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
