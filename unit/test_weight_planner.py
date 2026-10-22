#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字段级 Loss 权重 Span 规划器 · 纯逻辑单测 (零外部依赖, 不需要 torch/transformers)

运行:
    python3 -m unittest unit.test_weight_planner -v
    或在 MVP/tests 目录: python3 -m unittest unit.test_weight_planner -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import sft_weight_planner as swp  # noqa: E402


PROMPT = ("<|im_start|>system\n你是舆情负面精判模型。目标主体：【广发证券】。<|im_end|>\n"
          "<|im_start|>user\n今天去广发证券体验了一把所谓的专业理财，呵呵。<|im_end|>\n"
          "<|im_start|>assistant\n")

COT = "Step 1: 文本含反讽。Step 2: 主体为核心。Step 3: 判负。"

FULL_JSON = ('{"subject": "广发证券", "relevance": "core", "sentiment": "negative", '
             '"hard_case_type": "irony", "risk_categories": ["product", "marketing"], '
             '"evidence": "才一天就亏了3000多"}')

EMPTY_JSON = ('{"subject": "广发证券", "relevance": "core", "sentiment": "positive", '
              '"hard_case_type": "explicit", "risk_categories": [], '
              '"evidence": "整体体验良好"}')


def output(cot=COT, js=FULL_JSON):
    return f"<|begin_of_thought|>{cot}<|end_of_thought|>{js}"


def plan_for(text, stage):
    """按指定 POC 阶段重算权重向量（模块级常量在 import 时读环境变量）。"""
    old = os.environ.get("SFT_POC_STAGE")
    os.environ["SFT_POC_STAGE"] = stage
    try:
        for name in list(sys.modules):
            if "sft_weight_planner" in name:
                del sys.modules[name]
        import sft_weight_planner as fresh
        return fresh.plan_token_weights(PROMPT, text, encode=fresh.char_level_encode)
    finally:
        if old is None:
            os.environ.pop("SFT_POC_STAGE", None)
        else:
            os.environ["SFT_POC_STAGE"] = old
        for name in list(sys.modules):
            if "sft_weight_planner" in name:
                del sys.modules[name]
        import sft_weight_planner as fresh  # noqa: F401  恢复默认 import


def weight_of_value(plan, field, js_text):
    """取某个字段「值」区间上的权重集合。"""
    plen = len(PROMPT)
    js = swp.find_json_start(js_text)
    jt = js_text[js:]
    import re
    m = re.search(r'"%s"\s*:\s*"([^"]*)"' % field, jt)
    if not m:
        return None
    a, b = plen + js + m.start(1), plen + js + m.end(1)
    return set(plan["weights"][a:b])


class TestPromptMasking(unittest.TestCase):
    def test_prompt_always_zero_weight(self):
        for stage in ("POC1", "POC2", "POC3"):
            plan = plan_for(output(), stage)
            w = plan["weights"]
            self.assertTrue(all(x == 0.0 for x in w[:plan["prompt_len"]]),
                            f"{stage}: prompt 段权重应全为 0")

    def test_cot_always_kept(self):
        """红线: CoT 绝不遮 —— 三阶段下 CoT 段权重必须恒为 1.0。"""
        for stage in ("POC1", "POC2", "POC3"):
            plan = plan_for(output(), stage)
            cot_spans = [s for s in plan["spans"] if s[2] == "cot"]
            self.assertEqual(len(cot_spans), 1, f"{stage}: 应恰有 1 个 CoT span")
            t_s, t_e = cot_spans[0][0], cot_spans[0][1]
            self.assertEqual(set(plan["weights"][t_s:t_e]), {1.0},
                             f"{stage}: CoT 段权重必须恒为 1.0")


class TestFieldWeights(unittest.TestCase):
    def test_poc1_all_output_equal_weight(self):
        plan = plan_for(output(), "POC1")
        w = plan["weights"]
        self.assertTrue(all(x in (0.0, 1.0) for x in w))
        self.assertAlmostEqual(sum(w), float(sum(1 for x in w if x > 0)), places=6)

    def test_poc3_field_weights(self):
        plan = plan_for(output(), "POC3")
        w = plan["weights"]
        for field, expect in (("hard_case_type", 4.0), ("sentiment", 3.0),
                               ("relevance", 3.0), ("evidence", 0.4)):
            self.assertEqual(weight_of_value(plan, field, output()), {expect},
                             f"POC3: {field} 值权重应为 {expect}")

    def test_poc2_field_values_backfilled_to_one(self):
        plan = plan_for(output(), "POC2")
        for field in ("hard_case_type", "sentiment", "relevance", "evidence"):
            self.assertEqual(weight_of_value(plan, field, output()), {1.0},
                             f"POC2: {field} 值权重应回补为 1.0")

    def test_key_names_stay_zero_under_skeleton_mask(self):
        """键名属骨架: POC2/POC3 下必须为 0; POC1 下为 1.0。"""
        for stage, expect in (("POC1", 1.0), ("POC2", 0.0), ("POC3", 0.0)):
            plan = plan_for(output(), stage)
            w = plan["weights"]
            plen = len(PROMPT)
            js = swp.find_json_start(output())
            jt = output()[js:]
            import re
            m = re.search(r'"hard_case_type"', jt)
            a, b = plen + js + m.start(), plen + js + m.end()
            self.assertEqual(set(w[a:b]), {expect},
                             f"{stage}: hard_case_type 键名权重应为 {expect}")


class TestSubjectField(unittest.TestCase):
    def test_subject_not_weighted_under_skeleton_mask(self):
        """subject 是输入回显: 未登记 → POC2/POC3 下权重恒 0 (不监督)。"""
        for stage in ("POC2", "POC3"):
            plan = plan_for(output(), stage)
            self.assertEqual(weight_of_value(plan, "subject", output()), {0.0},
                             f"{stage}: subject 回显不应被监督")


class TestEmptyRiskCategories(unittest.TestCase):
    """空表 [] 必须留监督 —— 否则模型学不会「该空就空」, 会硬编 Key。"""

    def test_empty_brackets_get_supervision(self):
        for stage, expect in (("POC1", 1.0), ("POC2", 1.0), ("POC3", 2.0)):
            plan = plan_for(output(js=EMPTY_JSON), stage)
            w = plan["weights"]
            plen = len(PROMPT)
            js = swp.find_json_start(output(js=EMPTY_JSON))
            jt = output(js=EMPTY_JSON)[js:]
            import re
            m = re.search(r'"risk_categories"\s*:\s*(\[[^\]]*\])', jt)
            a, b = plen + js + m.start(1), plen + js + m.end(1)
            self.assertEqual(set(w[a:b]), {expect},
                             f"{stage}: 空表 [] 括号权重应为 {expect}")

    def test_empty_table_sample_is_not_zero_gradient(self):
        """空表样本在 POC2/POC3 下仍须有非零有效 token (分母不得坍缩)。"""
        for stage in ("POC2", "POC3"):
            plan = plan_for(output(js=EMPTY_JSON), stage)
            self.assertGreater(sum(1 for x in plan["weights"] if x > 0), 0,
                               f"{stage}: 空表样本不得零梯度")


class TestDenominatorAndBounds(unittest.TestCase):
    def test_weights_length_matches_full_len(self):
        plan = plan_for(output(), "POC3")
        self.assertEqual(len(plan["weights"]), plan["full_len"])

    def test_truncation_clamps(self):
        plan = swp.plan_token_weights(PROMPT, output(), encode=swp.char_level_encode,
                                       max_len=200)
        self.assertEqual(len(plan["weights"]), plan["full_len"])
        self.assertLessEqual(plan["full_len"], 200)
        self.assertTrue(plan["truncated"])


class TestInvalidJson(unittest.TestCase):
    def test_broken_json_keeps_default_weight(self):
        """C3: JSON 解析失败 → 整段保留 1.0, 不遮不塌。"""
        broken = "<|begin_of_thought|>cot<|end_of_thought|>{\"subject\": \"x\", "
        plan = plan_for(broken, "POC2")
        w = plan["weights"]
        self.assertTrue(all(x in (0.0, 1.0) for x in w[plan["prompt_len"]:]))
        self.assertGreater(sum(1 for x in w[plan["prompt_len"]:] if x > 0), 0)


if __name__ == "__main__":
    unittest.main()
