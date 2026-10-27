#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 四维 Reward 打分器 · 单元测试

运行:
    python3 -m unittest unit.test_reward_fns -v
    或在 MVP/tests 目录: python3 -m unittest unit.test_reward_fns -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from reward_fns import (  # noqa: E402
    R_CORRECT_FULL, R_FALSE_ALARM, R_GATE_SHORT_CIRCUIT, R_MISS_SEVERE,
    assert_contract_invariants, compute_reward, count_cot_tokens,
    group_advantages, r_accuracy, r_format, r_grounding, r_length,
    extract_input_text, split_completion, zero_advantage_masking,
)


# ---------------------------------------------------------------
# 构造样本：一份合格输出、若干种「违规」输出
# ---------------------------------------------------------------

INPUT_TEXT = "今天去广发证券体验了一把所谓的专业理财，呵呵，真的是“稳赚不赔”啊！才一天就亏了3000多。"

GOLD = {
    "subject": "广发证券",
    "relevance": "core",
    "sentiment": "negative",
    "hard_case_type": "irony",
    "risk_categories": ["product", "marketing"],
    "evidence": "才一天就亏了3000多",
}


def _mk(cot: str, **kw) -> str:
    """拼一条格式合格的 completion。"""
    import json
    body = {"subject": "广发证券", "relevance": "core", "sentiment": "negative",
            "hard_case_type": "irony", "risk_categories": ["product", "marketing"],
            "evidence": "才一天就亏了3000多"}
    body.update(kw)
    return (f"<think>{cot}</think>"
            f"{json.dumps(body, ensure_ascii=False)}")


# 合格 CoT (长度落在 120~200 tokens 理想区间: 2026-10-27 按 750 条实测 p50 重设
# 阈值后，理想带 ≈165~275 汉字。下面是自然展开的三跳推理 ≈190 字 ≈138 tokens)
GOOD_COT = (
    "Step1 主体识别：通读全文，叙事焦点始终落在广发证券身上，主体相关度判 core。"
    "Step2 语义分析：字面用「稳赚不赔」的褒义表述，但紧跟「才一天就亏了3000多」的事实反转，"
    "典型反讽修辞，难例类型判 irony。Step3 风险归因：亏损直接指向理财产品表现归 product，"
    "「稳赚不赔」的宣传口径涉嫌误导归 marketing。综合判 negative。"
)
# 短 CoT (≈ 22 tokens，低于 120 门槛 → 过短罚)
SHORT_COT = "Step 1 core；Step 2 irony；Step 3 negative。"
# 超长 CoT (≥320 tokens，衰减到 0)
LONG_COT = "理由如下，先把上下文完整复述一遍。" * 40
# 中段 CoT (≈229 tokens，落在 200~320 衰减带)
MID_COT = "这是一个推理步骤，" * 35


class TestContractInvariants(unittest.TestCase):
    def test_three_invariants_and_monotonicity(self):
        self.assertTrue(assert_contract_invariants())

    def test_short_circuit_stricter_than_business_penalty(self):
        self.assertLess(R_GATE_SHORT_CIRCUIT, R_MISS_SEVERE)
        self.assertEqual(R_MISS_SEVERE, R_FALSE_ALARM * 2)

    def test_gates_never_positive(self):
        # 协议损坏 + 证据伪造 两种情况都必须是 -2.5，绝不能给正分
        self.assertEqual(r_format("没有 CoT 标签的裸 JSON"), R_GATE_SHORT_CIRCUIT)
        self.assertEqual(r_format("x" * 10), R_GATE_SHORT_CIRCUIT)


class TestFormatGate(unittest.TestCase):
    def test_valid_completion_is_zero(self):
        self.assertEqual(r_format(_mk(GOOD_COT)), 0.0)

    def test_missing_thought_tags(self):
        self.assertEqual(r_format('{"sentiment":"negative"}'), R_GATE_SHORT_CIRCUIT)

    def test_broken_json(self):
        self.assertEqual(
            r_format("<think>cot</think>{broken"),
            R_GATE_SHORT_CIRCUIT)

    def test_out_of_set_enum(self):
        # sentiment 越界 → 协议罚，且必须重于危机漏报
        self.assertEqual(r_format(_mk(GOOD_COT, sentiment="hate")), R_GATE_SHORT_CIRCUIT)

    def test_trailing_garbage(self):
        bad = _mk(GOOD_COT) + "  再补一句话"
        self.assertEqual(r_format(bad), R_GATE_SHORT_CIRCUIT)

    def test_nested_object_rejected(self):
        self.assertEqual(
            r_format(_mk(GOOD_COT, sentiment={"v": "negative"})),
            R_GATE_SHORT_CIRCUIT)


class TestGroundingGate(unittest.TestCase):
    def test_evidence_is_substring(self):
        self.assertEqual(r_grounding(_mk(GOOD_COT), INPUT_TEXT), 0.0)

    def test_fabricated_evidence_short_circuits(self):
        bad = _mk(GOOD_COT, evidence="这句话是我编的，原文里根本没有")
        self.assertEqual(r_grounding(bad, INPUT_TEXT), R_GATE_SHORT_CIRCUIT)

    def test_partial_paste_still_fails(self):
        # 改一个标点都算伪造
        bad = _mk(GOOD_COT, evidence="才一天就亏了3000多.")
        self.assertEqual(r_grounding(bad, INPUT_TEXT), R_GATE_SHORT_CIRCUIT)


class TestAccuracyTiering(unittest.TestCase):
    def test_full_hit_with_relevance_capped_at_one(self):
        score, d = r_accuracy(_mk(GOOD_COT), GOLD)
        self.assertEqual(score, R_CORRECT_FULL)   # +1.0 封顶，+0.2 不外溢到 +1.2
        self.assertTrue(d["accuracy_pass"])

    def test_severe_miss_is_minus_two(self):
        # gold 负面 (带正式 6 类风险) 却判成 positive → 严重危机漏报
        score, d = r_accuracy(_mk(GOOD_COT, sentiment="positive"), GOLD)
        self.assertEqual(score, R_MISS_SEVERE)
        self.assertTrue(d["miss_severe"])
        self.assertFalse(d["accuracy_pass"])

    def test_miss_stays_minus_two_even_if_relevance_wrong(self):
        # 危机漏报档不叠加 relevance 轻罚 (否则弱主观字段压过危机漏报)
        score, _ = r_accuracy(_mk(GOOD_COT, sentiment="positive", relevance="passing"), GOLD)
        self.assertEqual(score, R_MISS_SEVERE)

    def test_false_alarm_is_minus_one(self):
        # gold 非负面却判负
        gold = dict(GOLD, sentiment="neutral", risk_categories=[])
        score, d = r_accuracy(_mk(GOOD_COT), gold)
        self.assertEqual(score, R_FALSE_ALARM)
        self.assertTrue(d["false_alarm"])

    def test_wrong_hard_case_gets_partial_credit(self):
        # 2026-10-27 阶梯式放宽: 情感+risk+relevance 对、仅难点类型错 → 0.7
        # (旧口径 -1.0 使 47/50 组 k=0 全零优势，RL 无梯度)
        score, d = r_accuracy(_mk(GOOD_COT, hard_case_type="explicit"), GOLD)
        self.assertAlmostEqual(score, 0.7)
        self.assertFalse(d["accuracy_pass"] is True and d["hard_case_ok"])
        self.assertTrue(d["sentiment_ok"] and d["risk_ok"])

    def test_wrong_risk_gets_sentiment_only_credit(self):
        # 2026-10-27 阶梯式: 情感对、risk 错 → 0.5 (风险标签允许逐步学)
        score, d = r_accuracy(_mk(GOOD_COT, risk_categories=["product"]), GOLD)
        self.assertAlmostEqual(score, 0.5)
        self.assertTrue(d["sentiment_ok"])
        self.assertFalse(d["risk_ok"])

    def test_wrong_base_plus_wrong_relevance_stacks(self):
        # 契约: 判错档再叠 -0.2。阶梯式后 hc 错落在 0.7 档 → 0.7-0.2 = 0.5
        score, _ = r_accuracy(
            _mk(GOOD_COT, hard_case_type="explicit", relevance="passing"), GOLD)
        self.assertAlmostEqual(score, 0.5)

    def test_wrong_relevance_deducts_point_two(self):
        score, _ = r_accuracy(_mk(GOOD_COT, relevance="passing"), GOLD)
        self.assertAlmostEqual(score, R_CORRECT_FULL - 0.2)

    def test_wrong_risk_deducts(self):
        score, _ = r_accuracy(_mk(GOOD_COT, risk_categories=["product"]), GOLD)
        self.assertLess(score, R_CORRECT_FULL)


class TestLengthReward(unittest.TestCase):
    def test_length_zero_when_not_correct(self):
        # 门禁不变量2: 判错 → 长度分硬置 0 (交白卷不能套长度满分)
        self.assertEqual(r_length(_mk(GOOD_COT), accuracy_pass=False), 0.0)

    def test_ideal_interval_full_score(self):
        self.assertEqual(r_length(_mk(GOOD_COT), accuracy_pass=True), 1.0)

    def test_too_short_penalized(self):
        v = r_length(_mk(SHORT_COT), accuracy_pass=True)
        self.assertLess(v, 0.0)
        self.assertGreaterEqual(v, -1.0)

    def test_too_long_decays_to_zero(self):
        self.assertEqual(r_length(_mk(LONG_COT), accuracy_pass=True), 0.0)

    def test_partial_decay_between_65_and_120(self):
        # 65~120 token 衰减带: 有分但不满分
        v = r_length(_mk(MID_COT), accuracy_pass=True)
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1.0)

    def test_token_counter_has_zh_ratio(self):
        # SOP 换算: 40 tokens ≈ 55 汉字
        self.assertEqual(count_cot_tokens("字" * 55), 40)


class TestTotalReward(unittest.TestCase):
    def test_perfect_trajectory_hits_max(self):
        r = compute_reward(_mk(GOOD_COT), INPUT_TEXT, GOLD)
        self.assertEqual(r["total"], 2.0)          # 判对 +1.0 + 理想长度 +1.0
        self.assertLessEqual(r["total"], 2.0)

    def test_short_circuit_wins(self):
        r = compute_reward(_mk(GOOD_COT, evidence="这句话我编的原文里根本没有"), INPUT_TEXT, GOLD)
        self.assertEqual(r["total"], R_GATE_SHORT_CIRCUIT)
        self.assertEqual(r["detail"]["short_circuit"], "grounding")

    def test_total_within_value_range(self):
        cases = [
            (_mk(GOOD_COT), GOLD),
            (_mk(GOOD_COT, sentiment="positive"), GOLD),
            (_mk(GOOD_COT, relevance="passing"), GOLD),
        ]
        for comp, g in cases:
            r = compute_reward(comp, INPUT_TEXT, g)
            self.assertGreaterEqual(r["total"], -2.5)
            self.assertLessEqual(r["total"], 2.0)

    def test_corruption_beats_miss_in_bad_ranking(self):
        """核心防御: 崩格式 (-2.5) 必须比危机漏报 (-2.0) 更差。"""
        corrupt = compute_reward("<think>x</think>{bad",
                                 INPUT_TEXT, GOLD)["total"]
        miss = compute_reward(_mk(GOOD_COT, sentiment="positive"),
                              INPUT_TEXT, GOLD)["total"]
        self.assertLess(corrupt, miss)


class TestGroupAggregation(unittest.TestCase):
    def test_group_advantages_zero_mean(self):
        adv = group_advantages([-1.0, 1.0, 0.0, 2.0])
        self.assertAlmostEqual(sum(adv), 0.0, places=5)

    def test_uniform_group_is_masked(self):
        # 全对 / 全错: 分数相同 → 整组优势清零，不放大任何轨迹
        self.assertEqual(group_advantages([2.0] * 4), [0.0] * 4)

    def test_default_does_not_divide_by_std(self):
        """默认口径：只减均值。优势直接等于偏离量，保留业务惩罚的绝对尺度。"""
        adv = group_advantages([2.0, 1.0, -1.0, -2.0])
        self.assertEqual(adv, [2.0, 1.0, -1.0, -2.0])
        # 减均值后和必须为 0
        self.assertAlmostEqual(sum(adv), 0.0, places=6)

    def test_scale_by_std_is_opt_in_only(self):
        """除 std 仅作对照实验保留；调用方不显式要求就不得生效。"""
        plain = group_advantages([2.0, 1.0, -1.0, -2.0])
        scaled = group_advantages([2.0, 1.0, -1.0, -2.0], scale_by_std=True)
        self.assertNotEqual(plain, scaled)
        self.assertAlmostEqual(scaled[0], 2.0 / math.sqrt(2.5), places=5)   # std = √2.5

    def test_junk_group_must_not_dominate_real_group(self):
        """核心口径: 只差一档的垃圾组，优势幅度必须小于真有判别信息的组。

        除 std 的旧口径下这两组是反的（垃圾组 +1.73 > 真信息组 +1.26），
        等于教模型「学垃圾里最不烂的那条」—— 见 GRPO 手册 §1.2。
        """
        junk = group_advantages([-2.5, -2.5, -2.5, -2.0])
        real = group_advantages([2.0, 1.0, -1.0, -2.0])
        self.assertLess(max(map(abs, junk)), max(map(abs, real)))
        self.assertAlmostEqual(max(map(abs, junk)), 0.375, places=6)

    def test_std_division_would_invert_the_ranking(self):
        """把「为什么不能除 std」钉成可执行断言：文档里的数必须能复现。"""
        junk_scaled = group_advantages([-2.5, -2.5, -2.5, -2.0], scale_by_std=True)
        real_scaled = group_advantages([2.0, 1.0, -1.0, -2.0], scale_by_std=True)
        self.assertAlmostEqual(max(map(abs, junk_scaled)), math.sqrt(3), places=4)   # ≈1.732
        self.assertAlmostEqual(max(map(abs, real_scaled)), 2.0 / math.sqrt(2.5), places=4)  # ≈1.265
        self.assertGreater(max(map(abs, junk_scaled)), max(map(abs, real_scaled)))

    def test_zero_advantage_masking_on_dead_group(self):
        # 全错组 (整组触顶门禁罚) 强制清零
        self.assertEqual(zero_advantage_masking([-2.5] * 4), [0.0] * 4)

    def test_masking_passes_through_live_group(self):
        adv = zero_advantage_masking([-1.0, 1.0, 0.0, 2.0])
        self.assertNotEqual(adv, [0.0] * 4)

    def test_empty_group(self):
        self.assertEqual(group_advantages([]), [])


class TestTwoFieldPrompt(unittest.TestCase):
    """GRPO 落盘只有 prompt + gold：grounding 原文必须能从 prompt 反解出来。"""

    PROMPT = ("<|im_start|>system\n你是舆情负面精判模型。目标主体：【广发证券】<|im_end|>\n"
              "<|im_start|>user\n" + INPUT_TEXT + "<|im_end|>\n<|im_start|>assistant\n")

    def test_extract_input_from_chat_prompt(self):
        self.assertEqual(extract_input_text(self.PROMPT), INPUT_TEXT)

    def test_extract_input_from_bracket_prompt(self):
        prompt = f"【文本】：{INPUT_TEXT}\n【关注主体】：广发证券"
        self.assertEqual(extract_input_text(prompt), INPUT_TEXT)

    def test_grounding_works_on_two_field_contract(self):
        # 只靠 prompt + gold 就能完成 grounding 断言
        inp = extract_input_text(self.PROMPT)
        self.assertEqual(r_grounding(_mk(GOOD_COT), inp), 0.0)
        bad = _mk(GOOD_COT, evidence="我编的一句话原文没有")
        self.assertEqual(r_grounding(bad, inp), R_GATE_SHORT_CIRCUIT)


class TestSplitHelper(unittest.TestCase):
    def test_split_valid(self):
        cot, pred, flags = split_completion(_mk(GOOD_COT))
        self.assertTrue(flags["valid_thought_tags"])
        self.assertTrue(flags["valid_json"])
        self.assertEqual(pred["sentiment"], "negative")
        self.assertTrue(cot.startswith("Step1"))

    def test_split_missing_tags(self):
        _, pred, flags = split_completion("no tags")
        self.assertFalse(flags["valid_thought_tags"])
        self.assertIsNone(pred)


if __name__ == "__main__":
    unittest.main(verbosity=2)
