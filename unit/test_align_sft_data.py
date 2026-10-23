#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式对齐器 · 纯逻辑单测 (零外部依赖)

覆盖: 模板协议段与 output 形态一致性 / gold 映射 / 闭集校验 /
      evidence 子串判定 / 剔除计数 / meta 透传

运行:
    python3 -m unittest unit.test_align_sft_data -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import sft_template                    # noqa: E402
import schema_contract as sc           # noqa: E402
from corpus_schema import INDUSTRY_SCHEMAS, schema_to_inline  # noqa: E402
import align_sft_data as aligner       # noqa: E402


# ---- 构造一条全合规的源数据 ----

def make_row(**over):
    text = "真的服了，才一天就亏了3000多，广发证券这理财也太坑了。"
    row = {
        "event_id": "t-001",
        "industry": "finance_insurance",
        "hard_case_type": "irony",
        "sentiment": "negative",
        "relevance": "core",
        "risk_keys": ["finance", "marketing"],
        "target_subject": "广发证券",
        "subject_hit": True,
        "provenance": "synthetic",
        "input": text,
        "evidence_span": "才一天就亏了3000多",
    }
    row.update(over)
    return row


GOOD_ROW = make_row()


class TestGoldMapping(unittest.TestCase):
    def test_gold_keys_follow_required_fields(self):
        gold = aligner.gold_of(GOOD_ROW)
        self.assertEqual(list(gold.keys()), list(sc.REQUIRED_FIELDS))

    def test_risk_keys_to_risk_categories(self):
        gold = aligner.gold_of(GOOD_ROW)
        self.assertEqual(gold["risk_categories"], ["finance", "marketing"])
        self.assertEqual(gold["subject"], "广发证券")
        self.assertEqual(gold["evidence"], "才一天就亏了3000多")

    def test_gold_passes_validate_schema(self):
        self.assertIsNone(sc.validate_schema(aligner.gold_of(GOOD_ROW)))


class TestTemplate(unittest.TestCase):
    """新模板（用户定稿）：主体/原文在 user 侧，system 侧 = 角色+任务+行业体系+输出协议。"""

    def test_cot_protocol_always_described(self):
        """新模板固定含两步协议（SFT 数据均带 CoT），且枚举自契约插值。"""
        instr = sft_template.build_instruction("finance_insurance")
        self.assertIn("<think>", instr)
        self.assertIn("Step 1/Step 2/Step 3", instr)
        self.assertIn("严禁使用 ```json", instr)
        for v in sc.VALID_SENTIMENT:
            self.assertIn(f'"{v}"', instr)
        for v in sc.VALID_RELEVANCE:
            self.assertIn(f'"{v}"', instr)
        for v in sc.VALID_HARD_CASE:
            self.assertIn(f'"{v}"', instr)

    def test_priority_order_written(self):
        """hard_case 多命中时的降序优先级必须写进协议。"""
        instr = sft_template.build_instruction("automotive")
        self.assertIn("irony > metaphor > multi_subject > comparative > "
                      "counterfactual > fact_implicit > explicit", instr)

    def test_evidence_rule_no_empty_string(self):
        """evidence 必须 4~80 字且禁止空串（契约 EVIDENCE_MIN/MAX 插值）。"""
        instr = sft_template.build_instruction("fmcg")
        self.assertIn(f"长度 {sc.EVIDENCE_MIN_LEN} 到 {sc.EVIDENCE_MAX_LEN} 字", instr)
        self.assertIn("不得填空串", instr)

    def test_industry_risk_system_rendered(self):
        """行业风险体系按 `大类en(大类cn): 词表` 渲染。"""
        instr = sft_template.build_instruction("finance_insurance")
        self.assertIn("金融保险(finance_insurance)", instr)
        self.assertIn("- product (产品/质量): 理财暴雷", instr)

    def test_instruction_is_subject_invariant(self):
        """同行业不同主体的 instruction 逐字一致（主体在 user 侧）。"""
        self.assertEqual(sft_template.build_instruction("automotive"),
                         sft_template.build_instruction("automotive"))

    def test_user_input_subject_line(self):
        """user 侧 = 【目标主体】行 + 【输入文本】；subject 正则可反解。"""
        import re
        ui = sft_template.build_user_input("广发证券", "才一天就亏了3000多。")
        m = re.search(sft_template.SUBJECT_RE, ui)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).strip(), "广发证券")
        self.assertIn("【输入文本】", ui)


class TestCheckRow(unittest.TestCase):
    def test_good_row_passes(self):
        self.assertIsNone(aligner.check_row(GOOD_ROW))

    def test_evidence_not_substring_dropped(self):
        row = make_row(evidence_span="亏了五千多")
        self.assertIn("连续子串", aligner.check_row(row))

    def test_subject_hit_false_dropped(self):
        row = make_row(subject_hit=False)
        self.assertIn("subject_hit", aligner.check_row(row))

    def test_hard_case_out_of_closed_set(self):
        row = make_row(hard_case_type="sarcasm")
        self.assertIn("hard_case_type 越界", aligner.check_row(row))

    def test_risk_out_of_closed_set(self):
        row = make_row(risk_keys=["politics"])
        self.assertIn("risk_keys 越界", aligner.check_row(row))

    def test_empty_input_dropped(self):
        row = make_row(input="  ")
        self.assertIn("input 为空", aligner.check_row(row))

    def test_overlong_input_dropped(self):
        row = make_row(input="长" * 2001, evidence_span=None)  # ev 校验在长度之后也会拦
        self.assertIsNotNone(aligner.check_row(row))


class TestCotInheritance(unittest.TestCase):
    """CoT 只继承不生成：源行自带 <think> 则逐字沿用，否则只出裸 JSON。"""

    def test_source_with_cot_is_preserved_verbatim(self):
        src = '<think>\n原始推理\n</think>{"subject":"广发证券"}'
        row = aligner.flatten(make_row(output=src))
        out, has = aligner.render_output(row, aligner.gold_of(row))
        self.assertTrue(has)
        self.assertEqual(out, src)          # 逐字沿用，不做任何格式改写

    def test_think_tail_newline_is_user_format_kept(self):
        """</think> 后带换行是用户定稿格式，继承时原样保留。"""
        src = '<think>x</think>\n{"subject":"广发证券"}'
        row = aligner.flatten(make_row(output=src))
        out, _ = aligner.render_output(row, aligner.gold_of(row))
        self.assertEqual(out, src)

    def test_source_without_cot_emits_bare_json(self):
        row = aligner.flatten(make_row())
        out, has = aligner.render_output(row, aligner.gold_of(row))
        self.assertFalse(has)
        self.assertFalse(out.startswith("<think>"))
        json.loads(out)


if __name__ == "__main__":
    unittest.main()
