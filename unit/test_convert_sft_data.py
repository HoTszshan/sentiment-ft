#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema v1 批次转换器 · 纯逻辑单测 (零外部依赖)

覆盖: CoT 标签替换不碰其他字符 / instruction 同行业逐字一致 /
      meta 白名单 / output 与 gold 冲突剔除 / evidence 子串质检 / 剔除计数

运行:
    python3 -m unittest unit.test_convert_sft_data -v
"""

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import schema_contract as sc           # noqa: E402
import task_template as tt             # noqa: E402
import convert_sft_data as cg           # noqa: E402


def make_row(**over):
    text = "这理财才一天就亏了3000多，真是稳赚不赔啊。"
    row = {
        "event_id": "s3C-fin-irony-core-n-001",
        "target_subject": "广发证券",
        "industry": "finance_insurance",
        "input": text,
        "output": ("<|begin_of_thought|>\nStep 1: 反讽。\n<|end_of_thought|>"
                   '{"subject": "广发证券", "relevance": "core", "sentiment": "negative", '
                   '"hard_case_type": "irony", "risk_categories": ["finance"], '
                   '"evidence": "才一天就亏了3000多"}'),
        "gold": {"subject": "广发证券", "sentiment": "negative", "relevance": "core",
                 "hard_case_type": "irony", "risk_categories": ["finance"],
                 "evidence": "才一天就亏了3000多"},
        "meta": {"anno_mode": "x", "arbiter": "y", "diff_fields": [],
                 "majority_hard_case_type": "irony"},
        "provenance": "synthetic",
    }
    row.update(over)
    return row


class TestConvertOutput(unittest.TestCase):
    def test_tag_replacement_only(self):
        """除两个 CoT 标签外，其余字符逐字保留（含换行）。"""
        raw = "<|begin_of_thought|>\n推理\n<|end_of_thought|>\n{\"a\":1}"
        got = cg.convert_output(raw)
        self.assertEqual(got, "<think>\n推理\n</think>\n{\"a\":1}")

    def test_parseable_after_convert(self):
        row = make_row()
        out = cg.convert_output(row["output"])
        cot, pred, flags = sc.parse_output(out)
        self.assertIsNotNone(pred)
        self.assertIsNone(flags["schema_error"])
        self.assertEqual(pred, row["gold"])


class TestConvertRow(unittest.TestCase):
    def test_happy_path(self):
        row, reason = cg.convert_row(make_row())
        self.assertIsNone(reason)
        self.assertEqual(row["event_id"], "s3C-fin-irony-core-n-001")
        self.assertIn("<think>", row["output"])
        self.assertNotIn("<|begin_of_thought|>", row["output"])
        # instruction 与同行业模板逐字一致
        self.assertEqual(row["instruction"], tt.build_instruction("finance_insurance"))
        # user 题面结构
        self.assertTrue(row["input"].startswith("【目标主体】广发证券\n【输入文本】\n"))

    def test_meta_whitelist(self):
        """meta 只留白名单字段，打标过程字段（anno_mode/arbiter/diff_fields）丢弃。"""
        row, _ = cg.convert_row(make_row())
        m = row["meta"]
        self.assertEqual(sorted(m.keys()),
                         ["event_id", "hard_case_type", "industry",
                          "input_raw", "provenance", "relevance",
                          "risk_categories", "sentiment"])
        self.assertNotIn("anno_mode", m)
        self.assertEqual(m["input_raw"], "这理财才一天就亏了3000多，真是稳赚不赔啊。")

    def test_industry_out_of_closed_set(self):
        _, reason = cg.convert_row(make_row(industry="aerospace"))
        self.assertIn("industry 越界", reason)

    def test_gold_schema_violation_dropped(self):
        row = make_row()
        row["gold"]["sentiment"] = "angry"          # 闭集外
        row["output"] = row["output"].replace('"negative"', '"angry"')
        _, reason = cg.convert_row(row)
        self.assertIsNotNone(reason)

    def test_evidence_not_substring_dropped(self):
        row = make_row()
        row["gold"]["evidence"] = "亏了五千多"
        _, reason = cg.convert_row(row)
        self.assertIn("连续子串", reason)

    def test_output_gold_conflict_dropped(self):
        """铁律「output JSON 为准」：与行内 gold 冲突即剔除（不静默取一边）。"""
        row = make_row()
        row["output"] = row["output"].replace('"sentiment": "negative"',
                                              '"sentiment": "positive"')
        _, reason = cg.convert_row(row)
        self.assertIn("冲突: sentiment", reason)

    def test_empty_input_dropped(self):
        _, reason = cg.convert_row(make_row(input="  "))
        self.assertIn("input 为空", reason)


class TestNewIndustries(unittest.TestCase):
    """第三轮 4 新行业：instruction 渲染不 KeyError，行业风险体系段正确。"""

    def test_all_new_industries_render(self):
        for ind, cn in [("healthcare_pharma", "医药健康"), ("food_catering", "餐饮食品"),
                        ("logistics_express", "物流快递"), ("online_edu", "在线教育")]:
            instr = tt.build_instruction(ind)
            self.assertIn(f"{cn}({ind})", instr)
            self.assertIn("- product (产品/质量):", instr)

    def test_round1_style_row_converts(self):
        """带 sft_bucket 的 round1 批次行：转换成功且 bucket 透传。"""
        row = make_row(industry="food_catering",
                       target_subject="霸王茶姬",
                       event_id="v3i-food_catering-metaphor-passing-nega-3")
        row["meta"]["sft_bucket"] = "food_catering|metaphor|negative"
        out, reason = cg.convert_row(row)
        self.assertIsNone(reason)
        self.assertEqual(out["meta"]["sft_bucket"], "food_catering|metaphor|negative")
        self.assertIn("餐饮食品(food_catering)", out["instruction"])

    def test_subject_whitespace_diff_keeps_gold(self):
        """spec 与 gold/output 的 subject 空格差异（如「极氪 001」vs「极氪001」）不剔除：
        spec 只是生成端目标，金标以 output JSON 为准，归一化空白后一致即放行。"""
        row = make_row(target_subject="极氪 001")
        row["gold"]["subject"] = "极氪001"
        row["output"] = row["output"].replace('"subject": "广发证券"', '"subject": "极氪001"')
        out, reason = cg.convert_row(row)
        self.assertIsNone(reason)
        self.assertIn("【目标主体】", out["input"])

    def test_subject_real_conflict_still_dropped(self):
        """归一化后仍不同的 subject 是真冲突，必须剔除（防止误放行）。"""
        row = make_row(target_subject="极氪 001")
        row["gold"]["subject"] = "极氪001"
        row["output"] = row["output"].replace('"subject": "广发证券"', '"subject": "蔚来ET5"')
        _, reason = cg.convert_row(row)
        self.assertIn("冲突: subject", reason)


class TestKeptInvariants(unittest.TestCase):
    def test_resolve_gold_zero_conflict_on_converted(self):
        """转换后的行喂 resolve_gold 必须零冲突（output 为准 + meta 兜底一致）。"""
        row, _ = cg.convert_row(make_row())
        gold, conflicts = sc.resolve_gold(row)
        self.assertEqual(conflicts, [])
        self.assertEqual(gold["subject"], "广发证券")
        self.assertEqual(gold["evidence"], "才一天就亏了3000多")


if __name__ == "__main__":
    unittest.main()
