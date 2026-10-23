#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据集装配层 · 纯逻辑单测 (零外部依赖)

验证三条装配不变量:
  1. 长度不变量: len(input_ids) == len(labels) == len(loss_weights)
  2. 掩码不变量: labels[:prompt_len] 全为 -100，且 loss_weights[:prompt_len] 全为 0
  3. 权重不变量: 字段 span 上的权重等于 FIELD_W 配置（POC3）/ 1.0（POC1/2 的 Value 回补）

另验证装配层与 prompt_template / sft_weight_planner 同源（不自己拼字符串）。

运行:
    cd MVP/tests && python3 -m unittest unit.test_sft_dataset -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import prompt_template as pt      # noqa: E402
import sft_dataset as sd          # noqa: E402
import sft_weight_planner as swp  # noqa: E402
from test_prompt_template import StubChatMLTokenizer, StubNoTemplateTokenizer  # noqa: E402


TOKENIZER_STUB = StubChatMLTokenizer()


def row(instruction="SYS", input_text="【目标主体】广发证券\n【输入文本】\n亏了3000。",
        output=None):
    if output is None:
        output = ('<think>Step 1: 反讽。</think>'
                  '{"subject": "广发证券", "relevance": "core", "sentiment": "negative", '
                  '"hard_case_type": "irony", "risk_categories": ["product"], '
                  '"evidence": "亏了3000"}')
    return {"instruction": instruction, "input": input_text, "output": output}


class TmpJsonl:
    """临时 JSONL 文件上下文（回读路径要真落盘才测得到）。"""

    def __init__(self, rows):
        self.rows = rows
        self.path = None

    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return self.path

    def __exit__(self, *exc):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


# ============================================================
# 1. 长度 / 掩码不变量
# ============================================================

class TestInvariants(unittest.TestCase):

    def setUp(self):
        self.ds = sd.SFTConversationDataset.__new__(sd.SFTConversationDataset)
        self.ds.tokenizer = TOKENIZER_STUB
        self.ds.max_len = 2560
        self.ds.uses_chat = True
        self.ds.strict_parity = True
        self.ds.samples = [row()]
        self.ds.parity = None
        # 桩 tokenizer 的 encode 是字符级（1 字符 = 1 token），与 char_level_encode 同构
        self.ds._encode = lambda t: [ord(c) for c in t]

    def test_lengths_aligned(self):
        item = self.ds.build_item(row())
        n = len(item["input_ids"])
        self.assertEqual(len(item["labels"]), n)
        self.assertEqual(len(item["loss_weights"]), n)

    def test_prefix_is_masked(self):
        item = self.ds.build_item(row())
        p = item["prompt_len"]
        self.assertGreater(p, 0)
        self.assertTrue(all(l == sd.IGNORE_INDEX for l in item["labels"][:p]))
        self.assertTrue(all(w == 0.0 for w in item["loss_weights"][:p]))

    def test_output_side_has_supervision(self):
        item = self.ds.build_item(row())
        p = item["prompt_len"]
        supervised = [l for l in item["labels"][p:] if l != sd.IGNORE_INDEX]
        self.assertGreater(len(supervised), 0)

    def test_prompt_len_matches_rendered_prompt(self):
        r = row()
        item = self.ds.build_item(r)
        want = len(self.ds._encode(pt.LEGACY_PROMPT_TMPL.format(instruction=r["instruction"], input=r["input"])))
        self.assertEqual(item["prompt_len"], want)

    def test_trailing_im_end_keeps_default_weight(self):
        """<|im_end|> 不被任何 span 覆盖 → 必须拿 default 权重，否则学不会停笔。"""
        item = self.ds.build_item(row())
        last = item["loss_weights"][-1]
        self.assertEqual(last, swp.FIELD_WEIGHTS["default"])

    def test_truncation_keeps_lengths(self):
        self.ds.max_len = 30
        item = self.ds.build_item(row())
        self.assertTrue(item["truncated"])
        self.assertLessEqual(len(item["input_ids"]), 30)
        self.assertEqual(len(item["labels"]), len(item["input_ids"]))


# ============================================================
# 2. 与上游模块同源（不重复拼模板）
# ============================================================

class TestSameSource(unittest.TestCase):

    def test_prompt_is_prompt_template_output(self):
        ds = sd.SFTConversationDataset.__new__(sd.SFTConversationDataset)
        ds.tokenizer = TOKENIZER_STUB
        ds.max_len = 2560
        ds.uses_chat = True
        ds.strict_parity = True
        ds._encode = lambda t: [ord(c) for c in t]
        r = row()
        item = ds.build_item(r)
        self.assertEqual(
            item["prompt_text"],
            pt.render_prompt(TOKENIZER_STUB, r["instruction"], r["input"]),
        )

    def test_full_text_is_concatenation(self):
        ds = sd.SFTConversationDataset.__new__(sd.SFTConversationDataset)
        ds.tokenizer = TOKENIZER_STUB
        ds.max_len = 2560
        ds.uses_chat = True
        ds.strict_parity = True
        ds._encode = lambda t: [ord(c) for c in t]
        r = row()
        item = ds.build_item(r)
        self.assertEqual(item["full_text"],
                         item["prompt_text"] + r["output"] + "<|im_end|>")

    def test_no_template_matches_handwritten(self):
        ds = sd.SFTConversationDataset.__new__(sd.SFTConversationDataset)
        ds.tokenizer = None
        ds.max_len = 2560
        ds.uses_chat = False
        ds.strict_parity = True
        ds._encode = lambda t: [ord(c) for c in t]
        r = row()
        item = ds.build_item(r)
        self.assertEqual(item["prompt_text"],
                         pt.LEGACY_PROMPT_TMPL.format(instruction=r["instruction"], input=r["input"]))


# ============================================================
# 3. 权重口径（随 POC 阶段切换）
# ============================================================

class TestWeights(unittest.TestCase):

    def _item(self):
        ds = sd.SFTConversationDataset.__new__(sd.SFTConversationDataset)
        ds.tokenizer = None          # 字符级，span→token 映射为恒等
        ds.max_len = 10000
        ds.uses_chat = False
        ds.strict_parity = True
        ds._encode = swp.char_level_encode
        return ds.build_item(row())

    def test_spans_present(self):
        item = self._item()
        keys = {s[2] for s in item["spans"]}
        self.assertIn("cot", keys)
        self.assertIn("sentiment", keys)

    def test_field_weight_values_by_stage(self):
        """POC3 才提权；POC1/POC2 的 Value 一律回补 1.0。"""
        item = self._item()
        by_key = {s[2]: s[3] for s in item["spans"]}
        if swp.USE_FIELD_WEIGHTS:
            self.assertEqual(by_key["hard_case_type"], swp.FIELD_WEIGHTS["hard_case_type"])
            self.assertEqual(by_key["sentiment"], swp.FIELD_WEIGHTS["sentiment"])
            self.assertEqual(by_key["evidence"], swp.FIELD_WEIGHTS["evidence"])
        else:
            for k in ("hard_case_type", "sentiment", "evidence"):
                self.assertEqual(by_key[k], swp.FIELD_WEIGHTS["default"])

    def test_cot_never_masked(self):
        """红线: CoT 区间的权重恒为 1.0（任何 POC 阶段都不许遮）。"""
        item = self._item()
        cot = [s for s in item["spans"] if s[2] == "cot"]
        self.assertTrue(cot)
        for _, _, _, w in cot:
            self.assertEqual(w, swp.FIELD_WEIGHTS["cot"])


# ============================================================
# 4. 文件加载 / parity gate / 统计
# ============================================================

class TestDatasetIO(unittest.TestCase):

    def test_load_and_len(self):
        with TmpJsonl([row(), row(input_text="另一条")]) as p:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            self.assertEqual(len(ds), 2)

    def test_getitem_returns_three_keys(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            item = ds[0]
            self.assertEqual(set(item), {"input_ids", "labels", "loss_weights"})

    def test_blank_lines_skipped(self):
        fd, p = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(row(), ensure_ascii=False) + "\n")
                f.write("\n   \n")
                f.write(json.dumps(row(), ensure_ascii=False) + "\n")
            self.assertEqual(len(sd.SFTConversationDataset(p, TOKENIZER_STUB)), 2)
        finally:
            os.unlink(p)

    def test_parity_runs_on_init(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            self.assertIsNotNone(ds.parity)
            self.assertTrue(ds.parity["ok"])

    def test_parity_gate_blocks_drift(self):
        """thinking 版模板 → 初始化就该炸，而不是训完才发现。"""
        with TmpJsonl([row()]) as p:
            with self.assertRaises(pt.TemplateParityError):
                sd.SFTConversationDataset(
                    p, StubChatMLTokenizer(thinking_variant=True), max_len=2560)

    def test_parity_can_be_disabled(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(
                p, StubChatMLTokenizer(thinking_variant=True),
                max_len=2560, strict_parity=False)
            self.assertIsNone(ds.parity)

    def test_no_template_tokenizer_falls_back(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(p, StubNoTemplateTokenizer(), max_len=2560)
            self.assertFalse(ds.uses_chat)

    def test_stats_shape(self):
        with TmpJsonl([row(), row(input_text="短")]) as p:
            st = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560).stats()
            self.assertEqual(st["n"], 2)
            self.assertIn("p95", st["total"])
            self.assertGreaterEqual(st["total"]["max"], st["total"]["p50"])

    def test_empty_file(self):
        fd, p = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            self.assertEqual(len(ds), 0)
            self.assertEqual(ds.stats()["n"], 0)
        finally:
            os.unlink(p)


# ============================================================
# 5. 预览
# ============================================================

class TestPreview(unittest.TestCase):

    def test_preview_contains_four_sections(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            text = ds.preview(0, max_tokens=80)
            for marker in ("① 原始字段", "② messages", "③ 模板渲染后完整文本", "④ token 视图"):
                self.assertIn(marker, text)

    def test_preview_shows_template_tokens(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            text = ds.preview(0, max_tokens=20)
            self.assertIn("<|im_start|>system", text)
            self.assertIn("<|im_start|>assistant", text)

    def test_preview_shows_weights_row(self):
        with TmpJsonl([row()]) as p:
            ds = sd.SFTConversationDataset(p, TOKENIZER_STUB, max_len=2560)
            self.assertIn("weight:", ds.preview(0, max_tokens=20))

    def test_preview_file_multi(self):
        with TmpJsonl([row(), row(input_text="第二条")]) as p:
            text = sd.preview_file(p, TOKENIZER_STUB, start=0, n=2, max_len=2560)
            self.assertIn("样本 #0", text)
            self.assertIn("样本 #1", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
