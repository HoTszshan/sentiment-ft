#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt 外壳层 · 纯逻辑单测 (零外部依赖，不需要 torch/transformers)

用桩 tokenizer 复刻 Qwen3 ChatML 模板，验证:
  1. chat 路径渲染结果 == 历史手写模板（parity，本模块存在的唯一理由）
  2. full == prompt + output + <|im_end|>（训练序列与推理前缀不分叉）
  3. thinking 版模板（插入空思维块）会被 parity 拦下，而不是静默漂移
  4. 无 chat_template 时自动退手写外壳

运行:
    cd MVP/tests && python3 -m unittest unit.test_prompt_template -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import prompt_template as pt  # noqa: E402


# ============================================================
# 桩 tokenizer
# ============================================================

class StubChatMLTokenizer:
    """复刻 Qwen2.5 / Qwen3 (Instruct, 非 thinking) 的 ChatML 模板。

    真实模板语义:
      {% for m in messages %}<|im_start|>{{ m.role }}\\n{{ m.content }}<|im_end|>\\n{% endfor %}
      {% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}
    """

    def __init__(self, thinking_variant: bool = False):
        self.chat_template = "stub-chatml"
        self.thinking_variant = thinking_variant
        self.pad_token_id = 0

    # ---- tokenize 侧（字符级桩：1 字符 = 1 token）----
    # 与 sft_weight_planner.char_level_encode 同构，让 span→token 映射是恒等的，
    # 单测里就能直接按字符下标断言权重落在哪个字段上。

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [chr(int(i)) if 32 <= int(i) < 0x3000 else f"<{int(i)}>" for i in ids]

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=False, **kwargs):
        assert tokenize is False, "本模块只走 tokenize=False 的字符串路径"
        parts = []
        for m in messages:
            content = m["content"]
            if self.thinking_variant and m["role"] == "assistant":
                # Qwen3 thinking 版：assistant 段强制包一层（无内容时为空思维块）
                content = "<think>\n\n</think>\n\n" + content
            parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class StubNoTemplateTokenizer:
    """没有 chat_template 的 tokenizer（老基座 / 自定义模型）。"""

    chat_template = None

    def apply_chat_template(self, *a, **k):  # pragma: no cover - 不应被调用
        raise AssertionError("无 chat_template 时不该走 chat 路径")


ROWS = [
    {
        "instruction": "你是舆情负面精判模型。目标主体：【广发证券】。",
        "input": "今天去广发证券体验了一把所谓的专业理财，呵呵。",
        "output": '<think>Step 1: 反讽。</think>{"sentiment": "negative"}',
    },
    {
        "instruction": "你是舆情分析专家。\n【行业风险体系：汽车(automotive)】\n- product (产品/质量): 刹车失灵",
        "input": "【目标主体】蔚来 ET5\n【输入文本】\n提车三天就异响，售后还推诿。",
        "output": '<think>\nStep 1: 核心主体。\n</think>\n{"sentiment": "negative"}',
    },
]


# ============================================================
# 1. parity：chat 渲染 == 历史手写模板
# ============================================================

class TestParity(unittest.TestCase):

    def setUp(self):
        self.tok = StubChatMLTokenizer()

    def test_prompt_bytes_identical_to_legacy(self):
        """这是整个模块的核心契约：chat 渲染必须逐字节等于手写模板。"""
        for row in ROWS:
            with self.subTest(instruction=row["instruction"][:20]):
                got = pt.render_prompt(self.tok, row["instruction"], row["input"])
                want = pt.LEGACY_PROMPT_TMPL.format(
                    instruction=row["instruction"], input=row["input"])
                self.assertEqual(got, want)

    def test_prompt_ends_with_assistant_prefix(self):
        got = pt.render_prompt(self.tok, "S", "U")
        self.assertTrue(got.endswith("<|im_start|>assistant\n"))
        self.assertIn("<|im_start|>system\nS<|im_end|>\n", got)
        self.assertIn("<|im_start|>user\nU<|im_end|>\n", got)

    def test_full_equals_prompt_plus_output_plus_im_end(self):
        for row in ROWS:
            prompt, full = pt.render_full(
                self.tok, row["instruction"], row["input"], row["output"])
            self.assertEqual(full, prompt + row["output"] + "<|im_end|>")

    def test_full_strips_exactly_one_trailing_newline(self):
        """模板在 <|im_end|> 后补的轮次换行必须剥掉，否则每条样本多学一个 \\n。"""
        _, full = pt.render_full(self.tok, "S", "U", "OUT")
        self.assertTrue(full.endswith("<|im_end|>"))
        self.assertFalse(full.endswith("\n"))

    def test_no_strip_on_content(self):
        """内容层的空白原样保留 —— 任何隐式 strip 都会改变字节。"""
        _, full = pt.render_full(self.tok, " S ", " U ", " O ")
        self.assertIn("<|im_start|>system\n S <|im_end|>", full)
        self.assertIn("<|im_start|>user\n U <|im_end|>", full)


# ============================================================
# 2. 漂移必须被拦下
# ============================================================

class TestDriftDetection(unittest.TestCase):

    def test_thinking_variant_raises(self):
        """thinking 版模板会插入空思维块 → full ≠ prompt+output+im_end，必须报错。"""
        tok = StubChatMLTokenizer(thinking_variant=True)
        with self.assertRaises(pt.TemplateParityError):
            pt.render_full(tok, "S", "U", "OUT")

    def test_thinking_variant_prompt_still_ok(self):
        """生成前缀不受 thinking 影响（差异只在 assistant 段之后），别误伤。"""
        tok = StubChatMLTokenizer(thinking_variant=True)
        self.assertEqual(
            pt.render_prompt(tok, "S", "U"),
            pt.LEGACY_PROMPT_TMPL.format(instruction="S", input="U"),
        )

    def test_assert_parity_flags_thinking_variant(self):
        tok = StubChatMLTokenizer(thinking_variant=True)
        with self.assertRaises(pt.TemplateParityError) as ctx:
            pt.assert_parity(tok, ROWS, limit=1)
        self.assertIn("不一致", str(ctx.exception))

    def test_assert_parity_passes_on_pure_chatml(self):
        rep = pt.assert_parity(StubChatMLTokenizer(), ROWS, limit=2)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["checked"], 2)


# ============================================================
# 3. 无模板降级
# ============================================================

class TestFallback(unittest.TestCase):

    def test_no_template_uses_handwritten(self):
        got = pt.render_prompt(StubNoTemplateTokenizer(), "S", "U")
        self.assertEqual(got, pt.LEGACY_PROMPT_TMPL.format(instruction="S", input="U"))
        self.assertEqual(got, "<|im_start|>system\nS<|im_end|>\n"
                              "<|im_start|>user\nU<|im_end|>\n"
                              "<|im_start|>assistant\n")

    def test_none_tokenizer_uses_handwritten(self):
        self.assertEqual(
            pt.render_prompt(None, "S", "U"),
            pt.LEGACY_PROMPT_TMPL.format(instruction="S", input="U"))

    def test_assert_parity_marks_not_applicable(self):
        """没有 chat 模板时，如实标 N/A —— 不假装 PASS。"""
        rep = pt.assert_parity(StubNoTemplateTokenizer(), ROWS)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["checked"], 0)
        self.assertIn("不适用", rep["note"])

    def test_dict_chat_template_detected(self):
        class MultiTemplate:
            chat_template = {"default": "...", "tool_use": "..."}
        self.assertTrue(pt.has_chat_template(MultiTemplate()))

    def test_empty_dict_template_not_detected(self):
        class EmptyTemplate:
            chat_template = {}
        self.assertFalse(pt.has_chat_template(EmptyTemplate()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
