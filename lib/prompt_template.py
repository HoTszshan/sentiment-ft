#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT Prompt 外壳层 · 唯一真理源

只干一件事：把 instruction / input / output 三元组套进 Qwen 的 ChatML 模板，
拼出模型真正看到的文本。

  render_prompt(tokenizer, instruction, input)            → 问题那一半（推理也喂这个）
  render_full(tokenizer, instruction, input, output)      → 问题+答案（训练喂这个）
  assert_parity(tokenizer, rows)                          → 批量校验拼写无漂移

为什么优先 tokenizer.apply_chat_template：
  模板是「模型出厂协议」，该由 tokenizer 说了算；手写字符串把 ChatML 细节
  硬编码进我们的代码，换基座时无人提醒。但 blindly 信任它也不行 ——
  所以每次渲染都验一条字节等式：

      full == prompt + output + "<|im_end|>"

  等式一破（典型：thinking 版 Qwen3 会在 assistant 段插空思维块），
  训练序列与推理前缀就分叉了，当场报错，而不是训完才发现格式崩了。

不依赖 torch / transformers：tokenizer 按鸭子类型注入（有 chat_template 属性 +
apply_chat_template 方法即可），无 GPU 机器也能跑单测与预览。
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

# 历史手写模板：parity 的字节基准 + tokenizer 无模板时的兜底。
# 与 Qwen2.5 / Qwen3 ChatML 逐字节相同（已验证）。
LEGACY_PROMPT_TMPL = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

ASSISTANT_SUFFIX = "<|im_end|>"


class TemplateParityError(RuntimeError):
    """渲染结果与手写基准不一致 —— 继续跑会让已落盘数据与指标全部作废。"""


def _legacy_prompt(instruction: str, input_text: str) -> str:
    return LEGACY_PROMPT_TMPL.format(instruction=instruction, input=input_text)


def has_chat_template(tokenizer: Any) -> bool:
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, dict):
        return len(tpl) > 0
    return bool(tpl)


def _chat(tokenizer: Any, messages: List[Dict[str, str]],
          add_generation_prompt: bool) -> str:
    out = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if not isinstance(out, str):
        raise TypeError(
            f"apply_chat_template(tokenize=False) 应返回 str，实际 {type(out).__name__}")
    if not add_generation_prompt:
        # 完整序列：模板在 <|im_end|> 后补的换行是「轮次分隔」，不是监督目标；
        # 剥掉它 full == prompt + output + <|im_end|> 的等式才成立。
        # 生成前缀路径不剥 —— 前缀必须以 <|im_start|>assistant\n 结尾。
        out = out[:-1] if out.endswith("\n") else out
    return out


def _messages(instruction: str, input_text: str,
              output: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": instruction},
            {"role": "user", "content": input_text}]
    if output is not None:
        msgs.append({"role": "assistant", "content": output})
    return msgs


def render_prompt(tokenizer: Any, instruction: str, input_text: str) -> str:
    """渲染「生成前缀」：到 <|im_start|>assistant\\n 为止。

    训练侧的前缀掩码区、推理侧的模型输入，两侧必须用本函数同一来源。
    """
    if tokenizer is not None and has_chat_template(tokenizer):
        return _chat(tokenizer, _messages(instruction, input_text),
                     add_generation_prompt=True)
    return _legacy_prompt(instruction, input_text)


def render_full(tokenizer: Any, instruction: str, input_text: str,
                output: str) -> Tuple[str, str]:
    """渲染「完整训练序列」，返回 (prompt_text, full_text)。

    契约：full_text 逐字节等于 prompt_text + output + "<|im_end|>"。
    chat 渲染结果若不满足该等式，直接抛 TemplateParityError。

    ⚠️ 禁用 thinking 版基座（如 Qwen3-4B-Thinking），必须用 Instruct 版：
       thinking 版的模板会强制在 assistant 段开头插一个「空思维块」
       <think>\\n\\n</think>\\n\\n。我们的 output 自带有内容的
       <think>推理…</think>，叠上去训练序列变成「空壳 + 一份推理」双层嵌套，
       模型学会先吐空壳再吐正文；推理时 schema_contract 按 </think> 切 JSON，
       切到的是空壳或残片，JSON 合法率 / evidence 命中全部报废。
       Instruct 版模板对 assistant 内容原样透传，无此问题。
       本函数的等式检查正是为此：开训前就拦下，而不是训完看指标全零再猜。
    """
    prompt_text = render_prompt(tokenizer, instruction, input_text)
    expected = prompt_text + output + ASSISTANT_SUFFIX

    if tokenizer is not None and has_chat_template(tokenizer):
        rendered = _chat(tokenizer, _messages(instruction, input_text, output),
                         add_generation_prompt=False)
        if rendered != expected:
            raise TemplateParityError(
                "chat 模板渲染与手写拼接不一致，训练序列会与推理前缀分叉。\n"
                f"  期望长度 {len(expected)}，实际长度 {len(rendered)}\n"
                "最常见原因：基座是 thinking 版 Qwen3，模板强制在 assistant 段插空思维块\n"
                "<think>\\n\\n</think>，与 output 自带的 <think>推理</think> 叠成双层，\n"
                "推理时解析必炸。**禁用 thinking 版基座，换 Instruct 版**（如 -2507）。"
            )
    return prompt_text, expected


def assert_parity(tokenizer: Any, rows: Iterable[Dict[str, Any]],
                  limit: int = 32) -> Dict[str, Any]:
    """批量校验：抽 limit 条，逐条比 chat 渲染与手写基准。

    无 chat 模板时无事可校验，如实返回 N/A 而非假装 PASS。
    不一致抛 TemplateParityError（供 Dataset 初始化 / --check_parity 调用）。
    """
    if tokenizer is None or not has_chat_template(tokenizer):
        return {"ok": True, "checked": 0,
                "note": "无 chat 模板可比对，parity 不适用"}

    rows = list(rows)[:limit]
    mismatches = []
    for idx, row in enumerate(rows):
        ins = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        out = row.get("output", "") or ""
        want = _legacy_prompt(ins, inp) + out + ASSISTANT_SUFFIX
        try:
            _, got = render_full(tokenizer, ins, inp, out)
        except TemplateParityError as e:
            # render_full 的详细报错（含 thinking 版成因）别吞掉，直接上抛
            raise TemplateParityError(
                f"样本 #{idx}：{e}") from e
        if got != want:
            mismatches.append(idx)
    if mismatches:
        raise TemplateParityError(
            f"chat 模板与手写基准不一致（{len(mismatches)}/{len(rows)} 条），已中止：\n"
            f"  样本下标: {mismatches}")
    return {"ok": True, "checked": len(rows)}
