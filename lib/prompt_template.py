788#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT Prompt 外壳层 · 唯一真理源

把 instruction / input / output 套进 Qwen ChatML 模板，拼出模型真正看到的文本。

  render_prompt(tokenizer, instruction, input)       → 生成前缀（训练掩码区 = 推理输入）
  render_full(tokenizer, instruction, input, output) → 完整训练序列
  assert_parity(tokenizer, rows)                     → 批量校验 chat 渲染无漂移

优先用 tokenizer.apply_chat_template（模板由模型出厂决定），但每次渲染都验字节等式
`full == prompt + output + "<|im_end|>"`。等式一破说明训练序列与推理前缀分叉，
当场报错，而不是训完看指标全零再猜。

不依赖 torch / transformers：tokenizer 按鸭子类型注入，无 GPU 也能跑单测。
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

# 与 Qwen2.5 / Qwen3 ChatML 逐字节相同。用作 parity 基准 + 无 chat 模板时兜底。
LEGACY_PROMPT_TMPL = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

ASSISTANT_SUFFIX = "<|im_end|>"

# ---- 前缀预置（think prefix seeding）----
# 开启后把 "<think>\n" 写进 assistant 前缀：训练侧落在掩码区（不计 loss），
# 推理侧落在生成前缀（不占解码预算）。模型不必学"第一个 token 吐什么"。
#
# 默认关闭。实验三组对照已证明：解冻 <think>/</think>之后，
# 模型 40 步内能自己学会吐 <think>，本开关无额外收益；且开启后模型
# 依赖推理端复现同一前缀，adapter 不再自包含。保留开关仅用于复现该对照。
THINK_PREFIX = "<think>\n"
USE_THINK_PREFIX: bool = False


def set_think_prefix(enabled: bool) -> bool:
    """设置前缀预置开关。必须在构造 Dataset 之前调用，训练与推理须取同一值。"""
    global USE_THINK_PREFIX
    USE_THINK_PREFIX = bool(enabled)
    return USE_THINK_PREFIX


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
        # 训练完整样本时：<|im_end|>后自动追加的换行仅作轮次分隔，不属于监督目标，需要剔除；
        # 生成前缀分支（add_generation_prompt=True）保留末尾换行，维持chat模板协议。
        out = out[:-1] if out.endswith("\n") else out
    return out


def _messages(instruction: str, input_text: str,
              output: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": instruction},
            {"role": "user", "content": input_text}]
    if output is not None:
        msgs.append({"role": "assistant", "content": output})
    return msgs


def render_prompt(tokenizer: Any, instruction: str, input_text: str,
                  few_shot: Optional[List[Dict[str, str]]] = None,
                  few_shot_mode: str = "full") -> str:
    """渲染 生成前缀（训练掩码区 = 推理输入），止于 `<|im_start|>assistant\n`。训练与推理必须同源调用。

    few_shot（可选，T2 提示词加固专用）：每条 {"instruction","input","output"}。
    仅推理侧传参使用；训练侧任何调用不得传入（保持与训练模板同源）。

    few_shot_mode 控制示例的注入形态（few-shot 示例含完整 CoT 时 token 膨胀严重，
    实测 3 条 full 注入 prompt 达 4000+ token，故提供紧凑档）：
      "full"    —— 每条示例完整三段（system 指令 → user 输入 → assistant 金标全量），
                   3 条约 +3350 token；最贴近训练分布，格式示范最完整。
      "compact" —— 指令只注入一次（首条 system），示例折叠为
                   「user: 目标主体+原文 → assistant: 仅金标 JSON（去 CoT）」，
                   3 条约 +450 token；格式示范保留（JSON 形态可见），CoT 示范放弃。
    """
    if tokenizer is not None and has_chat_template(tokenizer):
        msgs: List[Dict[str, str]] = []
        if few_shot:
            if few_shot_mode == "compact":
                # 指令只出现一次；示例不带 CoT（JSON 保留），防 token 膨胀
                for k, ex in enumerate(few_shot):
                    if k == 0:
                        msgs.extend(_messages(ex.get("instruction", "") or "",
                                              ex.get("input", "") or "",
                                              _strip_cot(ex.get("output", "") or "")))
                    else:
                        msgs.append({"role": "user", "content": ex.get("input", "") or ""})
                        msgs.append({"role": "assistant",
                                     "content": _strip_cot(ex.get("output", "") or "")})
            else:  # full
                for ex in few_shot:
                    msgs.extend(_messages(ex.get("instruction", "") or "",
                                          ex.get("input", "") or "",
                                          _assistant_content(ex.get("output", "") or "")))
        msgs.extend(_messages(instruction, input_text))
        prompt = _chat(tokenizer, msgs,
                       add_generation_prompt=True)
    else:
        prompt = _legacy_prompt(instruction, input_text)
    if USE_THINK_PREFIX:
        prompt += THINK_PREFIX
    return prompt


def _strip_cot(output: str) -> str:
    """去掉 output 里的 <think>…</think> CoT 段，只留 JSON（few-shot compact 模式用）。"""
    import re
    m = re.search(r"</think>\s*(.*)$", output, re.DOTALL)
    return m.group(1).strip() if m else output


def _strip_think_prefix(output: str) -> str:
    """裁掉 output 开头的 "<think>\n"（已由前缀预置提供，不能重复出现）。"""
    if USE_THINK_PREFIX and output.startswith(THINK_PREFIX):
        return output[len(THINK_PREFIX):]
    return output


def _assistant_content(output_body: str) -> str:
    """assistant 段内容：开启预置时补回前缀，供 chat 渲染与手写基准对齐。"""
    return (THINK_PREFIX + output_body) if USE_THINK_PREFIX else output_body


def render_full(tokenizer: Any, instruction: str, input_text: str,
                output: str) -> Tuple[str, str]:
    """渲染完整训练序列，返回 (prompt_text, full_text)。

    契约：`full_text == prompt_text + output_body + "<|im_end|>"`，chat 渲染不符即抛错。

    ⚠️ 必须用 Instruct 版基座。thinking 版（如 Qwen3-4B-Thinking）会在 assistant 段
    强制插空思维块 `<think>\n\n</think>\n\n`，与本项目 output 自带的 CoT 叠成双层：
    模型学会先吐空壳，推理时按 `</think>` 切 JSON 切到残片，格式指标全废。
    """
    output_body = _strip_think_prefix(output)
    prompt_text = render_prompt(tokenizer, instruction, input_text)
    expected = prompt_text + output_body + ASSISTANT_SUFFIX

    if tokenizer is not None and has_chat_template(tokenizer):
        rendered = _chat(
            tokenizer,
            _messages(instruction, input_text, _assistant_content(output_body)),
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
    """抽 limit 条，逐条比对 chat 渲染与手写基准；不一致抛 TemplateParityError。

    无 chat 模板时如实返回 N/A，不假装 PASS。
    """
    if tokenizer is None or not has_chat_template(tokenizer):
        return {"ok": True, "checked": 0,
                "note": "无 chat 模板可比对，parity 不适用"}

    rows = list(rows)[:limit]
    mismatches = []
    for idx, row in enumerate(rows):
        ins = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        out = _strip_think_prefix(row.get("output", "") or "")
        want = _legacy_prompt(ins, inp) + _assistant_content(out) + ASSISTANT_SUFFIX
        try:
            _, got = render_full(tokenizer, ins, inp, out)
        except TemplateParityError as e:
            raise TemplateParityError(f"样本 #{idx}：{e}") from e
        if got != want:
            mismatches.append(idx)
    if mismatches:
        raise TemplateParityError(
            f"chat 模板与手写基准不一致（{len(mismatches)}/{len(rows)} 条），已中止：\n"
            f"  样本下标: {mismatches}")
    return {"ok": True, "checked": len(rows)}
