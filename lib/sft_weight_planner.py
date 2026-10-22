#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情负面精判 · 字段级 Loss 权重 Span 规划器 (SOP §6.2 对齐版)

设计动机（对应 Code Review 发现的两处缺陷）:
  1. 原实现在 token 序列上做「编码后子序列滑窗匹配」，BPE 分词器对引号/数字存在合并歧义，
     易匹配失败 → 字段漏加权。
  2. 原实现未对 evidence 显式降权 0.4x，与 SOP §6.2 不符。

本模块改为「字符级 span → token 索引」的精确映射：
  - 先定位 JSON 值在**字符**上的起止位置（带转义与嵌套括号处理）；
  - 再用 `len(encode(text[:char_pos]))` 把字符位置换算成 token 下标。
该换算与分词算法无关，对 BPE / SentencePiece / char-level 一律精确，杜绝滑窗匹配失效。

本模块**不依赖 torch / transformers**，可独立单测。
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

# ===== POC 阶段开关（《SFT训练实战手册》§1.4 · 方案 A）=====
#   POC1 = 仅前缀掩码：prompt→0，输出侧一律等权 1.0（baseline，消融对照组）
#   POC2 = POC1 + 骨架掩码：JSON 死模板→0，字段 Value 回补 1.0
#   POC3 = POC2 + 字段级加权 FIELD_W：4.0 / 3.0 / 2.0 / 0.4
# 三阶段共用同一份代码，只切换这一个开关，保证消融时「一次只动一个变量」。
POC_STAGE: str = (os.environ.get("SFT_POC_STAGE") or "POC1").strip().upper()
VALID_STAGES = ("POC1", "POC2", "POC3")
if POC_STAGE not in VALID_STAGES:
    raise ValueError(f"[POC] 非法阶段 {POC_STAGE!r}，可选: {' / '.join(VALID_STAGES)}")

USE_SKELETON_MASK: bool = POC_STAGE in ("POC2", "POC3")
USE_FIELD_WEIGHTS: bool = POC_STAGE == "POC3"

# ===== SOP §6.2 字段级损失权重 =====
FIELD_WEIGHTS: Dict[str, float] = {
    "prompt": 0.0,          # 输入侧：系统提示 / Schema / 原文  → 全掩
    "cot": 1.0,             # 思维链：保梯度（红线：CoT 绝不遮）
    "hard_case_type": 4.0,  # 最难学、SFT 赢过小模型的唯一理由
    "sentiment": 3.0,       # 方向承载
    "relevance": 3.0,       # 防误报关键
    "risk_categories": 2.0, # 多标签
    "evidence": 0.4,        # 长文本，降权防稀释离散标签梯度
    "skeleton": 0.0,        # JSON 结构骨架（键名/引号/括号/逗号/冒号）→ 遮蔽
                            #   仅 POC2/POC3 生效（POC1 下不参与，等权由 span_weight 兜回 1.0）；
                            #   理由见 3.2 Q1：基座已有 JSON 先验，再算模板字符属过度学习，
                            #   且会把大头梯度烧在背模板上、稀释领域标签判别。
    "default": 1.0,         # 协议分隔符与尾部（<think> / </think> 等）
                            #   必须留梯度：这些分隔符未必在基座词表里，遮了就再也学不会发
}

# JSON 中需要特殊加权的字段 → 权重键
# 注: `subject` 故意不登记 —— 它是输入 target_subject 的回显、逐字重复、条件熵≈0,
#     属「输入回显」而非判别结论。不登记 → 随骨架掩码置 0 (POC2/POC3), 即主动不监督。
#     这与空表 `[]` 恰好相反: 空表括号必须留监督(否则模型学不会「该空就空」),
#     故在 plan_field_char_spans 里单独登记, 见下方 risk_categories 分支。
JSON_FIELD_TO_WEIGHT_KEY = {
    "hard_case_type": "hard_case_type",
    "sentiment": "sentiment",
    "relevance": "relevance",
    "evidence": "evidence",
}


def span_weight(weight_key: str) -> float:
    """按当前 POC 阶段返回字段区间权重：POC3 用 FIELD_W；POC1/POC2 输出侧等权 1.0。

    POC1：无骨架掩码，输出侧整段本就是 1.0，此处为 no-op。
    POC2：骨架已置 0，字段 Value 靠本函数回补 1.0（键名等骨架仍为 0）。
    POC3：字段 Value 拿各自的 4.0/3.0/2.0/0.4。
    """
    return FIELD_WEIGHTS[weight_key] if USE_FIELD_WEIGHTS else FIELD_WEIGHTS["default"]

THOUGHT_END_TOKEN = "</think>"   # 与 schema_contract.THOUGHT_END 同源 (Qwen3 原生标签)

EscapeAware = Callable[[str], List[int]]


def find_json_start(output_text: str) -> int:
    """
    定位 output 中 6 字段 JSON 的起始字符下标。
    约定输出形态: <think>{CoT}</think>{JSON}
    """
    if THOUGHT_END_TOKEN in output_text:
        idx = output_text.index(THOUGHT_END_TOKEN) + len(THOUGHT_END_TOKEN)
        while idx < len(output_text) and output_text[idx].isspace():
            idx += 1
        return idx
    # 降级：无思考标签时，取第一个 '{'
    brace = output_text.find("{")
    return brace if brace != -1 else len(output_text)


def find_value_span(json_text: str, key: str, search_from: int = 0) -> Optional[Tuple[int, int]]:
    """
    在 json_text 中定位 `"key": <value>` 的 **值** 的字符起止区间 [start, end)。
    支持字符串（含转义）、数组（含嵌套）、裸字面量（数字/bool/null）。
    """
    pattern = re.compile(r'"%s"\s*:\s*' % re.escape(key))
    m = pattern.search(json_text, search_from)
    if not m:
        return None

    start = m.end()
    if start >= len(json_text):
        return None

    ch = json_text[start]

    # 字符串值：处理 \" 转义
    if ch == '"':
        i = start + 1
        while i < len(json_text):
            if json_text[i] == "\\":
                i += 2
                continue
            if json_text[i] == '"':
                return (start, i + 1)
            i += 1
        return None

    # 数组值：括号配对
    if ch == "[":
        depth = 0
        i = start
        while i < len(json_text):
            if json_text[i] == "[":
                depth += 1
            elif json_text[i] == "]":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    # 裸字面量（数字 / true / false / null）
    m2 = re.compile(r"[^,}\s]+").match(json_text, start)
    if m2:
        return (start, m2.end())
    return None


def plan_field_char_spans(output_text: str) -> List[Tuple[int, int, str]]:
    """
    规划 Assistant 输出中需要特殊加权的字段区间（绝对字符下标，相对 output_text 起点）。

    返回: [(char_start, char_end, weight_key), ...]
      - CoT 区间: (cot_start, cot_end, "cot") —— 半开区间，不含 <|end_of_thought|> 分隔符
      - JSON 各字段: (val_start, val_end, weight_key)

    注: 分隔符 <|end_of_thought|> 本身归入 default 权重（结构骨架），与 CoT 的 1.0x 数值相同，
        语义上区分开便于 dry-run 检视与后续独立调参。
    """
    spans: List[Tuple[int, int, str]] = []

    if "<think>" in output_text:
        cot_start = output_text.index("<think>") + len("<think>")
    else:
        cot_start = 0

    json_start = find_json_start(output_text)
    cot_end = output_text.index(THOUGHT_END_TOKEN) if THOUGHT_END_TOKEN in output_text else json_start

    # 1. CoT 区间（1.0x）—— 截止到分隔符之前
    if cot_end > cot_start:
        spans.append((cot_start, cot_end, "cot"))

    # 2. JSON 字段区间 —— 逐个字段定位其「值」的 span（值唯一，避免键名误命中）
    json_text = output_text[json_start:]
    try:
        parsed = json.loads(json_text.strip())
    except Exception:
        return spans

    if not isinstance(parsed, dict):
        return spans

    # risk_categories 为数组，其内部每个枚举值单独提权
    risk_span = find_value_span(json_text, "risk_categories")
    if risk_span:
        arr_text = json_text[risk_span[0]:risk_span[1]]
        if re.fullmatch(r"\[\s*\]", arr_text):
            # 空表 ([])：括号本身携带「该空就空」的判别信号，必须保留监督。
            # 若随骨架一并遮掉，空表样本的 [] 变成零梯度 —— 模型只走「有枚举值」
            # 那条受监督的路径，等于反向教会它硬编 Key（与空表注入的初衷相反）。
            spans.append((json_start + risk_span[0], json_start + risk_span[1],
                          "risk_categories"))
        else:
            for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', arr_text):
                vs = risk_span[0] + m.start()
                ve = risk_span[0] + m.end()
                spans.append((json_start + vs, json_start + ve, "risk_categories"))

    for key, wkey in JSON_FIELD_TO_WEIGHT_KEY.items():
        if key == "risk_categories":
            continue
        span = find_value_span(json_text, key)
        if span:
            spans.append((json_start + span[0], json_start + span[1], wkey))

    return spans


def plan_token_weights(
    prompt_text: str,
    output_text: str,
    encode: EscapeAware,
    max_len: Optional[int] = None,
) -> Dict[str, object]:
    """
    生成 token 级损失权重向量。

    参数:
      prompt_text: 输入侧（system + user + assistant 前缀），整段权重为 0
      output_text: Assistant 输出（CoT + JSON）
      encode: 文本 → token id 列表 的编码函数
      max_len: 截断长度上限（与 training max_seq_length 一致）

    返回:
      {
        "weights": [...],            # token 级权重
        "full_len": int,
        "prompt_len": int,
        "truncated": bool,
        "spans": [(t_start, t_end, weight_key, weight), ...],  # 供 dry-run 检视
      }
    """
    prompt_ids = encode(prompt_text)
    full_ids = encode(prompt_text + output_text)

    truncated = False
    if max_len is not None and len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        truncated = True
    full_len = len(full_ids)
    prompt_len = min(len(prompt_ids), full_len)

    weights = [FIELD_WEIGHTS["prompt"]] * prompt_len + \
              [FIELD_WEIGHTS["default"]] * (full_len - prompt_len)

    # ---- 3. JSON 结构骨架遮蔽 (Skeleton Masking) —— 仅 POC2 / POC3 ----
    # 只遮 JSON「语法」：键名 / 引号 / 括号 / 冒号 / 逗号 —— 位置固定、信息熵为零，
    # 算它们的 Loss 等于把大头梯度烧在背模板上。
    # 绝不遮协议分隔符 <think> / </think>：它们位于 prompt 之后、
    # JSON 之前，归 default 1.0 必须留梯度 —— 这些串未必在基座词表里，遮了就再也学不会发。
    # 前提是 JSON 真解析得开：解析失败则整体保留 default，避免整条样本零梯度、分母坍缩。
    if USE_SKELETON_MASK:
        json_start_char = find_json_start(output_text)
        json_ok = False
        if json_start_char < len(output_text):
            try:
                json_ok = isinstance(json.loads(output_text[json_start_char:].strip()), dict)
            except Exception:
                json_ok = False
        if json_ok:
            t_json = len(encode((prompt_text + output_text)[:len(prompt_text) + json_start_char]))
            t_json = max(min(t_json, full_len), prompt_len)
            for i in range(t_json, full_len):
                weights[i] = FIELD_WEIGHTS["skeleton"]

    # 4. 字符级 span → token 下标（POC2/POC3 下覆盖在骨架 0 之上：字段 Value 重新拿到权重）
    token_spans = []
    for char_s, char_e, wkey in plan_field_char_spans(output_text):
        abs_s = len(prompt_text) + char_s
        abs_e = len(prompt_text) + char_e

        t_s = len(encode((prompt_text + output_text)[:abs_s]))
        t_e = len(encode((prompt_text + output_text)[:abs_e]))

        t_s = max(min(t_s, full_len), prompt_len)
        t_e = max(min(t_e, full_len), t_s)
        if t_e <= t_s:
            continue

        w = span_weight(wkey)
        for i in range(t_s, t_e):
            weights[i] = w
        token_spans.append((t_s, t_e, wkey, w))

    # 越界防御：确保长度一致
    weights = weights[:full_len]

    return {
        "weights": weights,
        "full_len": full_len,
        "prompt_len": prompt_len,
        "truncated": truncated,
        "spans": token_spans,
        "poc_stage": POC_STAGE,
        "use_skeleton_mask": USE_SKELETON_MASK,
        "use_field_weights": USE_FIELD_WEIGHTS,
    }


# ===== 供无 tokenizer 环境演示/自测的字符级 stub =====
def char_level_encode(text: str) -> List[int]:
    """字符级伪编码：1 字符 = 1 token，用于离线验证 span 换算正确性。"""
    return [ord(c) for c in text]
