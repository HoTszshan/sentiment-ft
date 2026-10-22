#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证据支撑性判定 · 公共函数（评估手册 TODO 5.4「共调」项）

背景:
  语料生成侧「查④」与评估手册 Rubric 第③项问的是**同一个问题** ——
  「这段 evidence 能不能推出这个结论（情感 + 风险）」。
  之前是两处各写一份实现（`generate_corpus.judge_support` 一份、
  `rubric_check.evidence_supports` 一份），口径必然漂移：
  查④ 问的是语义支撑（调裁判模型），Rubric 问的却是「是不是原文子串」
  （机械断言，名字与实际做的事对不上，而且与查② 重复）。

本文件把这件事收敛成**一个判定源**:
  - `build_support_prompt()`  —— 拼给裁判模型的标准提问（唯一一份）;
  - `judge_evidence_support()` —— 调裁判 + 解析结论，两处共调;
  - 机械硬前提（evidence 必须是原文严格连续子串 + 长度窗 4~80）
    抽成 `mechanical_precheck()`，由调用方先跑 —— 不合格直接判负，
    不必浪费一次裁判调用。

三条铁律（与 `rubric_check` 一致，防被后来改成「进梯度」）:
  - 仅离线调用；
  - 结果不进梯度、不做连续分，只做样本准入;
  - 数据单向流动。

纯标准库，无网络依赖；`call` 是注入的 `Callable[[str], str]`（prompt → 原始返回文本），
本机单测可传假裁判。
"""

import json
import re
from typing import Any, Callable, Dict, Optional, Tuple

# 证据长度窗（与 schema_contract.EVIDENCE_MIN_LEN / MAX_LEN 一致，重复定义在此
# 是为了本文件零 import；两处数值必须同步，改一处就要改另一处 —— 见自检）
EVIDENCE_MIN_LEN = 4
EVIDENCE_MAX_LEN = 80

# 裁判输出的判定词（自由文本回退路径）
PASS_TOKENS = ("yes", "pass", "通过", "满足", "ok")
FAIL_TOKENS = ("no", "fail", "不通过", "不满足", "违反")


def mechanical_precheck(evidence: Any, input_text: Optional[str]
                        ) -> Tuple[bool, str]:
    """
    机械硬前提：evidence 必须是原文的**严格连续子串**且长度落在 4~80 窗内。

    返回 (是否通过, 原因)。原文缺失时判「不可判定」—— 交由上层决定是否跳过。
    """
    if not isinstance(evidence, str) or not evidence:
        return False, "缺 evidence，结论无支撑"
    if not (EVIDENCE_MIN_LEN <= len(evidence) <= EVIDENCE_MAX_LEN):
        return False, f"evidence 长度越界 (len={len(evidence)})"
    if not input_text:
        return True, "原文缺失，机械前提跳过"
    if evidence not in input_text:
        return False, "evidence 非原文严格连续子串"
    return True, ""


def build_support_prompt(text: str, target: str, pred: Dict[str, Any]) -> str:
    """拼给裁判模型的标准提问（唯一一份，查④ 与 Rubric 共用）。"""
    return (
        "你是舆情判定复核员。只回答：给定的<证据>能否推出"
        f"「对[{target}]的情感=[{pred.get('sentiment')}]、命中风险={pred.get('risk_categories')}」。独立判断。\n\n"
        f"原文：{text}\n"
        f"<证据>：{pred.get('evidence')}\n"
        f"待核结论：sentiment={pred.get('sentiment')} risk={pred.get('risk_categories')}\n\n"
        '输出 JSON：{"support":"yes|no","reason":"简短原因"}'
    )


def parse_support_reply(raw: str) -> Tuple[bool, str]:
    """
    解析裁判回复 → (是否支撑, 原因)。

    两条解析路径:
      ① 结构化 JSON（{"support": "yes|no", "reason": ...}）—— 首选;
      ② 自由文本里的判定词（yes/no/PASS/FAIL/通过/不通过）—— 兜底。
    两条都解析不出来 → 判**不支撑**（不可判定 ≠ 放行）。
    """
    raw = (raw or "").strip()
    if not raw:
        return False, "裁判未给出结论"

    # 路径①：JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "support" in obj:
            sup = str(obj.get("support", "")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if sup.startswith("y"):
                return True, reason or "裁判判定支撑"
            if sup.startswith("n"):
                return False, reason or "裁判判定不支撑"
    except Exception:
        pass

    # 路径②：自由文本判定词（先判 FAIL，避免 "not ok" 之类的误判）
    low = raw.lower()
    for tok in FAIL_TOKENS:
        if tok in low:
            return False, f"裁判判定不支撑: {raw[:80]}"
    for tok in PASS_TOKENS:
        if tok in low:
            return True, f"裁判判定支撑: {raw[:80]}"
    return False, f"裁判回复无法解析: {raw[:80]}"


def judge_evidence_support(call: Optional[Callable[[str], str]],
                           text: str, target: str,
                           pred: Dict[str, Any]) -> Tuple[bool, str]:
    """
    证据支撑性判定（语义层）—— 查④ 与 Rubric 第③项共调的唯一入口。

    参数:
      call —— 注入的裁判调用 `prompt -> 原始文本`；None 表示不跑语义判定
              （只做机械前提），返回 (True, "未注入裁判，语义项跳过")。
    返回: (是否支撑, 原因)
    """
    if call is None:
        return True, "未注入裁判，语义项跳过"

    prompt = build_support_prompt(text, target, pred)
    try:
        raw = call(prompt) or ""
    except Exception as ex:
        return False, f"裁判调用异常: {ex}"
    return parse_support_reply(raw)
