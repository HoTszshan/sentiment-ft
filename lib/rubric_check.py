#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 Rubric 三判定 · 推理链复核（评估手册缺口 4）

背景: 现有「四查」第 ③ 关只是 **CoT 与最终标签的正则自洽比对**（形如
"Step2 判 irony" 里有没有 irony 字样）—— 这只验「说得自洽」，不验「说得对」，
更不验「有没有在推理」。一个把主体、结论、证据全跳过的空壳 CoT 照样能过。

本模块补三条 checklist:
  1. `subject_named`   —— CoT 是否点名了目标主体（不点名 = 推理根本没锚定对象）
  2. `no_contradiction`—— 推理与最终标签是否矛盾（CoT 说 core、标签写 passing 之类）
  3. `evidence_supports`—— 证据是否撑得起结论（evidence 必须是原文严格连续子串，
                            且长度窗合法；这是**可离线判定**的那一半）

三条约束（铁律，写在代码里防止被人后来改成"进梯度"）:
  - **仅离线调用** —— 本模块只在评测链路里跑，绝不进训练损失；
  - **结果不进梯度** —— 只产出 pass/fail 供人看，不回传任何梯度信号；
  - **数据单向流动** —— 只读评测侧数据，不写回训练语料。

`judge` 是一个注入的 `Callable[[str], str]`：把「CoT + 证据 + 原文」拼成
一段 prompt 交给一个离线裁判模型，返回裁判的原始文本。
**默认 None 时只跑可判定的规则子集**（subject 是否出现在 CoT、evidence 是否子串），
这样没有裁判模型的环境（CI / 本机）也能单测。
"""

import os
import re
import sys
from typing import Any, Callable, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evidence_support import (  # noqa: E402
    judge_evidence_support,
    mechanical_precheck,
)
from schema_contract import EVIDENCE_MAX_LEN, EVIDENCE_MIN_LEN  # noqa: E402

# 离线裁判模型的判定词（裁判返回文本里出现即视为通过）
JUDGE_PASS_TOKENS = ("PASS", "通过", "满足", "OK")
JUDGE_FAIL_TOKENS = ("FAIL", "不通过", "不满足", "违反")

# 枚举标识符扫描：只取「英文/下划线构成的完整词」，
# 避免 explicit ⊂ fact_implicit 这类子串误判
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _contains_any(text: str, tokens) -> bool:
    return any(t in text for t in tokens)


def rubric_check(cot: Optional[str], pred: Optional[Dict[str, Any]],
                 input_text: Optional[str],
                 judge: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """
    离线 Rubric 三判定。

    返回: {"pass": bool,
           "items": {"subject_named": bool|None,
                     "no_contradiction": bool|None,
                     "evidence_supports": bool|None},
           "reason": str}

    `items` 里的值:
      True  —— 判定通过；
      False —— 判定不通过；
      None  —— **本项不可判定**（缺输入、没注入裁判、或结论存疑），
               None 不参与 pass 判定（不因「测不了」而判失败，
               但会在 reason 里点名，让人知道这项是空跑）。

    判定细则:
      - `subject_named`: 取 pred["subject"] 与 cot，主体串在 CoT 中出现即通过；
        主体为空或 cot 为空 → None（不可判定）。
      - `no_contradiction`: CoT 里出现与最终标签互斥的枚举值即判矛盾。
        例：CoT 出现 "positive" 而 sentiment 标签是 "negative" → 矛盾。
        CoT 没提到任何极性/难点枚举 → None（不可判定，不误杀）。
      - `evidence_supports`: 两层 ——
          机械前提：evidence 必须是原文**严格连续子串**且长度窗 4~80 合法；
          语义支撑：「这段证据能不能推出这个结论」，由离线裁判判
          （**与语料生成侧查④ 同一份实现** lib/evidence_support.py）；
          未注入裁判时只跑机械前提那一半，reason 里点名「语义项跳过」。
        原文缺失（input_text 为空）→ None。
    """
    items: Dict[str, Optional[bool]] = {
        "subject_named": None,
        "no_contradiction": None,
        "evidence_supports": None,
    }
    reasons: list = []

    cot_text = (cot or "").strip()
    pred = pred if isinstance(pred, dict) else {}

    # ---- 1. subject_named ----
    subject = str(pred.get("subject") or "").strip()
    if subject and cot_text:
        items["subject_named"] = subject in cot_text
        if not items["subject_named"]:
            reasons.append(f"CoT 未点名目标主体「{subject}」")
    else:
        reasons.append("subject_named 不可判定（缺 subject 或缺 CoT）")

    # ---- 2. no_contradiction ----
    # 互斥对：CoT 里出现与最终标签**相反**的闭集值 → 推理与标签矛盾。
    # 注意口径：CoT 只提到**与标签一致**的枚举值时不算矛盾；
    # 只有出现「与标签不同」的枚举值才判矛盾 —— 但 CoT 文本里「判 irony」与
    # 「判 negative」这类**表述本身不构成互斥证据**（中文推理常先列候选再排除），
    # 所以互斥检查只看**语义冲突方向**：极性上出现与标签相反的另一个极性值，
    # 或难点上出现与标签不同的另一个难点值，且 CoT 里**没有**「非」「不是」「排除」
    # 等排除词修饰它。简化落地：标签值必须也在 CoT 里出现（推理落到了标签上），
    # 同时出现了另一个**同族**枚举值且未跟排除词 → 才判矛盾。
    from schema_contract import VALID_HARD_CASE, VALID_SENTIMENT  # 局部导入避免循环
    contradiction = _check_contradiction(cot_text, pred, VALID_SENTIMENT, VALID_HARD_CASE)
    if contradiction is None:
        # 标签值在 CoT 中出现过 → 推理确实落到了标签上，判通过；
        # 完全没提 → 不可判定
        label_in_cot = _label_mentioned(cot_text, pred)
        if label_in_cot:
            items["no_contradiction"] = True
        else:
            reasons.append("no_contradiction 不可判定（CoT 未提及极性/难点枚举）")
    else:
        items["no_contradiction"] = not contradiction
        if contradiction:
            reasons.append(f"推理与标签矛盾：{contradiction}")

    # ---- 3. evidence_supports（与语料生成侧「查④」共调）----
    # 两层判定:
    #   机械前提 —— evidence 必须是原文严格连续子串 + 长度窗 4~80
    #                （不合格直接判负，不浪费裁判调用）
    #   语义支撑 —— 「这段证据能不能推出这个结论」，由离线裁判判
    #                （**与查④ 同一份实现** lib/evidence_support.py）
    judge_verdict: Optional[bool] = None
    evidence = pred.get("evidence")
    mech_ok, mech_why = mechanical_precheck(evidence, input_text)
    if not mech_ok:
        items["evidence_supports"] = False
        reasons.append(mech_why)
    elif not input_text:
        # 原文缺失 → 机械前提跳过后，语义项也无法判 → 不可判定
        items["evidence_supports"] = None
        reasons.append("evidence_supports 不可判定（缺原文）")
    else:
        if judge is None:
            # 无裁判：只跑机械前提那一半（CI / 本机可单测）
            items["evidence_supports"] = True
            reasons.append("evidence_supports 仅机械前提通过（未注入裁判，语义项跳过）")
        else:
            sup_ok, sup_why = judge_evidence_support(
                judge, input_text, str(pred.get("subject") or ""), pred)
            items["evidence_supports"] = sup_ok
            if not sup_ok:
                reasons.append(f"证据撑不起结论：{sup_why}")
            judge_verdict = sup_ok

    # ---- 4. 推理链整体复核（注入的离线裁判）----
    # 第③项已用「查④ 同一份实现」判过证据支撑；这里再用裁判复核**整条推理链**
    # （CoT 是否真围绕该主体在推理），裁判说不通过 → 一票否决。
    if judge is not None:
        prompt = _build_judge_prompt(cot_text, pred, input_text or "")
        try:
            raw = judge(prompt) or ""
        except Exception as ex:  # 裁判不可用不该让整条评测崩掉
            raw = ""
            reasons.append(f"judge 调用失败：{ex}")
        if _contains_any(raw, JUDGE_FAIL_TOKENS):
            judge_verdict = False
            reasons.append("离线裁判判定不通过")
        elif _contains_any(raw, JUDGE_PASS_TOKENS):
            judge_verdict = True
        else:
            reasons.append("judge 未给出可解析结论")
            judge_verdict = None

    # ---- 汇总 ----
    judged = [v for v in items.values() if v is not None]
    ok = all(v for v in judged) if judged else False
    # 裁判判了 FAIL，或裁判压根没给出可解析结论（不可判定 ≠ 放行）→ 一票否决
    if judge is not None and judge_verdict is not True:
        ok = False
    if not reasons:
        reasons.append("三项全部通过")
    return {"pass": ok, "items": items, "reason": "；".join(reasons),
            "judge_verdict": judge_verdict}


def _label_mentioned(cot_text: str, pred: Dict[str, Any]) -> bool:
    """最终标签值是否在 CoT 中被提及（推理确实落到了标签上）。"""
    tokens = set(_IDENT_RE.findall(cot_text))
    sent = pred.get("sentiment")
    hc = pred.get("hard_case_type")
    return (isinstance(sent, str) and sent in tokens) or \
           (isinstance(hc, str) and hc in tokens)


def _check_contradiction(cot_text: str, pred: Dict[str, Any],
                         valid_sentiment, valid_hard_case) -> Optional[str]:
    """
    CoT 与最终标签的互斥检查。

    实现要点: 用**标识符扫描**而不是朴素 `in` 子串判断 ——
    `explicit` 是 `fact_implicit` 的子串，朴素判断会把「CoT 写了 fact_implicit、
    标签也是 fact_implicit」误判成「CoT 出现 explicit，与标签矛盾」。

    返回 None = CoT 没提到任何可判定的枚举值（不可判定，不误杀）；
    返回字符串 = 矛盾描述。
    """
    if not cot_text:
        return None
    tokens = set(_IDENT_RE.findall(cot_text))

    # 极性：CoT 出现与标签不同的极性值 → 矛盾
    sent = pred.get("sentiment")
    if isinstance(sent, str) and sent in valid_sentiment:
        for cand in valid_sentiment:
            if cand == sent:
                continue
            if cand in tokens:
                return f"CoT 出现「{cand}」而 sentiment 标签为「{sent}」"
    # 难点类型：同上
    hc = pred.get("hard_case_type")
    if isinstance(hc, str) and hc in valid_hard_case:
        for cand in valid_hard_case:
            if cand == hc:
                continue
            if cand in tokens:
                return f"CoT 出现「{cand}」而 hard_case_type 标签为「{hc}」"
    return None


def _build_judge_prompt(cot: str, pred: Dict[str, Any], input_text: str) -> str:
    """拼给离线裁判的 prompt（三段：原文 / 推理 / 结论）。"""
    return (
        "你是舆情标注质量复核员。请判断推理链是否支撑结论，只回答 PASS 或 FAIL。\n"
        f"【原文】\n{input_text}\n"
        f"【推理】\n{cot}\n"
        f"【结论】\n{pred}\n"
    )
