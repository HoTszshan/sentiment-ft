#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情负面精判 · Rule-based GRPO 四维 Reward 打分器

唯一真理源: MVP/notes/舆情负面精判_GRPO数据准备实战手册.md §五「四维 Reward 契约」

设计要点 (面试防御点):
1. 两个门禁项 (R_format / R_grounding) 只给 {0, -2.5}，**绝不给正分** —— 防止
   「格式对 + 粘贴原文 + 判别全扔」白拿分刷格式。
2. 门禁短路罚 -2.5 必须**重于**最严重的业务罚 -2.0 (危机漏报) —— 否则模型在
   拿不准时会**主动崩 JSON 止损骗分**，等于亲手开辟一条对齐欺骗通道。
3. 漏报 = 误报 ×2 (-2.0 vs -1.0) —— 漏掉 P0 危机的代价远超误报，奖励必须显式
   编码这种不对称，否则模型整体倒向「宁可多报」，误报率飙回 70%~90% 老路。
4. R_length 只有「判对 + 溯源双通过」才激活；判错或证据伪造 → 硬置 0。
5. R ∈ [-2.5, +2.0]，且严格单调:
   -2.5 (协议/证据) < -2.0 (危机漏报) < -1.0 (误报) < 0 (判对但 CoT 空壳) < +1.0~+2.0

零外部依赖: 不 import torch / transformers / pydantic，纯 Python 可单测。
"""

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 闭集枚举与闭集校验已下沉到 schema_contract.py（评估手册缺口 11）：
# 训练侧与评测侧必须共用同一份判定式，否则「训练在罚的、验收却看不见」。
# 下面按**原名再导出**，保证 `from reward_fns import VALID_RISK` 之类老导入零改动。
from schema_contract import (  # noqa: E402,F401
    CRISIS_RISK_KEYS,
    EVIDENCE_MAX_LEN,
    EVIDENCE_MIN_LEN,
    REQUIRED_FIELDS,
    VALID_HARD_CASE,
    VALID_RELEVANCE,
    VALID_RISK,
    VALID_SENTIMENT,
    _list_of_str,
    validate_schema,
)

# ============================================================
# 闭集枚举 (与 SFT 输出协议、anno_prompt 法典逐字一致)
# —— 权威定义在 lib/schema_contract.py，此处仅作兼容再导出
# ============================================================

# ============================================================
# 值域与档位常量 (单调性铁律的物化)
# ============================================================

R_GATE_SHORT_CIRCUIT = -2.5   # 协议损坏 / 证据造假
R_MISS_SEVERE = -2.0          # 严重危机漏报
R_FALSE_ALARM = -1.0          # 误报 / 普通判错
R_CORRECT_FULL = 1.0          # 判别全中 (封顶)
R_RELEVANCE_LIGHT = 0.2       # relevance 轻权重
R_MAX = 2.0                   # 判对 + CoT 紧凑
R_MIN = -2.5

# CoT 长度档位 (tokens)
COT_SHORT = 40    # ≈55~60 汉字，三跳推理最低有效表达
COT_IDEAL_HI = 65
COT_DEAD = 120


# ============================================================
# 解析工具
# ============================================================

_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
_TEXT_RE = re.compile(r"【文本】[:：]\s*(.*?)(?:\n【|\Z)", re.DOTALL)


def extract_input_text(prompt: str) -> str:
    """
    从 GRPO 落盘的 prompt (只有 prompt+gold 两字段) 里取出待判原文，
    供 R_grounding 做「evidence ⊂ 原文」硬断言，避免在数据里重复落一份 input。
    """
    if not prompt:
        return ""
    m = _USER_RE.search(prompt)
    if m:
        return m.group(1).strip()
    m = _TEXT_RE.search(prompt)
    if m:
        return m.group(1).strip()
    return prompt


def split_completion(completion: str) -> Tuple[Optional[str], Optional[dict], dict]:
    """
    拆解模型输出: (cot_text, parsed_json, flags)
    flags: valid_thought_tags / valid_json / trailing_garbage

    判定式**委托 `schema_contract.parse_output`**（唯一判定源），
    本函数只做签名兼容（丢弃 schema_error 标志）——
    两份逐字重复的解析实现必然漂移，谁都不许再抄一份。
    """
    from schema_contract import parse_output
    cot, pred, flags = parse_output(completion)
    return cot, pred, {
        "valid_thought_tags": flags["valid_thought_tags"],
        "valid_json": flags["valid_json"],
        "trailing_garbage": flags["trailing_garbage"],
    }




def count_cot_tokens(cot_text: str, tokenizer=None) -> int:
    """
    CoT token 计数 (只量思考链，不含 JSON 骨架与 evidence 原文)。
    - 有 tokenizer: 走真实分词
    - 无 tokenizer: 按 SOP 推导的近似换算 —— 40 tokens ≈ 55~60 汉字，
      即 tokens ≈ chars × 40/55 ≈ 0.727 × chars (Qwen 中文分词经验值)
    """
    if cot_text is None:
        return 0
    if tokenizer is not None:
        return len(tokenizer(cot_text)["input_ids"])
    return max(1, int(round(len(cot_text) * (40.0 / 55.0))))


# ============================================================
# 四维 Reward
# ============================================================

def r_format(completion: str) -> float:
    """门禁①: 协议合规。+0 / -2.5 (短路)。绝不给正分。"""
    _, pred, flags = split_completion(completion)
    if not flags["valid_thought_tags"]:
        return R_GATE_SHORT_CIRCUIT
    if not flags["valid_json"]:
        return R_GATE_SHORT_CIRCUIT
    if flags["trailing_garbage"]:
        return R_GATE_SHORT_CIRCUIT
    reason = validate_schema(pred)
    return R_GATE_SHORT_CIRCUIT if reason else 0.0


def r_grounding(completion: str, input_text: str) -> float:
    """门禁②: evidence 必须是原文严格连续子串。+0 / -2.5 (短路)。"""
    _, pred, _ = split_completion(completion)
    if pred is None:
        return R_GATE_SHORT_CIRCUIT
    ev = str(pred.get("evidence", ""))
    if not ev or ev not in (input_text or ""):
        return R_GATE_SHORT_CIRCUIT
    return 0.0


def r_accuracy(completion: str, gold: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    奖励①: 与 gold 比对判别主线 (sentiment + risk 定档 / hard_case / relevance)。
    返回 (分数, 明细)。
    """
    _, pred, _ = split_completion(completion)
    detail = {"pred": pred, "miss_severe": False, "false_alarm": False,
              "hard_case_ok": False, "sentiment_ok": False, "risk_ok": False,
              "relevance_ok": False, "accuracy_pass": False}
    if pred is None or gold is None:
        return R_FALSE_ALARM, detail

    g_sent = gold.get("sentiment")
    p_sent = pred.get("sentiment")
    g_rk = set(_list_of_str(gold.get("risk_categories", [])))
    p_rk = set(_list_of_str(pred.get("risk_categories", [])))
    g_hc = gold.get("hard_case_type")
    p_hc = pred.get("hard_case_type")
    g_rel = gold.get("relevance")
    p_rel = pred.get("relevance")

    detail["sentiment_ok"] = (g_sent == p_sent)
    detail["risk_ok"] = (g_rk == p_rk)
    detail["hard_case_ok"] = (g_hc == p_hc)
    detail["relevance_ok"] = (g_rel == p_rel)

    # 定档: 漏报 vs 误报 vs 普通判错
    # 业务不对称: 漏掉 P0 危机 > 多报一条 → -2.0 / -1.0 两档;
    # ③④ 只算普通判错: 有裁决链兜底的细粒度偏差，不吃漏报/误报级的罚
    if g_sent == "negative" and p_sent != "negative":
        # 真实负面却没判负 = 漏报；gold 命中正式 6 类风险 → 严重危机漏报
        severe = bool(g_rk & CRISIS_RISK_KEYS)
        detail["miss_severe"] = severe
        base = R_MISS_SEVERE if severe else R_FALSE_ALARM
    elif g_sent != "negative" and p_sent == "negative":
        detail["false_alarm"] = True
        base = R_FALSE_ALARM
    elif not detail["sentiment_ok"] or not detail["risk_ok"]:
        base = R_FALSE_ALARM
    elif not detail["hard_case_ok"]:
        # 极性与风险都对，仅难点类型判错: 记普通判错
        # (有 7 类全序裁决链兜底，不属于主观噪声)
        base = R_FALSE_ALARM
    else:
        base = R_CORRECT_FULL
        detail["accuracy_pass"] = True

    # relevance 轻判 (契约 §五): 弱主观字段只做 ±0.2 微调，不参与定档;
    #   - 全中: 判对 +1.0 / 判错 +0.8 (已含在封顶内不外溢)
    #   - 判错档: -1.0 / 追加 -0.2 → -1.2
    #   - 危机漏报档 (-2.0) **不叠加** → 防弱主观字段压过危机漏报
    if base == R_MISS_SEVERE:
        score = base
    elif base == R_CORRECT_FULL:
        score = base if detail["relevance_ok"] else round(base - R_RELEVANCE_LIGHT, 4)
    else:
        score = base if detail["relevance_ok"] else round(base - R_RELEVANCE_LIGHT, 4)

    # accuracy_pass = 全字段口径的「判对」门票: k_of 计数、r_length 激活都只认它
    detail["accuracy_pass"] = detail["accuracy_pass"] and detail["relevance_ok"]
    return round(score, 4), detail


def r_length(completion: str, accuracy_pass: bool, tokenizer=None) -> float:
    """
    奖励②: CoT 长度。仅「判对 + 溯源双通过」才激活；否则硬置 0。
      <40   → cot/40 - 1        (过短罚)
      40~65 → +1.0             (理想区间)
      65~120→ 1-(cot-65)/55     (线性衰减)
      ≥120  → 0
    """
    if not accuracy_pass:
        return 0.0
    cot, _, _ = split_completion(completion)
    if cot is None:
        return 0.0
    n = count_cot_tokens(cot, tokenizer=tokenizer)
    if n < COT_SHORT:
        return round(n / float(COT_SHORT) - 1.0, 4)
    if n <= COT_IDEAL_HI:
        return 1.0
    if n < COT_DEAD:
        return round(1.0 - (n - COT_IDEAL_HI) / float(COT_DEAD - COT_IDEAL_HI), 4)
    return 0.0


def compute_reward(completion: str, input_text: str, gold: Dict[str, Any],
                   tokenizer=None) -> Dict[str, Any]:
    """
    汇总四维 reward: 门禁短路 + 奖励项相加，总值域 [-2.5, +2.0]。
    返回 {total, r_format, r_grounding, r_accuracy, r_length, detail}
    """
    rf = r_format(completion)
    if rf <= R_GATE_SHORT_CIRCUIT:
        return {"total": R_GATE_SHORT_CIRCUIT, "r_format": rf, "r_grounding": 0.0,
                "r_accuracy": 0.0, "r_length": 0.0,
                "detail": {"short_circuit": "format", "reason": "协议不合规"}}

    rg = r_grounding(completion, input_text)
    if rg <= R_GATE_SHORT_CIRCUIT:
        return {"total": R_GATE_SHORT_CIRCUIT, "r_format": rf, "r_grounding": rg,
                "r_accuracy": 0.0, "r_length": 0.0,
                "detail": {"short_circuit": "grounding", "reason": "evidence 非原文子串"}}

    ra, adetail = r_accuracy(completion, gold)

    # 门禁不变量2: 长度分只发给「判对 且 溯源通过」的轨迹
    ri = r_length(completion, accuracy_pass=adetail["accuracy_pass"], tokenizer=tokenizer)

    total = round(rf + rg + ra + ri, 4)
    return {"total": total, "r_format": rf, "r_grounding": rg,
            "r_accuracy": ra, "r_length": ri,
            "detail": {"short_circuit": None, **adetail}}


def group_advantages(totals: Sequence[float], eps: float = 1e-8,
                     scale_by_std: bool = False) -> List[float]:
    """
    GRPO组内相对优势：同题G份采样自对比，无Critic。
    默认仅减均值、不除组内std，对齐GRPOConfig(scale_rewards="none")。
    原因：奖励为固定离散阶梯分，组内分差常很小；除以std会放大噪声，让全错组内最优样本获得过高优势，诱导模型学习错误行为排序。
    std归一化适用于尺度不稳定的学习型RM；本任务奖励尺度固定，无需二次校准。scale_by_std=True仅用于对照实验。
    
    std 过小 (全对 / 全错 / 同分) → 整组优势清零, 两种口径下结果相同。
    """
    vals = list(totals)
    if not vals:
        return []
    mean = sum(vals) / len(vals)
    centered = [v - mean for v in vals]
    if not scale_by_std:
        if all(abs(c) < eps for c in centered):
            return [0.0] * len(vals)
        return [round(c, 6) for c in centered]
    std = math.sqrt(sum(c * c for c in centered) / len(vals))
    if std < eps:  # 全对/全错: 方差为 0 → 整组优势清零
        return [0.0] * len(vals)
    return [round(c / std, 6) for c in centered]


def zero_advantage_masking(totals: Sequence[float]) -> List[float]:
    """
    Zero-Advantage Masking: 全错组 (整组触顶门禁罚) 整组清零。
    RL 无米下锅时不放大「错得轻的那次」，退回 SFT 补课。
    """
    vals = list(totals)
    if vals and max(vals) <= R_GATE_SHORT_CIRCUIT + 1e-9:
        return [0.0] * len(vals)
    return group_advantages(vals)


# ============================================================
# 门禁不变量自检 (可被单测直接调用)
# ============================================================

def assert_contract_invariants() -> bool:
    """三条门禁不变量 + 单调性铁律，任一破坏即抛 AssertionError。"""
    # 1. 门禁项只取 {0, -2.5}，绝不给正分 —— 用**真实行为**验证，
    #    不对常量断常量（恒真断言防不住 r_format 实现被改坏）
    _ok_out = ('<think>x</think>'
               '{"subject":"某品牌","relevance":"core","sentiment":"negative",'
               '"hard_case_type":"explicit","risk_categories":["product"],'
               '"evidence":"才一天就亏了3000多"}')
    assert r_format(_ok_out) == 0.0, "合规输出必须得 0 分（门禁不放行也不扣分）"
    assert r_format("格式崩了") == R_GATE_SHORT_CIRCUIT, "崩格式必须短路 -2.5"
    assert r_format('<think>x</think>{"sentiment":"越界"}') \
        == R_GATE_SHORT_CIRCUIT, "闭集越界必须短路 -2.5"
    # 2. 短路罚 > 业务罚 (严格更负)
    assert R_GATE_SHORT_CIRCUIT < R_MISS_SEVERE, "协议罚必须重于危机漏报"
    assert R_GATE_SHORT_CIRCUIT < R_FALSE_ALARM
    # 3. 漏报 = 误报 ×2
    assert R_MISS_SEVERE == R_FALSE_ALARM * 2, "漏报与误报必须是 2 倍不对称"
    # 4. 单调铁律
    order = [R_GATE_SHORT_CIRCUIT, R_MISS_SEVERE, R_FALSE_ALARM, 0.0, R_CORRECT_FULL, R_MAX]
    assert order == sorted(order), f"单调性被破坏: {order}"
    # 5. 总分值域
    assert R_MIN == -2.5 and R_MAX == 2.0
    return True


if __name__ == "__main__":
    ok = assert_contract_invariants()
    print("=" * 70)
    print("GRPO 四维 Reward 契约自检")
    print(f"  R_format   : 0 / {R_GATE_SHORT_CIRCUIT}  (门禁，短路)")
    print(f"  R_grounding: 0 / {R_GATE_SHORT_CIRCUIT}  (门禁，短路)")
    print(f"  R_accuracy : [{R_MISS_SEVERE}, {R_CORRECT_FULL}]  (漏报-2.0 / 误报-1.0 / 全中+1.0)")
    print(f"  R_length   : [{R_FALSE_ALARM}, {R_CORRECT_FULL}]  (仅判对时激活)")
    print(f"  总分值域   : [{R_MIN}, {R_MAX}]")
    print(f"  单调链     : -2.5 (协议) < -2.0 (危机漏报) < -1.0 (误报) < 0 (空壳) < +1~+2")
    print(f"  三条不变量 : {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
