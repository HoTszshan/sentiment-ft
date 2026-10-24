#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2 输出协议契约 · 唯一判定源（训练侧与评测侧共用）

为什么要有这个文件（评估手册缺口 11）:
  训练侧 `reward_fns.validate_schema()` 把「闭集越界 / 缺字段 / 嵌套对象」判为违规 →
  `R_format` 短路 -2.5；评测侧 `evaluate_sft.py` 过去只做 `json.loads()`，同一条输出在
  评测端被算成「格式合法」。**两侧判定不一致 = 训练在惩罚的东西，验收却看不见**，
  这是把训练契约引入验收的第一道缝。

  落地: 闭集枚举 + `validate_schema()` + `parse_output()` 全部收进本文件；
  训练侧 `reward_fns` 改为 import 再导出（老代码 `from reward_fns import VALID_RISK` 零改动）；
  评测侧 `evaluate_sft` / `rubric_check` / `release_gate` 一律从本文件取判定，
  **谁都不许再各写一份**。

铁律:
  - 闭集枚举只有这一处定义，`corpus_schema` / `reward_fns` / 评测脚本全部引用它；
  - `validate_schema()` 判定式逐字保持原样（训练侧的 -2.5 短路语义不许被评测端改软）；
  - 本文件零外部依赖（只用 json / re），可被单测直接 import。

真理源: MVP/notes/舆情负面精判_评估手册_SFT_GRPO.md 模块一 ②「契约层」
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 闭集枚举 (与 SFT 输出协议、anno_prompt 法典逐字一致)
# ============================================================

VALID_SENTIMENT = {"negative", "positive", "neutral"}
VALID_RELEVANCE = {"core", "passing", "foil"}
VALID_HARD_CASE = {
    "irony", "metaphor", "multi_subject", "comparative",
    "counterfactual", "fact_implicit", "explicit",
}
VALID_RISK = {
    "product", "service", "marketing", "compliance",
    "governance", "finance", "general_negative",
}
# 正式 6 类业务风险 (general_negative 是兜底吐槽，不构成「危机」)
CRISIS_RISK_KEYS = VALID_RISK - {"general_negative"}

# 输出协议必填字段 (扁平单对象 JSON)
REQUIRED_FIELDS = ("subject", "relevance", "sentiment",
                   "hard_case_type", "risk_categories", "evidence")
# evidence 长度窗 (契约硬约束: 4~80 字)
EVIDENCE_MIN_LEN = 4
EVIDENCE_MAX_LEN = 80

# 思考标签 (训练模板与推理模板逐字一致)
THOUGHT_BEGIN = "<think>"   # Qwen3 词表原生标签 (id 151667)，不再用自定义碎 token
THOUGHT_END = "</think>"    # Qwen3 词表原生标签 (id 151668)

_THOUGHT_RE = re.compile(re.escape(THOUGHT_BEGIN) + r"(.*?)" + re.escape(THOUGHT_END),
                         re.DOTALL)

# 终句标记：属于本轮对话的收尾符，不是模型"多写了话"。
# 生成侧 stop token 与 decode(skip_special_tokens=False) 组合会让每条输出都带它，
# 若计入尾赘，尾赘率恒为 100%，是纯测量噪声。
TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")


def strip_terminal_markers(text: str) -> str:
    """剥掉末尾的对话收尾符（可叠多个，中间允许空白）。"""
    out = text.rstrip()
    changed = True
    while changed:
        changed = False
        for mk in TERMINAL_MARKERS:
            if out.endswith(mk):
                out = out[: -len(mk)].rstrip()
                changed = True
    return out


# ============================================================
# 契约校验
# ============================================================

def _list_of_str(x: Any) -> List[str]:
    """把任意值安全转成字符串列表 (非数组一律当空表处理)。"""
    if not isinstance(x, (list, tuple)):
        return []
    return [str(v) for v in x]


def validate_schema(pred: Optional[dict]) -> Optional[str]:
    """
    闭集校验主判定式：**原样搬自 reward_fns，判定逻辑一行不许改**。

    返回违规原因字符串；合法返回 None。
    """
    if pred is None:
        return "JSON 解析失败"
    required = ["subject", "relevance", "sentiment", "hard_case_type",
                "risk_categories", "evidence"]
    missing = [k for k in required if k not in pred]
    if missing:
        return f"缺字段 {missing}"
    if not isinstance(pred.get("sentiment"), str) or pred["sentiment"] not in VALID_SENTIMENT:
        return f"sentiment 越界: {pred.get('sentiment')!r}"
    if not isinstance(pred.get("relevance"), str) or pred["relevance"] not in VALID_RELEVANCE:
        return f"relevance 越界: {pred.get('relevance')!r}"
    if not isinstance(pred.get("hard_case_type"), str) or \
            pred["hard_case_type"] not in VALID_HARD_CASE:
        return f"hard_case_type 越界: {pred.get('hard_case_type')!r}"
    rk = pred["risk_categories"]
    if not isinstance(rk, list):
        return "risk_categories 非数组"
    bad = [k for k in _list_of_str(rk) if k not in VALID_RISK]
    if bad:
        return f"risk_categories 越界 {bad}"
    ev = pred.get("evidence", "")
    if not isinstance(ev, str) or not (EVIDENCE_MIN_LEN <= len(ev) <= EVIDENCE_MAX_LEN):
        return f"evidence 长度违规 (len={len(str(ev))})"
    if any(isinstance(v, dict) for v in pred.values()):
        return "存在嵌套对象 (契约要求平铺)"
    return None


# ============================================================
# 输出解析
# ============================================================

def _new_flags() -> Dict[str, Any]:
    return {"valid_thought_tags": False, "valid_json": False,
            "trailing_garbage": False, "schema_error": None}


def parse_output(output_str: Any) -> Tuple[Optional[str], Optional[dict], Dict[str, Any]]:
    """
    拆解模型输出: `(cot_text, parsed_json, flags)`

    flags 含义:
      valid_thought_tags —— 思考标签是否闭合 (缺标签 = 协议不全)
      valid_json         —— JSON 段能否解析成 dict
      trailing_garbage   —— JSON 之后是否有多余文本
      schema_error       —— 闭集校验违规原因 (None = 合法)

    语义与训练侧 `reward_fns.split_completion` **逐字一致**：
    缺标签一律判协议失败，不做裸 JSON 降级（降级只发生在评测侧的诊断路径
    `parse_lenient`）。
    """
    flags = _new_flags()
    if not isinstance(output_str, str):
        return None, None, flags

    # 收尾符先剥掉再解析：它是解码的 stop token，不是模型多写的正文。
    # 不剥的话 JSON 段后面永远挂着 <|im_end|>，尾赘率恒为 100%。
    output_str = strip_terminal_markers(output_str)

    m = _THOUGHT_RE.search(output_str)
    if not m:
        # 缺 CoT 标签 = 协议不全
        return None, None, flags
    flags["valid_thought_tags"] = True
    cot = m.group(1).strip()

    tail = output_str[m.end():].lstrip()
    if not tail.startswith("{"):
        flags["trailing_garbage"] = True
        return cot, None, flags
    # 先找顶层 JSON 对象边界：首个 "{" 配对其闭合 "}"
    # （用 rfind 会把「JSON 之后含 } 的尾赘文本」吃进解析段，
    #   json 失败却丢了 trailing_garbage 诊断信号）
    depth = 0
    last = -1
    in_str = False
    esc = False
    for i, ch in enumerate(tail):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last = i
                break
    if last == -1:
        # 括号不配对（截断输出）：解析失败；后面还有非空白内容本身就该记尾赘
        if tail[1:].strip():
            flags["trailing_garbage"] = True
        return cot, None, flags
    try:
        parsed = json.loads(tail[: last + 1])
    except Exception:
        return cot, None, flags
    if tail[last + 1:].strip():
        flags["trailing_garbage"] = True
    flags["valid_json"] = isinstance(parsed, dict)
    pred = parsed if isinstance(parsed, dict) else None
    if pred is not None:
        flags["schema_error"] = validate_schema(pred)
    return cot, pred, flags


def parse_lenient(output_str: Any) -> Tuple[Optional[str], Optional[dict], Dict[str, Any]]:
    """
    诊断用宽松解析：无思考标签时降级尝试裸 JSON。

    **降级不改变协议判定** —— `valid_thought_tags` 仍为 False，闭合率照常掉，
    只是在报告里额外给出一个可读的结构，便于定位「格式漂移」类缺陷。
    """
    cot, pred, flags = parse_output(output_str)
    if pred is None and not flags["valid_thought_tags"] and isinstance(output_str, str):
        try:
            obj = json.loads(output_str.strip())
        except Exception:
            obj = None
        if isinstance(obj, dict):
            pred = obj
            flags["valid_json"] = True
            flags["schema_error"] = validate_schema(obj)
    return cot, pred, flags


# ============================================================
# 行级契约：pred 载荷归一 + 金标取值
# ============================================================

def unwrap_output_text(raw: str) -> str:
    """从预测行里取出真正的模型输出文本（兼容 {"output": str} 包装）。

    剥壳逻辑唯一实现（此前在 evaluate_sft / scripts/release_gate 各抄一份）。
    非 dict / 非 output 键 / 解析失败 → 原样返回 stripped 文本。
    """
    try:
        obj = json.loads((raw or "").strip())
        if isinstance(obj, dict) and isinstance(obj.get("output"), str):
            return obj["output"]
    except Exception:
        pass
    return (raw or "").strip()


def load_pred_line(raw: str) -> Tuple[Optional[dict], str]:
    """
    把预测文件的一行规范化为 6 字段 dict。返回 `(pred_dict, form)`。

    form 取值:
      "text"    —— 该行提供了模型输出文本（可判思考标签闭合率）
      "dict"    —— 该行直接给了结构化预测（无 CoT，闭合率不适用）
      "invalid" —— 既不是文本也不是合法 dict

    支持的三种输入形态（与《评测验收SOP》一致）:
      ① 裸模型输出字符串
      ② {"output": "..."} 或 {"output": {...}}
      ③ {"prediction": {...}}
    """
    raw = (raw or "").strip()
    if not raw:
        return None, "invalid"
    try:
        obj = json.loads(raw)
    except Exception:
        # 形态①: 裸模型输出字符串
        _, pred, _ = parse_lenient(raw)
        return pred, "text"

    if isinstance(obj, dict):
        if isinstance(obj.get("prediction"), dict):
            return obj["prediction"], "dict"
        if "output" in obj:
            out = obj["output"]
            if isinstance(out, dict):
                return out, "dict"
            _, pred, _ = parse_lenient(out)
            return pred, "text"
        if "hard_case_type" in obj:
            return obj, "dict"
    return None, "invalid"


def resolve_gold(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    金标字段来源统一（评估手册缺口 15）。

    铁律: **金标一律以 output JSON 为准**；`meta` 只用于
      ① 分桶维度（industry）；
      ② output JSON 缺字段时的兜底；
    两者冲突 → 返回冲突清单（不静默），由上层打印前 5 条。

    返回 `(gold_dict, conflicts)`；gold 统一含
    subject / relevance / sentiment / hard_case_type / risk_categories / evidence。
    """
    _, parsed, _ = parse_lenient(row.get("output", ""))
    out = parsed if isinstance(parsed, dict) else {}
    meta = row.get("meta", {}) or {}

    gold: Dict[str, Any] = {}
    conflicts: List[str] = []

    for field in ("sentiment", "relevance", "hard_case_type"):
        v = out.get(field)
        mv = meta.get(field)
        if v is None:
            gold[field] = mv                      # 兜底：output 缺 → 用 meta
        else:
            gold[field] = v
            if mv is not None and mv != v:
                conflicts.append(f"{field}: output={v!r} vs meta={mv!r}")

    rk = out.get("risk_categories")
    if rk is None:
        gold["risk_categories"] = meta.get("risk_categories", [])
    else:
        gold["risk_categories"] = rk
        mrk = meta.get("risk_categories")
        if isinstance(mrk, list) and sorted(map(str, mrk)) != sorted(map(str, rk)):
            conflicts.append(f"risk_categories: output={sorted(map(str, rk))} "
                             f"vs meta={sorted(map(str, mrk))}")

    gold["evidence"] = out.get("evidence", "")
    gold["subject"] = out.get("subject", meta.get("subject", ""))
    return gold, conflicts
