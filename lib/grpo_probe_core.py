#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 探针核心 · 纯逻辑 (零 torch/trl 依赖，可独立单测)

职责: 同题 G=4 采样结果 → 四维打分 → k∈{0..4} 分流。
原为 scripts/run_grpo_probe.py 的一部分，因 train_grpo.py 复用其中的
completion 形态兼容函数而抽出到 lib/ —— CLI 脚本不再被当成库 import。

k = 同一 prompt 采 G 份，其中「判对 且 evidence 溯源过门禁」的份数。
  k=0  全错   → RL 无米下锅 (std=0，梯度为 0)，退回 SFT 补课
  k=1~3       → 训练池，组内有方差、有正负样本可比
  k=4  全对   → 组内方差恒 0，花 RL 的钱跑了个更贵的 SFT

真理源:
  MVP/notes/舆情负面精判_GRPO数据准备实战手册.md §四 难度分流 (只留 k∈{1,2,3})
  MVP/notes/GRPO训练实战手册.md §1.3 超参 / §1.4 闭环
"""

import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_fns import extract_input_text                              # noqa: E402
from grpo_common import (                                              # noqa: E402
    bucket_label, k_of, score_one,
)

# 采样器契约: (prompt, g) -> g 条 completion 文本
Sampler = Callable[[str, int], List[str]]


# ============================================================
# 纯逻辑 (可单测)
# ============================================================

def extract_completion_text(x: Any) -> str:
    """
    TRL 各版本 reward 接到的 completion 形态不一:
      list[dict] / dict / str。统一取纯文本。
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("content", ""))
    if isinstance(x, (list, tuple)) and x:
        return extract_completion_text(x[0])
    return ""


def score_group(prompt: str, completions: Sequence[str], gold: Dict[str, Any],
                g: int = 4, input_text: Optional[str] = None) -> Dict[str, Any]:
    """
    一份 prompt 的 G 条采样 → 逐条四维总分 + 组内 k。

    grounding 原文取值优先级（评估手册缺口 16：训练/评测原文同源）:
      ① 传入的 `input_text`（数据行自带，build_grpo_probe 默认落盘）
      ② 反解 prompt 兜底（见 reward_fns.extract_input_text）
    """
    if not isinstance(input_text, str) or not input_text.strip():
        input_text = extract_input_text(prompt)
    totals, details, flags = [], [], []
    for comp in completions:
        r = score_one(comp, input_text, gold)
        totals.append(r["total"])
        details.append(r["detail"])
        d = r["detail"]
        flags.append({
            "short_circuit": d.get("short_circuit"),
            "sentiment_ok": bool(d.get("sentiment_ok")),
            "risk_ok": bool(d.get("risk_ok")),
            "hard_case_ok": bool(d.get("hard_case_ok")),
            "relevance_ok": bool(d.get("relevance_ok")),
            "accuracy_pass": bool(d.get("accuracy_pass")),
            "miss_severe": bool(d.get("miss_severe")),
            "false_alarm": bool(d.get("false_alarm")),
        })
    k = k_of(details)
    return {"totals": totals, "k": k, "bucket": bucket_label(k, g),
            "flags": flags, "input_text": input_text, "gold": gold}


def run_probe(records: Sequence[Dict[str, Any]], sampler: Sampler,
              g: int = 4) -> List[Dict[str, Any]]:
    """
    全池 rollout + 打分 + 分桶。records = grpo_prompts_*.jsonl 的行。
    sampler 可注入（单测就塞假采样器，不碰 GPU）。
    """
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        gold = rec["gold"]
        comps = sampler(prompt, g)
        if len(comps) != g:
            raise ValueError(f"第 {i} 条采样器返回 {len(comps)} 份，期望 G={g}")
        res = score_group(prompt, comps, gold, g,
                          input_text=rec.get("input_text"))
        rows.append({
            "idx": i,
            "event_id": rec.get("event_id") or rec.get("meta", {}).get("event_id") or f"row_{i}",
            "k": res["k"],
            "bucket": bucket_label(res["k"], g),
            "hard_case_type": gold.get("hard_case_type"),
            "sentiment": gold.get("sentiment"),
            "relevance": gold.get("relevance"),
            "industry": rec.get("meta", {}).get("industry"),
            "totals": res["totals"],
            "flags": res["flags"],
        })
    return rows


def oracle_sampler(gold: Dict[str, Any]) -> Sampler:
    """
    干跑用的假采样器: 直接拿 gold 反构一条「标准答案」completion (重复 G 次)。
    subject 从 prompt 的 目标主体：【…】 反解（gold 契约里没有 subject 字段）。
    """

    def _prompt_subject(prompt: str) -> str:
        m = re.search(r"目标主体[：:]\s*【(.+?)】", prompt or "")
        return m.group(1) if m else ""

    def _sampler(prompt: str, g: int) -> List[str]:
        body = {
            "subject": _prompt_subject(prompt),
            "relevance": gold.get("relevance"),
            "sentiment": gold.get("sentiment"),
            "hard_case_type": gold.get("hard_case_type"),
            "risk_categories": gold.get("risk_categories") or [],
            "evidence": gold.get("evidence"),
        }
        # 造一条落在 40~65 token 理想带的 CoT（与 test_reward_fns 同一套校准手法）
        cot = ("Step1 主体居叙事中心故core。Step2 字面语义与真实取向相反判irony。"
               "Step3 风险归product与marketing，负面定negative。")
        comp = f"<|begin_of_thought|>{cot}<|end_of_thought|>" + json.dumps(body, ensure_ascii=False)
        return [comp] * g

    return _sampler
