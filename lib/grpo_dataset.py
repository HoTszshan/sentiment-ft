#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO 数据契约：读 JSONL、检查必需字段、展平为 TRL 可消费的标量列。

不自定义 Dataset 类；调用方用 datasets.Dataset.from_list(build_dataset_rows(records))。
保持 import 时不依赖 datasets / torch / trl，以便在无 GPU 环境体检数据。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from reward_fns import extract_input_text

# gold 的 5 个字段（**没有 subject** —— subject 由 rollout 时从 prompt 回显）
GOLD_FIELDS = ("sentiment", "relevance", "hard_case_type", "risk_categories", "evidence")


def load_records(path: str) -> List[Dict[str, Any]]:
    """读取 GRPO JSONL；保留训练入口对缺文件/空文件的拒启口径。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[FATAL] 找不到数据: {p}")
    with p.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise SystemExit(f"[FATAL] 数据为空: {p}")
    return records


def flatten_gold(gold: Dict[str, Any]) -> Dict[str, Any]:
    """gold dict → 标量列 (risk_categories 是 list，JSON 序列化成字符串)。"""
    if not isinstance(gold, dict):
        raise ValueError(f"gold 必须是 dict，拿到 {type(gold).__name__}")
    out = {}
    for f in GOLD_FIELDS:
        if f not in gold:
            raise ValueError(f"gold 缺字段 {f}")
    out["gold_sentiment"] = str(gold["sentiment"])
    out["gold_relevance"] = str(gold["relevance"])
    out["gold_hard_case_type"] = str(gold["hard_case_type"])
    out["gold_risk_json"] = json.dumps(gold["risk_categories"] or [], ensure_ascii=False)
    out["gold_evidence"] = str(gold["evidence"])
    return out


def rebuild_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    """标量列 → gold dict (打分时还原)。"""
    rk = row.get("gold_risk_json", "[]")
    if isinstance(rk, str):
        rk = json.loads(rk or "[]")
    return {
        "sentiment": row.get("gold_sentiment"),
        "relevance": row.get("gold_relevance"),
        "hard_case_type": row.get("gold_hard_case_type"),
        "risk_categories": list(rk),
        "evidence": row.get("gold_evidence"),
    }


def build_dataset_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 {prompt, gold, [input_text], [meta]} 展平成 TRL 训练行。

    input_text 供 reward 校验 evidence ⊂ 原文，不能依赖 TRL 传回的 prompt 形态。
    优先取数据行自带的 input_text；旧数据或 --two_field 才反解 prompt。
    """
    rows = []
    for i, rec in enumerate(records):
        prompt = rec.get("prompt")
        if not prompt:
            raise ValueError(f"第 {i} 行缺 prompt")
        input_text = rec.get("input_text")
        if not isinstance(input_text, str) or not input_text.strip():
            input_text = extract_input_text(prompt)
        rows.append({
            "prompt": prompt,
            "input_text": input_text,
            **flatten_gold(rec["gold"]),
        })
    return rows


def build_eval_baseline(records: Sequence[Dict[str, Any]], n: int = 8
                        ) -> List[Dict[str, Any]]:
    """从训练池按 (sentiment, hard_case_type) 去重抽 n 条，作 eval 熔断基线集。

    只保证类型覆盖面，不是独立泛化集——样本与训练池同源，eval 均值跌了说明训练
    可能坏了，不跌不代表变好。不足 n 条时返回实际数量，不补造样本。
    """
    picked: List[Dict[str, Any]] = []
    seen = set()
    for row in records:
        key = (row["gold"]["sentiment"], row["gold"]["hard_case_type"])
        if key not in seen:
            seen.add(key)
            picked.append(row)
        if len(picked) == n:
            break
    return picked


def write_eval_baseline(records: Sequence[Dict[str, Any]], target: str,
                        n: int = 8) -> int:
    """build_eval_baseline + 落盘，返回实际条数（供入口打印回显）。"""
    picked = build_eval_baseline(records, n)
    out = Path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in picked)
    return len(picked)
