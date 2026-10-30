#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 探针核心 · 纯逻辑 (零 torch/trl 依赖，可独立单测)

一句话: 训练前给候选数据「试水」——每道题让模型答 4 次，按答对次数
筛出适合 GRPO 训练的题（半会不会的才留得下）。

  答对 0 次 → 太难，扔 (全错组无梯度，白烧卡)
  答对 1~3 次 → 留下进训练池 (有对有错才有训练信号)
  答对 4 次 → 太简单，扔 (全对组方差为 0)

入口: run_probe(records, sampler) 遍历 jsonl 全池；
      score_group 批改单题；oracle_sampler 是免模型的联调作弊器。

原为 scripts/run_grpo_probe.py 的一部分，因 train_grpo.py 复用其中的
completion 形态兼容函数而抽出到 lib/ —— CLI 脚本不再被当成库 import。

真理源:
  MVP/notes/舆情负面精判_GRPO数据准备实战手册.md §四 难度分流 (只留 k∈{1,2,3})
  MVP/notes/GRPO训练实战手册.md §1.3 超参 / §1.4 闭环
"""

import json
import os
import re

from task_template import SUBJECT_RE
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_fns import extract_input_text, split_completion           # noqa: E402
from grpo_common import (                                              # noqa: E402
    bucket_label, k_of, score_one,
)

# 采样器契约: (prompt, g) -> g 条 completion 文本
Sampler = Callable[[str, int], List[str]]

# 批量采样器契约: ([prompt...], g) -> List[List[g 条 completion]]，与 records 等长同序。
# vLLM 批量进队专用：把生成与打分解耦，生成一次跑完，打分循环零改动。
BatchSampler = Callable[[Sequence[str], int], List[List[str]]]


def run_probe_batch(records: Sequence[Dict[str, Any]], batch_sampler: BatchSampler,
                    g: int = 4) -> List[Dict[str, Any]]:
    """
    run_probe 的批量快路径: 所有 prompt 一次性进队 vLLM（连续批处理吃满吞吐），
    生成完毕后按题切 G 份复用 score_group，打分与逐题行结构与 run_probe 完全一致。
    """
    prompts = [rec["prompt"] for rec in records]
    groups = batch_sampler(prompts, g)
    if len(groups) != len(records):
        raise ValueError(f"批量采样器返回 {len(groups)} 组，期望 {len(records)} 组")
    # 复用单题路径: 把每组 completions 包成逐题 sampler，行结构/回显逻辑一处维护
    it = iter(groups)
    return run_probe(records, lambda _prompt, n: next(it)[:n], g=g)


def export_pool(records: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]],
                g: int, out_path: str) -> Dict[str, int]:
    """
    按探针结果导出 GRPO 训练池: 只留 k∈{1..g-1} 的原题行 (k=0/k=g 组内零方差, RL 无梯度)。
    行对齐用 rows[].idx（探针数据行无稳定 event_id，idx 是 records 的下标，恒可靠）。
    返回统计 {kept, dropped_k0, dropped_kmax, total}; out_path 为空串则只统计不落盘。
    """
    keep_idx = {r["idx"] for r in rows if 1 <= r["k"] <= g - 1}
    kept = [rec for i, rec in enumerate(records) if i in keep_idx]
    stat = {"kept": len(kept), "dropped_k0": sum(1 for r in rows if r["k"] == 0),
            "dropped_kmax": sum(1 for r in rows if r["k"] == g), "total": len(records)}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stat


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
        # 逐条标签级对错 + 各 reward 分量 (调 reward_fns 权重时逐轨迹看哪一维吃亏)
        flags.append({
            "short_circuit": d.get("short_circuit"),
            "sentiment_ok": bool(d.get("sentiment_ok")),
            "risk_ok": bool(d.get("risk_ok")),
            "hard_case_ok": bool(d.get("hard_case_ok")),
            "relevance_ok": bool(d.get("relevance_ok")),
            "accuracy_pass": bool(d.get("accuracy_pass")),
            "miss_severe": bool(d.get("miss_severe")),
            "false_alarm": bool(d.get("false_alarm")),
            # 四维 reward 分量 (门禁短路时 r_accuracy/r_length 恒 0)
            "r_format": r.get("r_format"),
            "r_grounding": r.get("r_grounding"),
            "r_accuracy": r.get("r_accuracy"),
            "r_length": r.get("r_length"),
            # 模型原始预测 (与 gold 逐字段对看; 门禁短路时为 None)
            "pred": d.get("pred"),
            "parse_flags": split_completion(comp)[2],
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
    n = len(records)
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
            # 调 reward 用原始材料: gold 全字段 + 原文 + G 条 completion 原文,
            # 配合 flags[].pred / r_* 分量逐轨迹定位「哪一维 reward 咬掉了这条」
            "gold": gold,
            "input_text": res["input_text"],
            "completions": list(comps),
        })
        # 逐条回显：每题 G 份生成是最耗时的部分，无回显时像卡死。
        # 尾部摘要 = G 条轨迹的标签对错计数，log 里直接看出这题各字段战绩
        # (明细仍以 rows.jsonl 的 flags[].pred / r_* 为准)
        fg = res["flags"]
        n_pass = sum(1 for f in fg if f["accuracy_pass"])
        n_gate = sum(1 for f in fg if f["short_circuit"])
        label_sum = {name: sum(1 for f in fg if f[name])
                     for name in ("sentiment_ok", "risk_ok",
                                  "hard_case_ok", "relevance_ok")}
        rows[-1]["_echo"] = (f"k={res['k']} ({rows[-1]['bucket']})  "
                             f"判对{n_pass}/{g} 门禁短路{n_gate}  "
                             + " ".join(f"{n}×{ln}" for ln, n in label_sum.items()))
        print(f"[PROBE {i+1}/{n}] {rows[-1]['_echo']}", flush=True)
    return rows


def oracle_sampler(gold: Dict[str, Any]) -> Sampler:
    """
    干跑用的假采样器: 直接拿 gold 反构一条「标准答案」completion (重复 G 次)。
    subject 优先读 gold["subject"]（build_grpo_probe 已落盘），缺了再从
    prompt 的 user 侧【目标主体】行反解（兼容旧格式「目标主体：【…】」）。
    """

    def _prompt_subject(prompt: str, gold: Dict[str, Any]) -> str:
        got = gold.get("subject")
        if got:
            return str(got)
        m = (re.search(SUBJECT_RE, prompt or "")
             or re.search(r"目标主体[：:]\s*【(.+?)】", prompt or ""))
        return m.group(1).strip() if m else ""

    def _sampler(prompt: str, g: int) -> List[str]:
        body = {
            "subject": _prompt_subject(prompt, gold),
            "relevance": gold.get("relevance"),
            "sentiment": gold.get("sentiment"),
            "hard_case_type": gold.get("hard_case_type"),
            "risk_categories": gold.get("risk_categories") or [],
            "evidence": gold.get("evidence"),
        }
        # 造一条落在理想带的 CoT（2026-10-27 阈值重设 120/200/320 后 ≈138 token）
        cot = ("Step1 主体识别：通读全文，叙事焦点始终落在广发证券身上，主体相关度判 core。"
               "Step2 语义分析：字面用「稳赚不赔」的褒义表述，但紧跟「才一天就亏了3000多」的事实反转，"
               "典型反讽修辞，难例类型判 irony。Step3 风险归因：亏损直接指向理财产品表现归 product，"
               "「稳赚不赔」的宣传口径涉嫌误导归 marketing。综合判 negative。")
        comp = f"<think>{cot}</think>" + json.dumps(body, ensure_ascii=False)
        return [comp] * g

    return _sampler
