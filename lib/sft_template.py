#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 指令模板 · 唯一真理源

训练数据 (align_sft_data.py)、推理探针 (probe_infer.py)、未来评测统一 import 此处，
保证「训练模板 = 推理模板逐字一致」。

模板结构（用户定稿版，2026-10-23）：
  - system 侧 instruction = 角色 + 任务目标 + 行业风险体系 + 输出规范（含 CoT 两步协议）
  - user 侧 = 【目标主体】{subject} + 【输入文本】{input}   ← 主体槽位在 user 侧
  - 枚举值一律从 schema_contract 常量插值，绝不手抄第二份 (单一真理源铁律)

⚠️ 主体不再写在 instruction 里（原「目标主体：【…】」格式已废弃）：
   grpo_probe_core / rubric_check 若需反解 subject，应从 user 侧「【目标主体】」
   行取（见 SUBJECT_RE），或直接读数据 meta.target_subject 字段。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schema_contract as sc  # noqa: E402
from corpus_schema import INDUSTRY_SCHEMAS, INDUSTRY_CN, RISK_KEY_CN  # noqa: E402

# user 侧主体行正则（原 instruction 内「目标主体：【…】」的迁移版）
SUBJECT_RE = r"【目标主体】\s*(.+)"

system_prompt_template = """
你是专业且客观严谨的舆情分析专家，精通【特定主体情感分类】与【风险事实抽取】。

【任务目标】
针对特定主体进行舆情精判，严格基于文本事实抽取风险，禁止主观脑补。

【行业风险体系：{industry_cn}({industry})】
{industry_risks_categories}

【输出规范】
1. 先在 <think>...</think> 内按 Step 1/Step 2/Step 3 进行结构化推理（主体相关性、困难类型、风险与情感）。
2. </think> 后直接输出 JSON 结果（严禁使用 ```json 和 ``` 包裹）。
3. JSON 字段必须严格使用以下英文枚举值：
   - "subject": (string) 目标主体。
   - "relevance": {relevance_enums} (对应: 核心/顺带提及/对比陪衬)
   - "sentiment": {sentiment_enums} (基于真实意图)
   - "hard_case_type": {hard_case_enums} (若命中多个，严格按此降序优先级输出：irony > metaphor > multi_subject > comparative > counterfactual > fact_implicit > explicit)
   - "risk_categories": (array) 仅限提取上方【行业风险体系】中命中的英文大类名。若非 negative 或无风险，必填 []。
   - "evidence": (string) 必须直接复制原文片段作为证据，长度 {evidence_min_len} 到 {evidence_max_len} 字。无则复述与主体最相关的原文片段，不得填空串。
"""

user_prompt_template = """
【目标主体】{subject}
【输入文本】
{input}
"""


def render_risk_categories(industry: str) -> str:
    """渲染【行业风险体系】列表：每行 `大类en(大类cn): 词表`。"""
    lines = []
    for key, vocab in INDUSTRY_SCHEMAS[industry].items():
        cn = RISK_KEY_CN.get(key, key)
        lines.append(f"- {key} ({cn}): {vocab}")
    return "\n".join(lines)


def _enum(items) -> str:
    return " | ".join(f'"{v}"' for v in items)


def build_instruction(industry: str) -> str:
    """
    system 侧 instruction。主体/原文槽位在 user 侧（build_user_input），
    因此 instruction 对同行业全部样本逐字一致 ——利于模型区分「协议」与「题面」。
    """
    cn = INDUSTRY_CN.get(industry, industry)
    hard_case_ordered = ["irony", "metaphor", "multi_subject", "comparative",
                         "counterfactual", "fact_implicit", "explicit"]

    return system_prompt_template.format(
        industry_cn=cn,
        industry=industry,
        industry_risks_categories=render_risk_categories(industry),
        relevance_enums=_enum(sorted(sc.VALID_RELEVANCE)),
        sentiment_enums=_enum(sorted(sc.VALID_SENTIMENT)),
        hard_case_enums=_enum(sorted(hard_case_ordered)),
        evidence_min_len=sc.EVIDENCE_MIN_LEN,
        evidence_max_len=sc.EVIDENCE_MAX_LEN,
    ).strip()


def build_user_input(subject: str, text: str) -> str:
    """user 侧题面：【目标主体】+【输入文本】。"""
    return user_prompt_template.format(subject=subject, input=text).strip()


def subject_of(row: dict) -> str:
    """行内主体来源：优先 output gold 的 subject，兜底 target_subject 字段。"""
    raw = (row.get("output") or "").strip()
    tail = raw.split(sc.THOUGHT_END, 1)[1] if sc.THOUGHT_END in raw else raw
    try:
        obj = json.loads(tail.strip())
        if isinstance(obj, dict) and obj.get("subject"):
            return str(obj["subject"])
    except (json.JSONDecodeError, ValueError):
        pass
    return str(row.get("target_subject") or "")


def build_example(row: dict) -> dict:
    """源行 → {instruction, input, output} 三元组（训练/推理同构）。"""
    return {
        "instruction": build_instruction(row["industry"]),
        "input": build_user_input(subject_of(row), row.get("input") or ""),
        "output": row.get("output") or "",
    }
