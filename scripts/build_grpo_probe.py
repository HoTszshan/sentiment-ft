#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 探针数据构建：把「新生成的语料池」转成 GRPO 落盘契约 {prompt, gold}

唯一真理源: MVP/notes/舆情负面精判_GRPO数据准备实战手册.md §0.2「落盘样本契约」

契约要点:
  - 只有 prompt + gold，**不带 completion** (CoT 由策略 On-Policy 现场采样，算完即弃)
  - gold 5 字段，与 SFT 输出协议完全一致:
      sentiment / relevance / hard_case_type / risk_categories / evidence
      · evidence → 供 R_grounding 做原文子串硬断言
      · sentiment+risk+hard_case → 供 R_accuracy 判别主线
      · relevance → 参与校验但只给轻权重
  - 额外保留 input_text (从 prompt 同源落盘)：grounding 断言时无需反解析 prompt；
    **默认落盘**（评估手册缺口 16：训练侧与评测侧的「原文」必须同源）。
    走严格两字段交付时加 --two_field 去掉（只用于跨版本兼容演练）。
  - --with_meta 只控制 meta 的落盘，不再连带控制 input_text。

用法:
    python3 scripts/build_grpo_probe.py \
        --src data/grpo_probe_pool.jsonl \
        --out data/grpo_prompts_probe.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from task_template import SUBJECT_RE  # noqa: E402
import prompt_template as pt         # noqa: E402

GOLD_FIELDS = ["sentiment", "relevance", "hard_case_type", "risk_categories", "evidence"]

# ============================================================
# 1. 行字段工具 (兼容顶层 / meta 嵌套两种源形态)
# ============================================================

def _row_meta(row: dict) -> dict:
    """行内 meta：优先嵌套 meta，否则行本身就是扁平题面（industry 在顶层）。"""
    m = row.get("meta")
    return m if isinstance(m, dict) else {}


def _field(row: dict, key: str, default=None):
    """取字段：顶层优先，回退 meta（两种源形态通吃）。"""
    if key in row and row[key] not in (None, ""):
        return row[key]
    return _row_meta(row).get(key, default)


def _raw_text(row: dict) -> str:
    """待判【裸原文】。已对齐行 input 是包了壳的题面，裸原文在 meta.input_raw。"""
    m = _row_meta(row)
    if m.get("input_raw"):
        return m["input_raw"]
    inp = row.get("input") or ""
    # 兜底：从 user 侧题面里剥出原文（【输入文本】之后）
    marker = "【输入文本】"
    if marker in inp:
        return inp.split(marker, 1)[1].lstrip("\n")
    return inp


# ============================================================
# 2. 出题: gold 解析 + prompt 渲染
# ============================================================

def extract_gold(row: dict) -> dict:
    """从语料行解析出 5 字段 gold。优先取 output 内的 JSON，回退 meta。"""
    out = row.get("output", "")
    tail = ""
    if "</think>" in out:
        tail = out.split("</think>")[-1].strip()
    elif "<|end_of_thought|>" in out:
        tail = out.split("<|end_of_thought|>")[-1].strip()
    elif out.strip().startswith("{"):
        tail = out.strip()

    parsed = {}
    try:
        last = tail.rfind("}")
        if last != -1:
            parsed = json.loads(tail[: last + 1])
    except Exception:
        parsed = {}

    meta = _row_meta(row)
    # subject: 同优先级取 output JSON，回退 user 侧【目标主体】行 / target_subject 字段
    ui = row.get("input") or ""
    m = re.search(SUBJECT_RE, ui)
    subject = (parsed.get("subject")
               or (m.group(1).strip() if m else None)
               or _field(row, "target_subject"))
    gold = {
        "subject": subject,
        "sentiment": parsed.get("sentiment") or meta.get("sentiment"),
        "relevance": parsed.get("relevance") or meta.get("relevance"),
        "hard_case_type": parsed.get("hard_case_type") or meta.get("hard_case_type"),
        "risk_categories": parsed.get("risk_categories", []),
        "evidence": parsed.get("evidence") or "",
    }
    if not isinstance(gold["risk_categories"], list):
        gold["risk_categories"] = []
    # 兜底: CoT 已被裁掉时，从 input 原文找一个 4~80 字的连续子串 (仍须严格子串)
    if not (4 <= len(gold["evidence"]) <= 80) or gold["evidence"] not in _raw_text(row):
        gold["evidence"] = ""
    return gold


def build_prompt(row: dict, tokenizer=None) -> str:
    """
    和SFT共用一套prompt_template.render_prompt渲染：system为指令，user为主体+文本。
    禁止手动拼<|im_start|>，模板分叉会导致GRPO rollout分布和SFT不匹配，reward大面积掉底。
    有tokenizer用chat模板，无tokenizer自动降级手写拼接。
    """
    return pt.render_prompt(
        tokenizer,
        row.get("instruction", "") or "",
        row.get("input", "") or "",
    )


# ============================================================
# 3. 分层抽样 (hard_case_type × sentiment 按比例，小桶保底)
# ============================================================

def stratified_sample(items: list, n: int, seed: int = 42) -> list:
    """
    按 (hard_case_type, sentiment) 21 桶**按比例**分层抽样（最大余数法）。

      - 每个非空桶先保 1 条（小桶不被漏掉）；
      - 剩余额度按桶大小比例分配，余数从大到小补齐；
      - 桶内配额不超过桶自身条数。
    「先保 1 + 按比例」避免了大桶（explicit×neutral）吃光预算的退化。
    """
    import random
    rng = random.Random(seed)
    buckets: Dict[tuple, list] = {}
    for it in items:
        g = it["gold"]
        buckets.setdefault((g.get("hard_case_type"), g.get("sentiment")), []).append(it)
    for k in buckets:
        rng.shuffle(buckets[k])

    keys = sorted(buckets, key=str)
    total = sum(len(v) for v in buckets.values())
    n = min(n, total)

    # 1) 每桶保底 1 条（桶数不超过 n 时）
    quota = {k: (1 if len(buckets[k]) else 0) for k in keys}
    if sum(quota.values()) > n:
        # 桶数比 n 还多：按桶大小取前 n 个，其余 0
        for k in sorted(keys, key=lambda x: -len(buckets[x]))[n:]:
            quota[k] = 0
        keys = [k for k in keys if quota[k]]
    # 2) 剩余额度按比例分（最大余数法）
    remain = n - sum(quota.values())
    if remain > 0:
        share = {k: remain * len(buckets[k]) / total for k in keys}
        for k in keys:
            add = min(int(share[k]), len(buckets[k]) - quota[k])
            quota[k] += add
            remain -= add
        # 余数补齐：按「当前欠额 share-quota」最大者优先（每轮重算，防同一桶吃满）
        while remain > 0:
            cand = [k for k in keys if quota[k] < len(buckets[k])]
            if not cand:
                break
            k = max(cand, key=lambda x: (share[x] - quota[x], len(buckets[x]), str(x)))
            quota[k] += 1
            remain -= 1

    picked = []
    for k in keys:
        picked.extend(buckets[k][: quota[k]])
    picked.sort(key=lambda it: (it["gold"].get("hard_case_type") or "",
                                it["gold"].get("sentiment") or ""))
    return picked


# ============================================================
# 4. 主流程: 质检淘汰 → 组装落盘 → 统计报告
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="语料池 → GRPO {prompt, gold} 契约")
    ap.add_argument("--src", required=True, help="新生成的语料池 jsonl")
    ap.add_argument("--out", required=True, help="GRPO 探针数据集输出路径")
    ap.add_argument("--with_meta", action="store_true",
                    help="额外落 meta (industry/难点/极性) 便于分桶统计")
    ap.add_argument("--sample", type=int, default=None,
                    help="分层抽样条数（按 hard_case_type × sentiment 21 桶，每桶至少 1 条）；"
                         "不给则全量")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument("--two_field", action="store_true",
                    help="严格两字段交付：不落 input_text（默认落盘，供 grounding 原文同源）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[FATAL] 找不到语料池: {src}")

    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    for i, row in enumerate(rows, 1):
        gold = extract_gold(row)
        missing = [k for k in GOLD_FIELDS if not gold.get(k) and k != "risk_categories"]
        if missing:
            dropped.append((i, f"gold 缺字段 {missing}"))
            continue
        if not gold["evidence"]:
            dropped.append((i, "evidence 非原文合法子串 (4~80 字)"))
            continue
        item = {"prompt": build_prompt(row), "gold": gold}
        # input_text 默认落盘：训练侧 grounding 与评测侧原文同源，不必各自反解 prompt
        if not args.two_field:
            item["input_text"] = _raw_text(row)   # 必须是裸原文，grounding 断言才成立
        if args.with_meta:
            item["meta"] = {
                "industry": _field(row, "industry"),
                "hard_case_type": gold["hard_case_type"],
                "sentiment": gold["sentiment"],
                "relevance": gold["relevance"],
                "provenance": "synthetic",
            }
        kept.append(item)

    if args.sample and args.sample < len(kept):
        kept = stratified_sample(kept, args.sample, args.seed)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- 分布统计 ----
    n = len(kept)
    print("=" * 70)
    print(f"GRPO 探针数据构建完成")
    print(f"  输入语料池 : {len(rows)} 条 -> {src}")
    print(f"  可用 prompt: {n} 条")
    print(f"  丢弃       : {len(dropped)} 条")
    for i, why in dropped[:5]:
        print(f"     #{i}: {why}")
    print(f"  输出       : {out_path}")
    print("=" * 70)
    if n:
        meta = "meta" if kept[0].get("meta") else None
        if not meta:
            print("  (两字段模式：不输出分桶统计，加 --with_meta 可看)")
        for name, key in [("难点", "hard_case_type"), ("情感", "sentiment"), ("语篇", "relevance")]:
            if not meta:
                break
            c = Counter(it[meta][key] for it in kept)
            print(f"  {name}: " + "  ".join(f"{k}:{v}" for k, v in
                                           sorted(c.items(), key=lambda x: -x[1])))
        if meta:
            c = Counter(it[meta]["industry"] for it in kept)
            print("  行业: " + "  ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: -x[1])))
        print("=" * 70)
        print("⚠️ 这是探针级 prompt+gold (构造区)。正式 grpo_prompts_3000.jsonl 需")
        print("   阶段一生产日志筛题 + 阶段二金标 + 阶段三 G=4 难度筛选后才能定稿。")
        print("⚠️ 无 completion 是设计：CoT 由策略 On-Policy 现场采样，算完组内优势即弃。")
    else:
        print("[WARN] 0 条可用 —— 检查语料池是否落盘")
        sys.exit(1)


if __name__ == "__main__":
    main()
