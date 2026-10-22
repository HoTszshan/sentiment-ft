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
        --out data/grpo/grpo_prompts_probe.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

GOLD_FIELDS = ["sentiment", "relevance", "hard_case_type", "risk_categories", "evidence"]


def extract_gold(row: dict) -> dict:
    """从语料行解析出 5 字段 gold。优先取 output 内的 JSON，回退 meta。"""
    out = row.get("output", "")
    tail = ""
    if "<|end_of_thought|>" in out:
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

    meta = row.get("meta", {})
    gold = {
        "sentiment": parsed.get("sentiment") or meta.get("sentiment"),
        "relevance": parsed.get("relevance") or meta.get("relevance"),
        "hard_case_type": parsed.get("hard_case_type") or meta.get("hard_case_type"),
        "risk_categories": parsed.get("risk_categories", []),
        "evidence": parsed.get("evidence") or "",
    }
    if not isinstance(gold["risk_categories"], list):
        gold["risk_categories"] = []
    # 兜底: CoT 已被裁掉时，从 input 原文找一个 4~80 字的连续子串 (仍须严格子串)
    if not (4 <= len(gold["evidence"]) <= 80) or gold["evidence"] not in row.get("input", ""):
        gold["evidence"] = ""
    return gold


def build_prompt(row: dict) -> str:
    """
    与训练侧逐字同构的 prompt:
      system = instruction (含目标主体与行业风险词表)
      user   = input (待判原文)
    走原生 chat 模板时直接吃 instruction/input 即可；这里拼好是为了 rollout 零解析。
    """
    return (
        "<|im_start|>system\n{inst}<|im_end|>\n"
        "<|im_start|>user\n{inp}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ).format(inst=row.get("instruction", ""), inp=row.get("input", ""))


def main():
    ap = argparse.ArgumentParser(description="语料池 → GRPO {prompt, gold} 契约")
    ap.add_argument("--src", required=True, help="新生成的语料池 jsonl")
    ap.add_argument("--out", required=True, help="GRPO 探针数据集输出路径")
    ap.add_argument("--with_meta", action="store_true",
                    help="额外落 meta (industry/难点/极性) 便于分桶统计")
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
            item["input_text"] = row.get("input", "")
        if args.with_meta:
            item["meta"] = {
                "industry": row.get("meta", {}).get("industry"),
                "hard_case_type": gold["hard_case_type"],
                "sentiment": gold["sentiment"],
                "relevance": gold["relevance"],
                "provenance": "synthetic",
            }
        kept.append(item)

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
