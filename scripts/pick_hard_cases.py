#!/usr/bin/env python3
"""
scripts/pick_hard_cases.py
从数据集中挑难例样本 (按 hard_case_type 过滤)，打印行号/类型/原文摘要。
TODO 清单 Step 4.5 (直观前后对比样张) 的选样工具。

用法:
  python3 scripts/pick_hard_cases.py --data_file data/mixed/sft_val.jsonl
  python3 scripts/pick_hard_cases.py --data_file data/train_synth_81.jsonl \
      --types irony,multi_subject --n 3
"""

import argparse
import json

DEFAULT_TYPES = ("irony", "multi_subject", "counterfactual")


def main():
    ap = argparse.ArgumentParser(description="按难例类型挑样本 (打印行号+摘要)")
    ap.add_argument("--data_file", required=True, help="jsonl 数据文件 (val 或全量领域集)")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES),
                    help=f"逗号分隔的 hard_case_type 列表 (默认 {','.join(DEFAULT_TYPES)})")
    ap.add_argument("--n", type=int, default=0, help="最多打印 N 条, 0=全部命中")
    args = ap.parse_args()

    want = tuple(t.strip() for t in args.types.split(",") if t.strip())
    hits = []
    for i, line in enumerate(open(args.data_file, encoding="utf-8")):
        if not line.strip():
            continue
        d = json.loads(line)
        gold = d.get("output") or d.get("gold") or {}
        # output 是 "CoT...</think>{JSON}" 形态时, hard_case_type 在 JSON 里 —— 兜底从 meta 取
        hc = gold.get("hard_case_type") if isinstance(gold, dict) else None
        if not hc:
            hc = (d.get("meta") or {}).get("hard_case_type")
        if hc in want:
            text = d.get("input_text") or d.get("input") or d.get("text") or ""
            hits.append((i, hc, text[:60].replace("\n", " ")))

    if not hits:
        print(f"未命中 {want} 类型 (共扫描 {args.data_file})")
        return
    for i, hc, text in hits[: args.n] if args.n else hits:
        print(f"  行号={i:3d}  类型={hc:<16s}  原文摘要: {text}")
    print(f"\n命中 {len(hits)} 条 (显示前 {min(len(hits), args.n) if args.n else len(hits)} 条)")


if __name__ == "__main__":
    main()
