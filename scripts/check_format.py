#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
格式自检（独立运行版）

对任意「基座 + adapter」组合跑一次格式 canary，不需要启动训练。
用于训练前摸底、训练后验收、或换 adapter 后单独检查。

用法:
    # 训练后验收
    python scripts/check_format.py \
        --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
        --adapter_path output/run_<时间戳>/sft_lora \
        --data_file data/mixed/sft_val.jsonl

    # 训练前摸底（不传 adapter，看基座原生水平）
    python scripts/check_format.py --base_model ... --data_file ...

指标含义与门槛见 README「步骤 3b · 格式 canary」。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import prompt_template as pt          # noqa: E402
import format_canary as fc            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="格式 canary 独立检查")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", default="", help="LoRA Adapter 目录，留空则测裸基座")
    ap.add_argument("--data_file", default="data/mixed/sft_val.jsonl")
    ap.add_argument("--n", type=int, default=0, help="取前 N 条，0=全量")
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--think_prefix", type=int, default=0, choices=(0, 1),
                    help="前缀预置，必须与训练时一致")
    ap.add_argument("--out", default="", help="把指标写成 JSON（可选）")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.stderr.write("[FATAL] 缺少依赖: pip install torch transformers peft\n")
        raise SystemExit(2)

    pt.set_think_prefix(bool(args.think_prefix))

    rows = [json.loads(l) for l in open(args.data_file, encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[: args.n]
    if not rows:
        raise SystemExit(f"[FATAL] 数据为空: {args.data_file}")

    print("=" * 66)
    print("格式 canary · 独立检查")
    print(f"  基座    : {args.base_model}")
    print(f"  Adapter : {args.adapter_path or '(无 — 裸基座)'}")
    print(f"  数据    : {args.data_file} ({len(rows)} 条)")
    print("=" * 66)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto")
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)

    metrics = fc.run_canary(model, tokenizer, rows, n=len(rows), max_new=args.max_new)

    print(f"\np(</think>)            : {metrics['p_close']:.3f}   (门槛 ≥0.5)")
    print(f"p(next|after</think>) : {metrics['p_next']:.3f}   (门槛 ≥0.5)")
    print(f"p(<|im_end|>)         : {metrics['p_stop']:.3f}   (门槛 ≥0.5)")
    print(f"greedy_ok             : {metrics['greedy_ok']}/{metrics['greedy_n']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n指标已落盘: {args.out}")


if __name__ == "__main__":
    main()
