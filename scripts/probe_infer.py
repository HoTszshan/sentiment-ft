#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探针实验 · 推理脚本 (Probe Inference)

作用: 加载 SFT 后的 LoRA Adapter，在验证集/隔离探针集上批量生成，
      产出可直接喂给 evaluate_sft.py 「模式 B」的预测文件 (pred.jsonl)。

生成 prompt 与训练侧逐字一致 (train_sft_lora.SFTWeightedDataset):
    <|im_start|>system\n{instruction}<|im_end|>\n
    <|im_start|>user\n{input}<|im_end|>\n
    <|im_start|>assistant\n

用法:
    python probe_infer.py \
        --base_model Qwen/Qwen3-4B-Instruct \
        --adapter_path output/probe_lora \
        --data_file data/mixed/sft_val.jsonl \
        --out output/pred_val.jsonl

产出文件: 每行 = 该样本模型的原始输出 (含 <|begin_of_thought|> ... JSON)，
          行序与 --data_file 完全对齐，供 evaluate_sft --gold_file/--pred_file 使用。
"""

import argparse
import json
import sys
import time
from pathlib import Path


PROMPT_TMPL = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main():
    ap = argparse.ArgumentParser(description="舆情精判 SFT 探针推理")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--adapter_path", default="", help="LoRA Adapter 目录，留空则用裸基座")
    ap.add_argument("--data_file", default="data/mixed/sft_val.jsonl")
    ap.add_argument("--out", default="output/pred_val.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="CoT~300 + JSON~150 + buffer；SOP 口径单次解码 65~105 tokens 之外给足余量")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全量")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.stderr.write("[FATAL] 缺少推理依赖: pip install torch transformers peft accelerate\n")
        raise SystemExit(2)

    if not Path(args.data_file).exists():
        raise SystemExit(f"[FATAL] 找不到数据文件: {args.data_file}")

    rows = [json.loads(l) for l in open(args.data_file, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print("=" * 70)
    print("探针推理")
    print(f"  基座    : {args.base_model}")
    print(f"  Adapter : {args.adapter_path or '(无 — 裸基座对照)'}")
    print(f"  数据    : {args.data_file} ({len(rows)} 条)")
    print(f"  产出    : {args.out}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                              padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"[OK] 已加载 LoRA Adapter: {args.adapter_path}")
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_tag = n_json = n_ev = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            prompt = PROMPT_TMPL.format(instruction=row["instruction"], input=row["input"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=0.9 if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=False)

            # 静态合规指标（四查的①②在推理侧可提前观测）
            if "<|begin_of_thought|>" in text and "<|end_of_thought|>" in text:
                n_tag += 1
            try:
                after = text.split("<|end_of_thought|>")[-1].lstrip()
                pred = json.loads(after[: after.rfind("}") + 1]) if "}" in after else None
                if isinstance(pred, dict):
                    n_json += 1
                    ev = str(pred.get("evidence", ""))
                    if ev and ev in row["input"]:
                        n_ev += 1
            except Exception:
                pass

            f.write(text.replace("\n", " ").strip() + "\n")
            f.flush()

            if i % 10 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] tag合规 {n_tag} | JSON合法 {n_json} | "
                      f"evidence 命中原文 {n_ev}", flush=True)

    print("=" * 70)
    print(f"[OK] 推理完成 {len(rows)} 条，耗时 {time.time() - t0:.1f}s")
    print(f"     CoT 标签闭合率 : {n_tag}/{len(rows)} ({n_tag/max(len(rows),1)*100:.1f}%)")
    print(f"     JSON 解析合法率: {n_json}/{len(rows)} ({n_json/max(len(rows),1)*100:.1f}%)")
    print(f"     evidence 原文命中: {n_ev}/{len(rows)} ({n_ev/max(len(rows),1)*100:.1f}%)")
    print(f"     预测文件: {out_path}")
    print("=" * 70)
    print("下一步: python evaluate_sft.py --gold_file "
          f"{args.data_file} --pred_file {args.out}")


if __name__ == "__main__":
    main()
