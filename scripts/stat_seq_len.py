#!/usr/bin/env python3
"""
scripts/stat_seq_len.py
训练数据序列长度统计：按 domain / anchor 分账，输出 prompt / output / total 的
p50 / p95 / p99 / max 分位数，定位 OOM 长样本来源。

用法（AutoDL，需 transformers 已装 + 基座路径有 tokenizer）:
  python3 scripts/stat_seq_len.py \
      --train_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl \
      --tokenizer /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --top_n 5

无 tokenizer 环境（任何机器）退化为字符级近似（--no_tokenizer 自动触发）:
  python3 scripts/stat_seq_len.py --train_file data/train_synth_81.jsonl

口径说明:
  - prompt = <|im_start|>system...assistant\n 模板 + instruction + input（前缀掩码区）
  - output = assistant 输出（CoT + JSON）+ <|im_end|>
  - total  = prompt + output，与训练 max_seq_length 同口径（含模板，不含 padding）
  - 按 meta.provenance == public_anchor 分 anchor / domain 两桶，直接回答
    「长 token 是领域样本还是锚点带来的」
"""

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

# 外壳与训练/推理同源；长度口径必须与实际喂进模型的序列一致，否则 p99 定的 max_len 是假的
import prompt_template as pt  # noqa: E402


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _bucket(row):
    meta = row.get("meta") or {}
    if meta.get("provenance") == "public_anchor" or \
            meta.get("hard_case_type") == "general_anchor":
        return "anchor"
    return "domain"


def _report(tag, totals, prompts, outputs, unit, top_rows, top_n):
    t = sorted(totals)
    p = sorted(prompts)
    o = sorted(outputs)
    print(f"\n--- {tag} ({len(t)} 条, 单位={unit}) ---")
    print(f"  {'':8s}{'p50':>8s}{'p95':>8s}{'p99':>8s}{'max':>8s}")
    for name, v in (("prompt", p), ("output", o), ("total", t)):
        print(f"  {name:8s}{_pct(v,50):>8d}{_pct(v,95):>8d}{_pct(v,99):>8d}{v[-1]:>8d}")
    if top_rows:
        top_sorted = sorted(top_rows, key=lambda x: -x[0])
        print(f"  Top{top_n} 最长样本:")
        for tok_n, idx, bucket, meta in top_sorted[:top_n]:
            hc = meta.get("hard_case_type", "?")
            src = meta.get("source", meta.get("provenance", "?"))
            print(f"    #{idx} {bucket:6s} {tok_n:>6d} {unit}  "
                  f"hard_case={hc}  source={src}")


def main():
    ap = argparse.ArgumentParser(description="训练数据序列长度统计（domain/anchor 分账）")
    ap.add_argument("--train_file", required=True, help="训练集 jsonl（建议混合后的 sft_train.jsonl）")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer/基座路径；不给或加载失败则退化为字符级近似")
    ap.add_argument("--top_n", type=int, default=5, help="打印最长 Top-N 样本索引")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train_file, encoding="utf-8") if l.strip()]
    print(f"数据: {args.train_file}  共 {len(rows)} 条")

    encode = None
    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            encode = lambda s: len(tok.encode(s, add_special_tokens=False))
            unit = "token"
        except Exception as e:
            tok = None
            print(f"⚠️ tokenizer 加载失败 ({e})，退化为字符级近似 (1 汉字≈1.4 token)")
    if encode is None:
        encode = lambda s: len(s or "")
        unit = "char"

    buckets = {}
    for i, r in enumerate(rows):
        b = _bucket(r)
        # 与训练/推理同源：chat 模板可用就用 chat（与 Dataset 同一渲染路径），否则手写
        prompt = pt.render_prompt(tok if args.tokenizer else None,
                                  r.get("instruction", ""),
                                  r.get("input", ""))
        output = (r.get("output", "") or "") + pt.ASSISTANT_SUFFIX
        d = buckets.setdefault(b, {"p": [], "o": [], "t": [], "top": []})
        np_, no, nt = encode(prompt), encode(output), encode(prompt + output)
        d["p"].append(np_)
        d["o"].append(no)
        d["t"].append(nt)
        d["top"].append((nt, i, b, r.get("meta") or {}))

    # total 降序拿 Top-N（按 token 实测或字符近似均可）
    top_all = sorted((t for d in buckets.values() for t in d["top"]),
                     key=lambda x: -x[0])

    for b in ("domain", "anchor"):
        if b in buckets:
            d = buckets[b]
            _report(f"{b}", d["t"], d["p"], d["o"], unit, d["top"], args.top_n)

    if "domain" in buckets and "anchor" in buckets:
        dd, da = buckets["domain"], buckets["anchor"]
        print(f"\n--- 归因速读 ---")
        print(f"  domain max = {max(dd['t'])} {unit}, anchor max = {max(da['t'])} {unit}")
        if max(da["t"]) > max(dd["t"]):
            print("  → 锚点样本更长: OOM 长尾主要来自 anchor 分账")
        else:
            print("  → 领域样本更长: OOM 长尾主要来自 domain 分账")

    # 全局 Top-N（不分账）
    _report("全量 Top-N（不分 domain/anchor）",
            [t for d in buckets.values() for t in d["t"]],
            [x for d in buckets.values() for x in d["p"]],
            [x for d in buckets.values() for x in d["o"]],
            unit, top_all, args.top_n)

    if unit == "char":
        print("\n⚠️ 以上为字符级近似。真 token 口径请在 AutoDL 上带 --tokenizer 重跑。")


if __name__ == "__main__":
    main()
