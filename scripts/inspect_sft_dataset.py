#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据模板预览器 (Inspect SFT Dataset)

回答一个问题：**这批数据套进模板之后，模型到底看到什么、学什么？**

四层递进打印：
  ① 原始三元组   instruction / input / output 原文
  ② messages     system / user / assistant 结构化视图
  ③ 模板渲染     apply_chat_template(tokenize=False) 的逐字结果（<|im_start|> 等可见）
  ④ token 视图   每个 token 的 label 掩码 + loss 权重 + 字段 span 表

为什么单独成脚本：
  训练只看 loss 曲线，看不出「前缀掩码是不是切错了」「FIELD_W 是不是落到了骨架而不是枚举值」。
  这些错误不会让 loss 变 NaN，只会让效果莫名变差，几十步训练后才发现已经太晚。

用法:
  # 无 GPU / 无 transformers：字符级近似看结构与权重分布（任何机器都能跑）
  python3 scripts/inspect_sft_dataset.py \
      --data_file data/probe_test_sft_train_synth.jsonl --num_samples 2

  # 接了 tokenizer：看真实 token 边界与真实长度
  python3 scripts/inspect_sft_dataset.py \
      --data_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl \
      --model_name_or_path /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --num_samples 2 --show_tokens

  # 只做 parity 冒烟（训练前 / e2e 里跑一次），不一致直接非 0 退出
  python3 scripts/inspect_sft_dataset.py \
      --data_file data/probe_test_sft_train_synth.jsonl \
      --model_name_or_path /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --check_parity
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import prompt_template as pt          # noqa: E402
import sft_dataset as sd              # noqa: E402
from sft_weight_planner import (     # noqa: E402
    FIELD_WEIGHTS, POC_STAGE, USE_FIELD_WEIGHTS, USE_SKELETON_MASK,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="SFT 数据模板预览：原始字段 → messages → 渲染文本 → token/权重")
    ap.add_argument("--data_file", required=True, help="SFT jsonl（instruction/input/output）")
    ap.add_argument("--model_name_or_path", default="",
                    help="tokenizer 来源；留空则用字符级近似（1 字符=1 token）")
    ap.add_argument("--max_seq_length", type=int, default=2560)
    ap.add_argument("--num_samples", type=int, default=1, help="打印条数（0=全量）")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_print_tokens", type=int, default=400,
                    help="④ token 视图每条最多打印多少 token")
    ap.add_argument("--show_tokens", action="store_true",
                    help="按 token 字符串打印（需 tokenizer），否则打印 id")
    ap.add_argument("--show_ignored", action="store_true", help="额外说明不参与 loss 的区段")
    ap.add_argument("--color", action="store_true", help="ANSI 绿色高亮参与 loss 的 token")
    ap.add_argument("--no_strict_parity", action="store_true",
                    help="关闭 chat 渲染与手写模板的字节比对（**只调试用**，会静默放过漂移）")
    ap.add_argument("--check_parity", action="store_true",
                    help="只跑 parity 校验并退出：一致 rc=0，不一致 rc=3")
    ap.add_argument("--only_stats", action="store_true", help="只打印长度分布")
    ap.add_argument("--export_dir", default="", help="把每条样本的 token/权重导出为 JSON")
    ap.add_argument("--local_files_only", action="store_true")
    return ap.parse_args()


def load_tokenizer(args):
    if not args.model_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[WARN] 未安装 transformers，退化为字符级近似（1 字符 = 1 token）",
              file=sys.stderr)
        return None
    try:
        tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    except Exception as e:
        print(f"[WARN] tokenizer 加载失败 ({type(e).__name__}: {e})，退化为字符级近似",
              file=sys.stderr)
        return None
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def print_header(args, tokenizer):
    unit = "token" if tokenizer is not None else "char (近似)"
    print("=" * 78)
    print("SFT 数据模板预览")
    print(f"  数据       : {args.data_file}")
    print(f"  分词器     : {args.model_name_or_path or '(无 — 字符级近似)'}   单位={unit}")
    print(f"  max_seq_len: {args.max_seq_length}")
    print(f"  POC 阶段   : {POC_STAGE}  (骨架掩码={'开' if USE_SKELETON_MASK else '关'} / "
          f"字段加权={'开' if USE_FIELD_WEIGHTS else '关'})")
    if USE_FIELD_WEIGHTS:
        print(f"  FIELD_W    : hard_case={FIELD_WEIGHTS['hard_case_type']}x "
              f"sentiment={FIELD_WEIGHTS['sentiment']}x relevance={FIELD_WEIGHTS['relevance']}x "
              f"risk={FIELD_WEIGHTS['risk_categories']}x evidence={FIELD_WEIGHTS['evidence']}x "
              f"CoT={FIELD_WEIGHTS['cot']}x")
    print("=" * 78)


def run_parity_check(ds, args) -> int:
    rows = ds.samples
    limit = max(args.num_samples, 0) or len(rows)
    if not ds.uses_chat:
        print("[NOTE] 无 chat 模板可比对，parity 不适用（手写外壳，无漂移风险）")
        return 0
    try:
        rep = pt.assert_parity(ds.tokenizer, rows, limit=limit)
    except pt.TemplateParityError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        return 3
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n[OK] parity 通过（{rep['checked']} 条逐字节一致）")
    return 0


def export_sample(out_dir, idx, item, tokenizer):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"sample_{idx:06d}.json"
    payload = {
        "input_ids": item["input_ids"],
        "labels": item["labels"],
        "loss_weights": item["loss_weights"],
        "prompt_len": item["prompt_len"],
        "truncated": item["truncated"],
        "spans": [[s, e, k, w] for s, e, k, w in item["spans"]],
    }
    if tokenizer is not None:
        payload["tokens"] = tokenizer.convert_ids_to_tokens(item["input_ids"])
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    args = parse_args()
    if not Path(args.data_file).exists():
        raise SystemExit(f"[FATAL] 找不到数据文件: {args.data_file}")

    tokenizer = load_tokenizer(args)
    print_header(args, tokenizer)

    try:
        ds = sd.SFTConversationDataset(
            args.data_file, tokenizer,
            max_len=args.max_seq_length,
            strict_parity=not args.no_strict_parity,
        )
    except pt.TemplateParityError as e:
        # parity 失败要给出可操作的下一步，而不是一串堆栈
        print(f"\n[FATAL] {e}", file=sys.stderr)
        print("\n处理方式（三选一）:\n"
              "  1) 该基座模板与历史手写模板不同（如 thinking 版会在 assistant 段插入空思维块）:\n"
              "       确认后去掉 tokenizer 的 chat_template 走手写外壳，或改用与训练一致的基座\n"
              "  2) 只想看看数据长什么样、不训练:\n"
              "       传 --no_strict_parity\n"
              "  3) 模板本当一致却报错 → 是 tokenizer 或 prompt_template 的 bug，先修再训",
              file=sys.stderr)
        return 3

    if args.check_parity:
        return run_parity_check(ds, args)

    if ds.parity:
        print(f"[OK] parity: {ds.parity['checked']} 条 chat 渲染 == 手写模板（逐字节）")
    elif not ds.uses_chat:
        print("[NOTE] 手写外壳：无 chat 模板可比对")

    if args.only_stats:
        print(json.dumps(ds.stats(), ensure_ascii=False, indent=2))
        return 0

    n = len(ds)
    end = n if args.num_samples <= 0 else min(n, args.start + args.num_samples)
    start = max(0, min(args.start, n))

    for i in range(start, end):
        print(ds.preview(
            i,
            show_tokens=args.show_tokens,
            show_ignored=args.show_ignored,
            color=args.color,
            max_tokens=args.max_print_tokens,
        ))
        if args.export_dir:
            item = ds.build_item(ds.samples[i])
            p = export_sample(args.export_dir, i, item, tokenizer)
            print(f"[OK] 导出: {p}")

    print(json.dumps(ds.stats(), ensure_ascii=False, indent=2))
    if tokenizer is None:
        print("\n⚠️ 字符级近似：长度与 token 边界不准，只用于看结构与权重分布。"
              "真 token 口径请带 --model_name_or_path。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
