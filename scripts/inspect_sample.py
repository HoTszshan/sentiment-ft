#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单条样本透视：一条数据 → 训练格式全貌 → 掩码形态 → 模型真实生成。

四段输出，调试「训练侧到底在学什么」用:
  ① 原始数据      —— instruction / input / output 原文
  ② 训练格式全文   —— chat 模板渲染后的 full_text（与 sft_dataset 逐字节同源）
  ③ 掩码形态       —— prompt 段（labels=-100，不学） vs output 段（参与 loss）
                      + POC 阶段权重注记（POC2 骨架 0 / POC3 字段加权）
  ④ 模型推理       —— 用同一 prompt 让模型真实生成（默认 greedy），
                      与金标 output 并排对比

用法:
  # 只看格式与掩码（不加载模型，秒级）
  python3 scripts/inspect_sample.py --data_file <train.jsonl> --line 12 --stage POC2

  # 连推理一起（加载模型，~1 分钟）
  python3 scripts/inspect_sample.py --data_file <train.jsonl> --line 12 --stage POC2 \
      --model /root/autodl-tmp/models/sft_senti/merged/R6a01_SFT-bf16 --backend vllm

  # few-shot 预览（T2 场景：看 few-shot 插进去后 prompt 长什么样）
  python3 scripts/inspect_sample.py --data_file <one.jsonl> --few_shot_file data/few_shot_examples.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import prompt_template as pt  # noqa: E402
from sft_dataset import IGNORE_INDEX, char_level_encode  # noqa: E402


def _hr(title: str) -> str:
    return f"\n{'=' * 26} {title} {'=' * 26}"


def _poc_note(stage: str) -> str:
    from sft_weight_planner import plan_token_weights
    note = {"POC1": "全 token 等权 1.0",
            "POC2": "骨架 token 权重 0（JSON 死模板不学）",
            "POC3": "hard_case 4.0 / sent·rel 3.0 / risk 2.0 / evidence 0.4 / 骨架 0"}
    return note.get(stage, "（未知阶段，按 POC1 理解）")


def main():
    ap = argparse.ArgumentParser(description="单条样本透视：格式 → 掩码 → 推理")
    ap.add_argument("--data_file", required=True, help="JSONL 数据文件")
    ap.add_argument("--line", type=int, default=1, help="第几行 (1-based)")
    ap.add_argument("--stage", default="POC2", choices=["POC1", "POC2", "POC3"],
                    help="POC 阶段（决定权重注记与掩码行为说明）")
    ap.add_argument("--model", default="", help="模型目录（给了才跑第④段推理）")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--few_shot_file", default="", help="T2 few-shot 预览（JSONL）")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data_file, encoding="utf-8") if l.strip()]
    if not (1 <= args.line <= len(rows)):
        raise SystemExit(f"--line 越界: 文件共 {len(rows)} 行")
    row = rows[args.line - 1]

    instruction = row.get("instruction", "") or ""
    input_text = row.get("input", "") or ""
    output_text = row.get("output", "") or ""

    few_shot = None
    if args.few_shot_file:
        few_shot = [json.loads(l) for l in open(args.few_shot_file, encoding="utf-8")
                    if l.strip()]

    # ---- 与 sft_dataset 同源渲染 ----
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.model or "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
        encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        prompt_text, full_text = pt.render_full(tok, instruction, input_text, output_text)
        if few_shot:
            prompt_text = pt.render_prompt(tok, instruction, input_text, few_shot=few_shot)
            full_text = prompt_text + pt._strip_think_prefix(output_text) + pt.ASSISTANT_SUFFIX
        has_tok = True
    except ImportError:
        tok = None
        encode = char_level_encode
        prompt_text = pt.render_prompt(None, instruction, input_text)
        full_text = prompt_text + pt._strip_think_prefix(output_text) + pt.ASSISTANT_SUFFIX
        has_tok = False

    # ---- ① 原始数据 ----
    print(_hr("① 原始数据"))
    print(f"[instruction]\n{instruction}")
    print(f"\n[input]\n{input_text}")
    print(f"\n[output · 金标]\n{output_text}")

    # ---- ② 训练格式全文 ----
    print(_hr("② 训练格式全文（chat 渲染后，与训练侧逐字节同源）"))
    print(full_text)

    # ---- ③ 掩码形态 ----
    print(_hr(f"③ 掩码形态（stage={args.stage}）"))
    print(f"权重口径: {_poc_note(args.stage)}")
    print(f"总 token 数: {len(encode(full_text))}  "
          f"(prompt 段 {len(encode(prompt_text))} / output 段 {len(encode(full_text)) - len(encode(prompt_text))})")
    if has_tok:
        prompt_ids = encode(prompt_text)
        full_ids = encode(full_text)
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX
        n_learn = sum(1 for x in labels if x != IGNORE_INDEX)
        print(f"labels=-100 (不学) : prompt 段 {prompt_len} tokens")
        print(f"参与 loss          : output 段 {n_learn} tokens")
        print(f"\n[掩码可视化] ▪=学习(-100 之外) ·=不学(-100)")
        vis = "".join("▪" if lab != IGNORE_INDEX else "·" for lab in labels)
        for i in range(0, len(vis), 80):
            print(f"  {vis[i:i + 80]}")
        print(f"\n[output 段 token 还原]（这是模型真正要学的全部内容）:")
        learn_ids = [lab for lab in labels if lab != IGNORE_INDEX]
        print(repr(tok.decode(learn_ids)))
    else:
        print("(未安装 transformers，按字符级近似展示)")

    # ---- ④ 模型推理 ----
    if not args.model:
        print(_hr("④ 模型推理（未指定 --model，跳过）"))
        return

    print(_hr(f"④ 模型推理（{args.backend}, greedy, max_new_tokens={args.max_new_tokens}）"))
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, trust_remote_code=True,
                  gpu_memory_utilization=0.7, max_model_len=4096)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        out = llm.generate([prompt_text], sp)[0].outputs[0].text
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
        inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
        out = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    print(f"\n[模型生成]\n{out}")
    print(f"\n[与金标对比]  一致: {'是' if out.strip() == pt._strip_think_prefix(output_text).strip() else '否（diff 看上两段）'}")


if __name__ == "__main__":
    main()
