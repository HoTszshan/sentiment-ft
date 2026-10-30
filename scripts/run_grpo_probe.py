#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 探针 · 同题 G=4 采样 → 四维打分 → k∈{0..4} 分流

这是 README §3.4 明标「未写」的那一件，也是正式训练池
`grpo_prompts_3000.jsonl` 三阶段筛题的第 3 步落地：

  阶段1 生产日志筛题 → 阶段2 金标打标 → **阶段3 G=4 难度筛选只留 k∈{1,2,3}** ← 本脚本

k = 同一 prompt 采 G 份，其中「判对 且 evidence 溯源过门禁」的份数。
  k=0  全错   → RL 无米下锅 (std=0，梯度为 0)，退回 SFT 补课
  k=1~3       → 训练池，组内有方差、有正负样本可比
  k=4  全对   → 组内方差恒 0，花 RL 的钱跑了个更贵的 SFT

不依赖 trl，只依赖 transformers/torch（推理侧）。**打分器仍是零依赖的 reward_fns**，
所以本脚本可以先在无 GPU 环境用 --dry_run 跑通全链路。

用法:
  # 1) 先干跑自检（不加载模型，用 gold 反构「标准答案」当采样结果）
  python3 scripts/run_grpo_probe.py --data data/probe_test_grpo_train_synth.jsonl --dry_run

  # 2) 真实 rollout (需 GPU + SFT 探针训完的 adapter)
  python3 scripts/run_grpo_probe.py \
      --data data/probe_test_grpo_train_synth.jsonl \
      --base_model models/Qwen3-4B-Instruct \
      --adapter output/run_<时间戳>/sft_lora \
      --g 4 --temperature 0.7 --max_new_tokens 512 \
      --out output/grpo/grpo_probe_kstats.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from grpo_common import (                                            # noqa: E402
    GrpoHyper, budget_note, summarize_buckets,
)
from grpo_probe_core import (                                        # noqa: E402
    Sampler, oracle_sampler, run_probe,
)
# run_probe_batch 延迟导入（仅 --backend vllm 用），保持 HF/dry_run 路径零新依赖


def print_summary(summary: Dict[str, Any], g: int, elapsed: float) -> None:
    hist = summary["hist"]
    print("\n--- k 分桶 (k = 组内判对且溯源过门禁的轨迹数) ---")
    width = max(1, max(hist.values()) if hist else 1)
    for k in range(g + 1):
        n = hist.get(str(k), hist.get(k, 0))
        bar = "█" * int(round(n / width * 30)) if width else ""
        tag = {0: "全错 → 丢弃 (RL 无梯度)", g: "全对 → 丢弃 (方差恒 0)"}.get(k, "✅ 训练池")
        print(f"  k={k}  {n:>4} 条 {bar}  {tag}")
    print(f"\n  池子       : {summary['n']} 条 / G={g}")
    print(f"  训练池(k∈1..{g-1}): {summary['keep']} 条 "
          f"({summary['keep_ratio']*100:.1f}%)  ← 这才是 GRPO 训练集")
    print(f"  丢弃 k=0   : {summary['drop_k0']} 条")
    print(f"  丢弃 k={g}   : {summary['drop_k_max']} 条")
    if summary.get("keep_by_hard_case"):
        print("  训练池难点分布: " + " · ".join(
            f"{k}={v}" for k, v in summary["keep_by_hard_case"].items()))
    if summary.get("keep_by_sentiment"):
        print("  训练池极性分布: " + " · ".join(
            f"{k}={v}" for k, v in summary["keep_by_sentiment"].items()))
    print(f"\n  耗时 {elapsed:.1f}s")


# ============================================================
# 真实 rollout (torch/transformers 延迟导入，保持本模块可被单测 import)
# ============================================================

def build_model_sampler(base_model: str, adapter: Optional[str], g: int,
                        temperature: float, max_new_tokens: int,
                        device: str = "auto") -> Sampler:
    """加载基座(+可选 LoRA) → 返回 (prompt, n) -> n 条 completion 的采样器。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cuda = torch.cuda.is_available()
    if device == "auto":
        device = "cuda" if cuda else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if cuda else torch.float32,
        trust_remote_code=True,
        device_map=device if device not in ("cuda", "cpu") else None,
    )
    # device_map=None 时权重留在 CPU；PreTrainedModel.device 是 property 恒有值，
    # 不能用 `is None` 判断是否已上卡——显式 .to() 才保证生成走 GPU
    model.to(device)
    print(f"[PROBE] 模型已上卡: {next(model.parameters()).device}")

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        # 合并进基座：探针只需纯前向，省一层 LoRA 分支开销
        model = model.merge_and_unload()
        print(f"[PROBE] 已加载并合并 adapter: {adapter}")
    model.eval()

    gen_kwargs = dict(
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        num_return_sequences=g,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    def _sampler(prompt: str, n: int) -> List[str]:
        with torch.no_grad():
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, **{**gen_kwargs, "num_return_sequences": n})
            texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=False)
        # 只截尾部 pad，保留 CoT 标签（skip_special_tokens 会连 CoT 标签一起吃掉）
        cleaned = []
        for t in texts:
            cleaned.append(re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$", "", t or "").strip())
        return cleaned

    return _sampler


def build_vllm_batch_sampler(base_model: str, adapter: Optional[str], g: int,
                             temperature: float, max_new_tokens: int,
                             gpu_memory_utilization: float = 0.9):
    """
    vLLM 批量采样器（对齐 sft_infer.py 的引擎参数，含踩坑结论）:
    ([prompt...], g) -> 每条 prompt g 份 completion。一次性全部进队，
    连续批处理吃满吞吐——1098 题 × G=4 约 4400 次生成，HF generate 1~3 小时，
    vLLM 分钟级。
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enable_lora = bool(adapter)
    t_load = time.time()
    llm = LLM(
        model=base_model,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
        tensor_parallel_size=1,
        # 防坑 (train_grpo.py 补丁④同源): 模型原生 max_position 256K，
        # 不限 max_model_len 时 KV profiling 直接 OOM。
        # prompt p99≈840 + 生成 512，2048 绰绰有余 (sft_infer 实测口径)。
        max_model_len=2048,
    )
    print(f"[PROBE] vLLM 引擎+模型加载: {time.time()-t_load:.1f}s")

    sampling_params = SamplingParams(
        n=g,
        temperature=temperature,
        top_p=0.9 if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
    )
    lora_req = LoRARequest("probe_adapter", 1, adapter) if enable_lora else None

    def _batch_sampler(prompts, n):
        t_gen = time.time()
        outputs = llm.generate(list(prompts), sampling_params, lora_request=lora_req)
        dt = time.time() - t_gen
        print(f"[PROBE] 批量生成 {len(prompts)} 题 × G={n}: {dt:.1f}s "
              f"(吞吐 {len(prompts)*n/max(dt,1e-9):.1f} 轨迹/s)")
        groups = []
        for out in outputs:
            texts = [c.text for c in out.outputs][:n]
            if len(texts) != n:
                raise ValueError(f"vLLM 返回 {len(texts)} 份，期望 G={n}")
            # 截尾部 pad，保留 CoT 标签（与 HF 路径同一清洗口径）
            groups.append([
                re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$", "", t or "").strip()
                for t in texts])
        return groups

    return _batch_sampler


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="GRPO 探针: G=4 采样 → 四维打分 → k 分流")
    ap.add_argument("--data", default="data/probe_test_grpo_train_synth.jsonl")
    ap.add_argument("--out", default="output/grpo/grpo_probe_kstats.json")
    ap.add_argument("--rows_out", default=None, help="逐题明细 jsonl（默认与 --out 同名 .rows.jsonl）")
    ap.add_argument("--base_model", default=None, help="基座路径/名（--dry_run 时可不给）")
    ap.add_argument("--adapter", default=None, help="SFT LoRA adapter 目录；留空 = 裸基座对照")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                    help="hf=transformers 逐题 generate（兼容旧路径）；"
                         "vllm=批量进队（1098 题 × G=4 分钟级，推荐）")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="vLLM KV cache 显存占比（仅 --backend vllm 生效）")
    ap.add_argument("--pool_out", default=None,
                    help="训练池导出路径: 只留 k∈{1..G-1} 的原题行落盘"
                         "（G0 探针 → G1 训练池的一步导出）")
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="实测整段输出 max 383（CoT ~257 + JSON ~81），留余量；"
                         "65~105 是 JSON 段口径，不是整段，别按它截")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="不加载模型：用 gold 反构标准答案当采样结果，验数据+打分+分桶全链路")
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"[FATAL] 找不到数据: {path}")
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("[FATAL] 数据为空")

    h = GrpoHyper(num_generations=args.g, temperature=args.temperature,
                  max_completion_length=args.max_new_tokens)

    print("=" * 74)
    print(f"GRPO 探针 · 同题 G={args.g} 采样 → 四维打分 → k 分流")
    print(f"  数据       : {path}  ({len(records)} 条)")
    print(f"  策略       : {'--dry_run (gold 反构，非真实模型)' if args.dry_run else args.base_model}"
          f"{' + adapter ' + args.adapter if args.adapter else ' (裸基座)'}")
    print(f"  T={h.temperature}  max_new_tokens={h.max_completion_length}  ← {budget_note(h)}")
    print("=" * 74)

    t0 = time.time()
    rows = None
    if args.dry_run:
        # 逐题各建一个 oracle 采样器 (gold 因题而异)
        def sampler(prompt, n):
            rec = next(r for r in records if r["prompt"] == prompt)
            return oracle_sampler(rec["gold"])(prompt, n)
        rows = run_probe(records, sampler, g=args.g)
    else:
        if not args.base_model:
            raise SystemExit("[FATAL] 非 dry_run 必须给 --base_model")
        if args.backend == "vllm":
            from grpo_probe_core import run_probe_batch
            batch_sampler = build_vllm_batch_sampler(
                args.base_model, args.adapter, args.g,
                args.temperature, h.max_completion_length,
                args.gpu_memory_utilization)
            rows = run_probe_batch(records, batch_sampler, g=args.g)
        else:
            sampler = build_model_sampler(
                args.base_model, args.adapter, args.g,
                args.temperature, h.max_completion_length)
            rows = run_probe(records, sampler, g=args.g)
    elapsed = time.time() - t0
    summary = summarize_buckets(rows, g=args.g)
    summary.update({"data": str(path), "base_model": args.base_model,
                    "adapter": args.adapter, "dry_run": bool(args.dry_run),
                    "temperature": h.temperature, "max_new_tokens": h.max_completion_length,
                    "elapsed_sec": round(elapsed, 1)})
    print_summary(summary, args.g, elapsed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    rows_out = Path(args.rows_out) if args.rows_out else out.with_suffix("").with_name(
        out.stem + ".rows.jsonl")
    with open(rows_out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  统计: {out}\n  逐题: {rows_out}")

    if args.pool_out:
        from grpo_probe_core import export_pool
        stat = export_pool(records, rows, g=args.g, out_path=args.pool_out)
        print(f"  训练池导出: {args.pool_out}  "
              f"(留 {stat['kept']} / 丢 k=0 {stat['dropped_k0']} + k={args.g} {stat['dropped_kmax']})")

    if args.dry_run:
        print("\n⚠️ dry_run 的 k 只证明「数据 + 打分器 + 分桶」链路通，**不是模型成绩**。")
    print("=" * 74)


if __name__ == "__main__":
    main()
