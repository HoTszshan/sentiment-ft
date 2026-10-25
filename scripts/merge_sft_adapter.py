#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT adapter 合并：base + ΔSFT → 一个可直接当基座读的完整模型目录

为什么要有这一步（而不是在 train_grpo.py 里 load_adapter）:

  1. **KL 参考口径**：TRL 的 GRPOTrainer 走 PEFT 路径时，参考模型来自
     model.disable_adapter()——它会关掉**全部** adapter。若把 SFT adapter 与待训
     GRPO LoRA 叠在同一个模型上，reference 会退化成裸基座，KL 惩罚于是变成
     「把策略往忘掉 SFT 的方向拽」，β 越大坏得越快。
  2. **解耦**：merge 是一次性离线操作；train_grpo.py 只认识「一个模型目录」，
     不关心它是原厂权重还是合并产物。

等价性:
  LoRA 增量可加。W0 + ΔSFT 作为新基座再挂 ΔGRPO，与「三段权重分别挂载」数学等价，
  所以生产侧 Base INT4 → SFT-LoRA → RL-LoRA 的挂载形态不受影响。

用法:
  # 体检（不 import torch，任何机器可跑）
  python3 scripts/merge_sft_adapter.py \
      --base models/Qwen3-4B-Instruct \
      --adapter output/probe_lora_20261022_101500 \
      --out models/Qwen3-4B-Instruct-SFT --dry_run

  # 合并（含合并前后 logits 一致性校验）
  python3 scripts/merge_sft_adapter.py \
      --base models/Qwen3-4B-Instruct \
      --adapter output/probe_lora_20261022_101500 \
      --out models/Qwen3-4B-Instruct-SFT
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ============================================================
# 体检（纯路径判断，不碰 torch）
# ============================================================

def preflight(base: str, adapter: str, out: str) -> dict:
    """合并前体检：路径/配置存在性 + LoRA 秩回显 + 基座一致性告警。"""
    base_p, ad_p, out_p = Path(base), Path(adapter), Path(out)

    if not (base_p / "config.json").exists():
        raise SystemExit(f"[FATAL] 基座不是 HF 模型目录（缺 config.json）: {base_p}")

    cfg_path = ad_p / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[FATAL] adapter 目录缺 adapter_config.json: {ad_p}")
    if not any((ad_p / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin")):
        raise SystemExit(f"[FATAL] adapter 目录找不到权重文件: {ad_p}")

    acfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    declared = acfg.get("base_model_name_or_path") or ""
    if declared and Path(declared).resolve() != base_p.resolve() and Path(declared).name != base_p.name:
        print(f"[WARN] adapter 训练时记录的基座是 {declared}，本次 --base 是 {base_p}；"
              f"一致之前不要合并——错配基座的加法结果没有意义")

    if (out_p / "config.json").exists():
        print(f"[WARN] 输出目录已有完整模型，将被覆盖: {out_p}")

    return {
        "base": str(base_p.resolve()),
        "adapter": str(ad_p.resolve()),
        "out": str(out_p.resolve()),
        "lora_r": acfg.get("r"),
        "lora_alpha": acfg.get("lora_alpha"),
        "target_modules": acfg.get("target_modules"),
    }


# ============================================================
# 合并前后的 logits 一致性校验
# ============================================================

VERIFY_PROMPT = (
    "<|im_start|>system\n你是舆情负面精判模型。<|im_end|>\n"
    "<|im_start|>user\n【文本】某公司被曝债务违约，股价三日跌停。<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _logits(model, tok, prompt: str):
    import torch
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**enc).logits.float().cpu()


def verify_parity(before, after, tol: float = 5e-2) -> dict:
    """
    合并前（base+adapter）与合并后（merged）在同一条 prompt 上的 logits 对比。
    bf16 加法有舍入，绝对误差看 tol；真正该对齐的是 argmax 与 top-1 概率。
    """
    import torch
    diff = (before - after).abs()
    max_abs = float(diff.max())
    last = -1
    argmax_same = bool((before[:, last, :].argmax(-1) == after[:, last, :].argmax(-1)).all())
    top1_diff = float((before[:, last, :].softmax(-1).max(-1).values
                       - after[:, last, :].softmax(-1).max(-1).values).abs().max())
    return {
        "max_abs_logit_diff": round(max_abs, 6),
        "argmax_equal": argmax_same,
        "top1_prob_diff": round(top1_diff, 6),
        "tolerance": tol,
        "passed": argmax_same and max_abs <= tol,
    }


# ============================================================
# main
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="base + SFT adapter → 合并模型")
    ap.add_argument("--base", required=True, help="原厂基座目录（含 config.json）")
    ap.add_argument("--adapter", required=True, help="SFT 训出的 adapter 目录")
    ap.add_argument("--out", required=True, help="合并后模型输出目录")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--no_verify", action="store_true", help="跳过合并前后 logits 一致性校验")
    ap.add_argument("--dry_run", action="store_true", help="只做路径体检，不加载模型")
    return ap


def main():
    args = _build_parser().parse_args()
    info = preflight(args.base, args.adapter, args.out)

    print("=" * 74)
    print("SFT adapter 合并 · base + ΔSFT → 单一模型目录")
    print("=" * 74)
    print(f"基座     : {info['base']}")
    print(f"adapter  : {info['adapter']}")
    print(f"输出     : {info['out']}")
    print(f"LoRA     : r={info['lora_r']} α={info['lora_alpha']} "
          f"target={len(info['target_modules'] or [])} 处")
    print(f"精度     : {args.dtype}   device: {args.device}")

    if args.dry_run:
        print("\n[DRY RUN] 路径体检通过，未加载模型。去掉 --dry_run 执行合并。")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device == "cpu" and dtype is torch.bfloat16:
        print("[WARN] CPU 上 bf16 前向可能不被部分算子支持；校验失败请改用 --dtype fp32")
    print(f"\n[1/4] 加载基座（{device}, {args.dtype}）…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True, device_map=None)
    model.to(device)

    print("[2/4] 挂载 SFT adapter …")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    parity = None
    if not args.no_verify:
        print("[3/4] 校验：合并前 logits …")
        before = _logits(model, tok, VERIFY_PROMPT)

    print("[4/4] merge_and_unload + 落盘 …")
    model = model.merge_and_unload()
    model.eval()

    if not args.no_verify:
        after = _logits(model, tok, VERIFY_PROMPT)
        parity = verify_parity(before, after)
        print(f"      max|Δlogit|={parity['max_abs_logit_diff']}  "
              f"argmax 相同={parity['argmax_equal']}  "
              f"→ {'PASS' if parity['passed'] else 'FAIL'}")
        if not parity["passed"]:
            raise SystemExit("[FATAL] 合并前后不一致，拒绝落盘——请检查 adapter 基座是否匹配")

    out_p = Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_p, safe_serialization=True)
    tok.save_pretrained(out_p)

    manifest = {
        "base": info["base"],
        "adapter": info["adapter"],
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lora": {"r": info["lora_r"], "alpha": info["lora_alpha"],
                 "target_modules": info["target_modules"]},
        "dtype": args.dtype,
        "parity": parity,
        "versions": {m: getattr(__import__(m), "__version__", "?")
                     for m in ("torch", "transformers", "peft")},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_p / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print(f"合并完成: {out_p}")
    print(f"证据链  : {out_p / 'merge_manifest.json'}")
    print(f"耗时    : {manifest['elapsed_sec']}s")
    print(f"下一步  : python3 scripts/train_grpo.py --model_name_or_path {out_p} ...")
    print("=" * 74)


if __name__ == "__main__":
    main()
