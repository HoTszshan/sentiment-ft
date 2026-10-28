#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探针实验 · 推理脚本 (Probe Inference)

作用: 加载 SFT 后的 LoRA Adapter，在验证集/隔离探针集上批量生成，
      产出可直接喂给 evaluate_sft.py 「模式 B」的预测文件 (pred.jsonl)。

生成 prompt 与训练侧同源 (lib/prompt_template.py):
    train_sft_lora.SFTConversationDataset → prompt_template.render_full()
    sft_infer                            → prompt_template.render_prompt()
两者走同一个渲染函数，且 Dataset 初始化时会做逐字节 parity 校验。

用法:
    python sft_infer.py \
        --base_model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
        --adapter_path /root/autodl-tmp/models/sft_senti/run_<时间戳>/sft_lora \
        --data_file data/mixed/sft_val.jsonl \
        --out output/pred_val.jsonl

产出文件: 每行 = 该样本模型的原始输出 (含 <think> ... JSON)，
          行序与 --data_file 完全对齐，供 evaluate_sft --gold_file/--pred_file 使用。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

# 外壳渲染唯一真理源：与训练侧 prompt_template 同源。
# 训练/推理的 prompt 一旦分叉，loss 降了而推理格式崩 —— 这种故障从曲线上看不出来。
import prompt_template as pt  # noqa: E402


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
    ap.add_argument("--think_prefix", action="store_true",
                    help="前缀预置：在 assistant 前缀尾部追加 '<think>\\n'。"
                         "必须与训练侧一致（训练默认关；训练开过才加此 flag）")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="推理引擎: hf (单卡 PyTorch 原生，默认稳定) | vllm (高吞吐批量推理)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                    help="vLLM 显存占用上限 (默认 0.7，防止与其他进程抢显存)")
    ap.add_argument("--log", default="",
                    help="日志落盘路径（stdout 全量 tee + 关键计时落档）；不给则只打印")
    args = ap.parse_args()

    # 日志 tee：所有 print 走 _say，给了 --log 就同步追加落盘
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(args.log, "a", encoding="utf-8") if args.log else None
    _t_start = time.time()

    def _say(msg: str = ""):
        print(msg, flush=True)
        if _log_fh:
            _log_fh.write(msg + "\n")
            _log_fh.flush()

    # 渲染协议开关：与训练侧同源（prompt_template.set_think_prefix）
    pt.set_think_prefix(args.think_prefix)

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

    _say("=" * 70)
    _say("探针推理")
    _say(f"  基座    : {args.base_model}")
    _say(f"  Adapter : {args.adapter_path or '(无 — 裸基座对照)'}")
    _say(f"  数据    : {args.data_file} ({len(rows)} 条)")
    _say(f"  产出    : {args.out}")
    if _log_fh:
        _say(f"  日志    : {args.log}")
    _say("=" * 70)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_tag = n_json = n_ev = 0

    if args.backend == "vllm":
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError:
            sys.stderr.write("[FATAL] 缺少 vLLM 依赖: pip install vllm\n")
            raise SystemExit(2)

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                                  padding_side="left")
        prompts = [
            pt.render_prompt(tokenizer, r.get("instruction", "") or "", r.get("input", "") or "")
            for r in rows
        ]

        enable_lora = bool(args.adapter_path)
        t_load = time.time()
        llm = LLM(
            model=args.base_model,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=enable_lora,
            max_lora_rank=64 if enable_lora else None,
            tensor_parallel_size=1,
            # 模型原生 max_position=256K，KV cache 装不下会直接拒启动；
            # prompt p99≈840 token + 生成 512，2048 绰绰有余
            max_model_len=2048,
        )
        _say(f"[TIME] 引擎+模型加载: {time.time()-t_load:.1f}s")

        t_render = time.time()
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.9 if args.temperature > 0 else 1.0,
            max_tokens=args.max_new_tokens,
        )
        lora_req = LoRARequest("probe_adapter", 1, args.adapter_path) if enable_lora else None

        # 批量一次进队：整体耗时 = t_gen；吞吐 = n / t_gen
        t_gen = time.time()
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        t_gen = time.time() - t_gen
        _say(f"[TIME] prompt 渲染: {time.time()-t_render:.1f}s | "
             f"批量生成 {len(rows)} 条: {t_gen:.1f}s "
             f"(吞吐 {len(rows)/max(t_gen,1e-9):.2f} 条/s, "
             f"均值 {t_gen/max(len(rows),1):.2f}s/条)")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (row, out) in enumerate(zip(rows, outputs), 1):
                text = out.outputs[0].text
                if pt.USE_THINK_PREFIX:
                    text = pt.THINK_PREFIX + text

                if "<think>" in text and "</think>" in text:
                    n_tag += 1
                try:
                    after = text.split("</think>")[-1].lstrip()
                    pred = json.loads(after[: after.rfind("}") + 1]) if "}" in after else None
                    if isinstance(pred, dict):
                        n_json += 1
                        ev = str(pred.get("evidence", ""))
                        if ev and ev in row.get("input", ""):
                            n_ev += 1
                except Exception:
                    pass

                f.write(text.replace("\n", " ").strip() + "\n")
                if i % 10 == 0 or i == len(rows):
                    _say(f"  [{i}/{len(rows)}] tag合规 {n_tag} | JSON合法 {n_json} | "
                         f"evidence 命中原文 {n_ev}")

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                                  padding_side="left")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        if args.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
            _say(f"[OK] 已加载 LoRA Adapter: {args.adapter_path}")
        model.eval()
        _say(f"[TIME] 模型加载: {time.time()-t0:.1f}s")

        with open(out_path, "w", encoding="utf-8") as f:
            n_total = len(rows)
            # 单条生成慢（百秒级/条 × 500 条），进度条按样本打：tqdm 覆盖 stderr，
            # 合规计数打在同一行后缀，不与 tqdm 冲突
            from tqdm import tqdm
            pbar = tqdm(total=n_total, desc="推理", unit="条",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")
            for i, row in enumerate(rows, 1):
                prompt = pt.render_prompt(tokenizer, row.get("instruction", "") or "",
                                          row.get("input", "") or "")
                # add_special_tokens=False: 模板首字符就是 <|im_start|>，再补 BOS 会与训练侧分叉
                inputs = tokenizer(prompt, return_tensors="pt",
                                   add_special_tokens=False).to(device)

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
                # 预置的 "<think>\n" 在 prompt 里、不在生成段里；不补回则预测文件
                # 永不含 <think>，闭合率恒为 0%（测量 bug，非模型问题）
                if pt.USE_THINK_PREFIX:
                    text = pt.THINK_PREFIX + text

                # 静态合规指标（四查的①②在推理侧可提前观测）
                if "<think>" in text and "</think>" in text:
                    n_tag += 1
                try:
                    after = text.split("</think>")[-1].lstrip()
                    pred = json.loads(after[: after.rfind("}") + 1]) if "}" in after else None
                    if isinstance(pred, dict):
                        n_json += 1
                        ev = str(pred.get("evidence", ""))
                        if ev and ev in row.get("input", ""):
                            n_ev += 1
                except Exception:
                    pass

                f.write(text.replace("\n", " ").strip() + "\n")
                f.flush()

                pbar.set_postfix_str(
                    f"tag {n_tag} | JSON {n_json} | ev {n_ev}")
                pbar.update(1)
                if i % 50 == 0 or i == n_total:
                    _el = time.time() - t0
                    _say(f"  [{i}/{n_total}] tag合规 {n_tag} | JSON合法 {n_json} | "
                         f"evidence 命中原文 {n_ev} | 累计 {_el:.0f}s "
                         f"(均值 {_el/i:.1f}s/条)")
            pbar.close()

    _t_total = time.time() - t0
    _say("=" * 70)
    _say(f"[OK] 推理完成 {len(rows)} 条，总耗时 {_t_total:.1f}s "
         f"(均值 {_t_total/max(len(rows),1):.2f}s/条)")
    _say(f"     CoT 标签闭合率 : {n_tag}/{len(rows)} ({n_tag/max(len(rows),1)*100:.1f}%)")
    _say(f"     JSON 解析合法率: {n_json}/{len(rows)} ({n_json/max(len(rows),1)*100:.1f}%)")
    _say(f"     evidence 原文命中: {n_ev}/{len(rows)} ({n_ev/max(len(rows),1)*100:.1f}%)")
    _say(f"     预测文件: {out_path}")
    _say("=" * 70)
    _say("下一步: python evaluate_sft.py --gold_file "
         f"{args.data_file} --pred_file {args.out}")
    if _log_fh:
        _log_fh.close()


if __name__ == "__main__":
    main()
