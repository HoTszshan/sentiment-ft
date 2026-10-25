#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情负面精判 · Rule-based GRPO 训练主脚本 (TRL GRPOTrainer)

唯一真理源:
  MVP/notes/GRPO训练实战手册.md §1.1 LoRA 三重减法 / §1.3 超参 / §1.4 闭环 / §1.5 三闸
  MVP/notes/舆情负面精判_GRPO数据准备实战手册.md §五 四维 Reward 契约

对标参考 (取其可用、弃其不可用):
  llm-engineering-core/12S2W10_rlhf/GRPO/train.py   ← 采用: TRL GRPOTrainer + 契约化 reward
                                                       + wandb 降级 + --limit 冒烟
  llm-engineering-core/13S3W5_rm_r1/RM-R1/rm_r1     ← 弃用: verl 全参分布式栈，与
                                                       「LoRA + 单卡 24GB」正面冲突

⚠️ 硬约束（直接抄参考会报错）
1. Reward 单函数实现短路逻辑：格式校验失败则其他奖励清零，不可拆分求和
2. max_completion_length 按实测配置：65~105仅JSON长度，不含CoT；上限不足会截断、奖励短路
3. loss_type="grpo"，需均值中心化；新版dr_grpo不满足该约定
4. 输入必须是合并SFT后的完整模型，不能单独加载adapter，避免KL回退基座遗忘SFT

评估熔断(TODO)：
--eval_file 开启评估，以首轮reward为基线；跌幅超限告警，--fuse_hard才终止，探针模式仅告警


用法:
  # 干跑：只打印超参预算 + 数据体检 + 版本体检，不加载模型（任何机器都能跑）
  python3 scripts/train_grpo.py --data data/grpo_prompts_probe.jsonl --dry_run

  # 真实探针训练 (需 GPU + trl；起点 = 已合并 SFT 的模型目录)
  #   models/Qwen3-4B-Instruct-SFT 由 scripts/merge_sft_adapter.py 一次产出：
  #   base + ΔSFT 先合成一个完整模型，本脚本只认「一个模型目录」。
  python3 scripts/train_grpo.py \
      --data data/grpo_prompts_probe.jsonl \
      --model_name_or_path models/Qwen3-4B-Instruct-SFT \
      --output_dir output/grpo_lora \
      --log_file logs/train_grpo.log \
      --wandb_project sentiment-grpo-probe
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from grpo_common import (                                    # noqa: E402
    GrpoHyper, Tee, assert_zam_equivalence, attach_log, budget_note, detach_log,
    build_dataset_rows, validate_generation_shape,
)
from grpo_probe_core import extract_completion_text           # noqa: E402


# ============================================================
# 自定义 flag 拦截 (与 train_sft_lora.py 同语义：日志双写 + wandb)
# ============================================================

CUSTOM_FLAGS = {
    "--log_file": ("log_file", str, None),
    "--wandb_project": ("wandb_project", str, "sentiment-grpo-probe"),
    "--wandb_run_name": ("wandb_run_name", str, lambda: f"grpo_{time.strftime('%m%d_%H%M%S')}"),
    "--data": ("data", str, "data/grpo_prompts_probe.jsonl"),
    "--no_wandb": ("no_wandb", bool, False),
}


def intercept_custom_flags(argv: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """把 --log_file / --wandb_* / --data 抠出来，剩下的交给 argparse。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    # 键名用规范名（spec[0]），不是命令行写法（"--no_wandb"）
    opts: Dict[str, Any] = {spec[0]: (spec[2]() if callable(spec[2]) else spec[2])
                            for spec in CUSTOM_FLAGS.values()}
    rest, i = [], 0
    while i < len(argv):
        a = argv[i]
        hit = CUSTOM_FLAGS.get(a)
        if hit:
            key, cast, default = hit
            if cast is bool:            # 开关型 flag，无值
                opts[key] = True
                i += 1
                continue
            if i + 1 >= len(argv):
                raise SystemExit(f"[FATAL] {a} 需要一个值")
            opts[key] = cast(argv[i + 1])
            i += 2
            continue
        rest.append(a)
        i += 1
    return opts, rest


# ============================================================
# 打分器 → TRL reward func
# ============================================================

def _pick(values: Any, i: int, g: int) -> Any:
    """
    TRL 各版本对 reward_func 的列值对齐方式不一致:
      (a) 逐 completion 对齐: len == n
      (b) 逐 prompt 对齐再复制: len == n // G  (第 i 条取第 i//G 题的列)
      (c) 标量
    三种都吃，避免"跑起来才发现 gold 串位"。
    """
    n_col = len(values) if isinstance(values, (list, tuple)) else None
    if n_col is None:
        return values
    if i < n_col:
        return values[i]
    # 列比轨迹短 → 大概率是每题一列
    j = i // max(1, g)
    return values[j] if j < n_col else values[-1]


def make_reward_fn(g: int = 4):
    """
    唯一一个 reward func（短路语义不可拆，见文件头硬约束 1）。
    返回逐轨迹总分 ∈ [-2.5, +2.0]。
    """
    def reward_fn(completions: Sequence[Any], **kwargs) -> List[float]:
        n = len(completions)
        scores: List[float] = []
        for i in range(n):
            comp = extract_completion_text(completions[i])
            gold = {
                "sentiment": _pick(kwargs.get("gold_sentiment"), i, g),
                "relevance": _pick(kwargs.get("gold_relevance"), i, g),
                "hard_case_type": _pick(kwargs.get("gold_hard_case_type"), i, g),
                "risk_categories": _gold_risk(_pick(kwargs.get("gold_risk_json"), i, g)),
                "evidence": _pick(kwargs.get("gold_evidence"), i, g),
            }
            input_text = _pick(kwargs.get("input_text"), i, g) or ""
            from reward_fns import compute_reward
            scores.append(compute_reward(comp, input_text, gold)["total"])
        return scores

    return reward_fn


def build_fuse_callback(eval_on: bool, fuse_threshold: float, fuse_hard: bool,
                        reward_fn, g: int):
    """
    对齐税熔断 callback（纯 TRL/transformers 公开 API，不 hack 内部）：
      on_evaluate → 用同一 reward_fn 给 eval 基线集的 model 生成的 completion 打分 →
      均值相对 baseline 跌幅超阈值 → [FUSE] 告警（fuse_hard 时抛异常终止）。

    baseline 取首轮 eval 均值——首轮不可能熔断（没有比较对象）。
    eval 数据行的 prompt 列从 TRL 侧透传；这里只在 on_evaluate 里自己调
    trainer.model.generate 采样（temperature 取 GRPOConfig 的配置）。
    """
    from grpo_common import should_fuse

    state = {"baseline": None, "last": None}

    class _FuseCallback(TrainerCallback):
        def on_evaluate(self, args, trainer_state, control, **kwargs):
            trainer = kwargs.get("model") and None  # TRL 不传 trainer；用闭包外变量
            return control

    # TrainerCallback 不递 trainer 实例，改用闭包捕获：由调用方在 trainer 构建后注入
    holder: Dict[str, Any] = {"trainer": None}

    class _FuseCallbackWithTrainer(TrainerCallback):
        def on_evaluate(self, trainer_args, state, control, **kwargs):
            tr = holder.get("trainer")
            if tr is None:
                return control
            try:
                totals = _eval_reward_mean(tr, reward_fn, g)
            except Exception as e:                       # eval 失败不能带崩训练
                print(f"[WARN] eval 打分失败，跳过本轮熔断判定: {e}")
                return control
            if state["baseline"] is None:
                state["baseline"] = totals
                print(f"[EVAL] baseline reward 均值 = {totals:.4f}")
                return control
            state["last"] = totals
            if should_fuse(state["baseline"], totals, fuse_threshold):
                msg = (f"[FUSE] 对齐税熔断：eval reward 均值 {totals:.4f} 相对 baseline "
                       f"{state['baseline']:.4f} 跌幅超 {fuse_threshold:.0%}")
                print(msg)
                if fuse_hard:
                    control.should_training_stop = True
            else:
                print(f"[EVAL] reward 均值 = {totals:.4f}（baseline {state['baseline']:.4f}）")
            return control

    cb = _FuseCallbackWithTrainer()
    # 把 trainer 实例塞进闭包：build_trainer 返回后由 main() 调 _attach(trainer)
    cb._attach = lambda tr: holder.__setitem__("trainer", tr)
    return cb


def _eval_reward_mean(trainer, reward_fn, g: int) -> float:
    """对 eval_dataset 逐题采 1 条 completion，用 reward_fn 打分取均值。"""
    import torch
    tok = trainer.processing_class
    ds = trainer.eval_dataset
    cfg = trainer.args
    max_new = getattr(cfg, "max_completion_length", 200) or 200
    temp = getattr(cfg, "temperature", 0.7) or 0.7
    model = trainer.model_wrapped if hasattr(trainer, "model_wrapped") else trainer.model
    model.eval()
    totals = []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds[i]
            prompt = row["prompt"] if isinstance(row["prompt"], str) else str(row["prompt"])
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=getattr(cfg, "max_prompt_length", 2048) or 2048)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(**enc, do_sample=True, temperature=temp,
                                 max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
            # 复用训练侧 reward 路径：gold 列名与训练一致
            gold_kwargs = {k: [row.get(k)] for k in
                           ("gold_sentiment", "gold_relevance", "gold_hard_case_type",
                            "gold_risk_json", "gold_evidence", "input_text")}
            scores = reward_fn([text], **gold_kwargs)
            totals.append(scores[0])
    model.train()
    return sum(totals) / max(1, len(totals))


def _gold_risk(v: Any) -> List[str]:
    if isinstance(v, str):
        try:
            v = json.loads(v or "[]")
        except Exception:
            v = []
    return list(v or [])


# ============================================================
# 组件装配 (延迟导入：保持 --dry_run 在无 torch 机器上也能跑)
# ============================================================

def _origin_note(model_path: str) -> str:
    """
    起点是不是合并产出的模型：看同目录有没有 merge_manifest.json（merge_sft_adapter.py 写的）。
    有就回显「已合并 SFT」，没有就是原厂基座 —— 只做提示，不阻断。
    """
    mf = Path(model_path) / "merge_manifest.json"
    if not mf.exists():
        return "  (原厂基座，未挂 SFT)"
    try:
        m = json.loads(mf.read_text(encoding="utf-8"))
        lora = m.get("lora") or {}
        tag = f"r={lora.get('r')} α={lora.get('alpha')}"
        return f"  (已合并 SFT adapter · {tag} · {m.get('merged_at', '?')})"
    except Exception:
        return "  (检测到 merge_manifest.json，但解析失败)"


def _merge_evidence(model_path: str) -> Optional[Dict[str, Any]]:
    """把合并证据链写进 grpo_metrics.json，事后能追溯这一版 GRPO 的起点血缘。"""
    mf = Path(model_path) / "merge_manifest.json"
    if not mf.exists():
        return None
    try:
        m = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "merge_manifest.json 解析失败"}
    return {k: m.get(k) for k in ("base", "adapter", "merged_at", "lora", "parity")}


def check_versions(strict: bool = False) -> Dict[str, str]:
    """打印 trl / transformers / peft 版本；Qwen3 需 transformers>=4.51。"""
    info, bad = {}, []
    for mod in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = "(未安装)"
            if strict and mod in ("torch", "transformers", "trl", "peft"):
                bad.append(mod)
    if bad:
        raise SystemExit(
            f"[FATAL] 缺少依赖 {bad}\n"
            f"  pip install 'torch>=2.2' 'transformers>=4.55' 'trl>=0.19' "
            f"'peft>=0.12' 'datasets' 'accelerate'\n"
            f"  （Qwen3 架构要求 transformers>=4.51；TRL 老版本 0.14 不认 Qwen3）")
    return info


def dry_run_inspect(data_path: str, h: GrpoHyper, sample: int = 2) -> None:
    """离线体检：数据契约 + 训练预算 + 门禁/掩码自检。不碰 GPU。"""
    print("-" * 74)
    print("数据契约体检")
    rows = build_dataset_rows(load_records(data_path))
    print(f"  行数: {len(rows)}  列: {list(rows[0].keys())}")
    for r in rows[:sample]:
        print(f"  例: sentiment={r['gold_sentiment']:<8} hard_case={r['gold_hard_case_type']:<15} "
              f"relevance={r['gold_relevance']:<7} risk={r['gold_risk_json']}")
        print(f"      evidence={r['gold_evidence']!r}")
        print(f"      input_text 反解长度={len(r['input_text'])} 字")
    empty = [r for r in rows if not r["input_text"]]
    print(f"  input_text 反解失败: {len(empty)} 条" + ("  ← ⚠️ 打分会全池短路，先查 prompt 格式"
                                                     if empty else "  ✅"))
    print("-" * 74)
    print("门禁不变量 + Zero-Advantage Masking 自检")
    print(f"  {'PASS' if assert_zam_equivalence() else 'FAIL'}  "
          f"(全错组 / 全对组 / 同分组 → 优势清零，与 TRL std=0 分支语义重合)")
    print("-" * 74)


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[FATAL] 找不到数据: {p}")
    with open(p, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    if not recs:
        raise SystemExit(f"[FATAL] 数据为空: {p}")
    return recs


def print_budget(h: GrpoHyper, n: int, extra: Dict[str, Any]) -> None:
    b = h.budget(n)
    print("=" * 74)
    print("训练规模预算 (每个数都能算出来源，不是拍的)")
    print("=" * 74)
    rows = [
        ("题数 N", f"{b['n_prompts']} 条"),
        ("每步题数", f"{b['prompts_per_step']} = per_device {h.per_device} × grad_accum {h.grad_accum}"),
        ("生成组数 G", f"{h.num_generations}   ← 组内要有方差，4 是最小偶数组"),
        ("有效 batch", f"{b['trajectories_per_step']} 条轨迹/步 = {b['prompts_per_step']} 题 × G={h.num_generations}"),
        ("采样温度 T", f"{h.temperature}   ← T=0 会退化成更贵的 SFT"),
        ("学习率 lr", f"{h.lr}   ← LoRA 口径：≈ SFT 1e-4 的 1/5；全参 RL 的 1e-6~5e-6 不适用"),
        ("KL 系数 β", f"{h.beta_kl}   ← 常数 KL 管分布漂移，长度由 R_length 管，两者正交"),
        ("CoT 预算", f"{h.max_completion_length}   ← {budget_note(h)}"),
        ("prompt 上限", f"{h.max_prompt_length}   ← =2560(SFT) - 512(completion)；实测 max 774，超了会截断原文"),
        ("每 epoch 步数", f"{b['steps_per_epoch']} 步"),
        ("总步数", f"{b['total_steps']} 步 (epoch={h.num_train_epochs})"),
        ("warmup", f"{b['warmup_steps']} 步"),
    ]
    for k, v in rows:
        print(f"  {k:<14}: {v}")
    print("-" * 74)
    for k, v in (extra or {}).items():
        print(f"  {k:<14}: {v}")
    print("=" * 74)


# ============================================================
# main
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    # 复用 parse_args 的定义体，避免重复维护
    class _P(argparse.ArgumentParser):
        def error(self, message):
            raise SystemExit(f"[参数错误] {message}")

    ap = _P(description="Rule-based GRPO 训练")
    ap.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct")
    ap.add_argument("--output_dir", default="output/grpo_lora")
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--beta_kl", type=float, default=0.04)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--eval_file", default=None,
                    help="对齐税基线集 jsonl（{prompt, gold, [input_text]}）；给了才开 eval 循环")
    ap.add_argument("--eval_steps", type=int, default=2,
                    help="每多少优化器步 eval 一次（仅 --eval_file 生效时）")
    ap.add_argument("--fuse_threshold", type=float, default=0.02,
                    help="对齐税熔断阈值：eval reward 均值相对首轮融资跌幅超过该值即告警")
    ap.add_argument("--fuse_hard", action="store_true",
                    help="熔断时真终止训练（默认只告警）")
    ap.add_argument("--save_steps", type=int, default=0,
                    help=">0 时按步存 checkpoint（save_strategy=steps），0=只在训练结束保存")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--report_to", default="wandb")
    ap.add_argument("--save_strategy", default="no")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main():
    opts, rest = intercept_custom_flags()
    ap = _build_parser()
    args = ap.parse_args(rest)

    log_handle = attach_log(opts["log_file"])
    if log_handle:
        print(f"[LOG] 训练日志同时写入: {opts['log_file']}")

    h = GrpoHyper(
        num_generations=args.num_generations,
        temperature=args.temperature,
        lr=args.learning_rate,
        beta_kl=args.beta_kl,
        per_device=args.per_device_train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
    )

    records = load_records(opts["data"])
    if args.limit:
        records = records[: args.limit]
    n = len(records)

    print("=" * 74)
    print("舆情负面精判 · Rule-based GRPO (无 Critic / 无 RM / 4 条纯 Python 规则)")
    print("=" * 74)
    print(f"数据       : {opts['data']}  ({n} 条)")
    print(f"起点模型   : {args.model_name_or_path}{_origin_note(args.model_name_or_path)}")
    print(f"输出       : {args.output_dir}")
    print(f"日志       : {opts['log_file'] or '(仅终端)'}")
    wandb_on = decide_wandb(args, opts)
    wandb_desc = (f"同步 {opts['wandb_project']} / {opts['wandb_run_name']}" if wandb_on
                  else "关闭 (--no_wandb / 未装 wandb / 非主进程)")
    print(f"wandb      : {wandb_desc}")

    # TRL 硬约束: per_device × world_size 必须被 G 整除
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    err = validate_generation_shape(n, h.num_generations, h.per_device, world_size)
    if err:
        raise SystemExit(f"[FATAL] {err}")
    b = h.budget(n)
    print_budget(h, n, {
        "world_size": world_size,
        "vllm": "开 (需 GPU 数 = 训练进程数 + 1)" if args.use_vllm else "关 (单卡，HF generate 采样)",
        "loss_type": "grpo (均值中心化 + std 归一，对齐 group_advantages)",
    })

    if args.dry_run:
        dry_run_inspect(opts["data"], h)
        if args.eval_file:
            ev = load_records(args.eval_file)
            print(f"eval 基线集体检: {args.eval_file} ({len(ev)} 条)  "
                  f"eval_steps={args.eval_steps}  熔断阈值={args.fuse_threshold:.0%}"
                  f"{'（硬终止）' if args.fuse_hard else '（仅告警）'}")
        else:
            print("eval 循环: 未设 --eval_file → 不开（正式阶段接 k=4 基线集，TODO 5.19）")
        print("\n[DRY RUN] 未加载模型/trl。真实训练去掉 --dry_run。")
        detach_log(log_handle)
        return

    # ---------------- 真实训练 ----------------
    info = check_versions(strict=True)
    print("依赖版本: " + " ".join(f"{k}={v}" for k, v in info.items()))
    print(f"ZAM 自检 : {'PASS' if assert_zam_equivalence() else 'FAIL'}")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    from peft import LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # prompt 上限体检: 截断会砍掉待判原文 → evidence 溯源必然失败 → 全池 -2.5
    lens = [len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]) for r in records]
    real_max = max(lens) if lens else 0
    if real_max > h.max_prompt_length:
        new_len = min(4096, int(real_max * 1.05) + 8)
        print(f"[WARN] prompt 实际最长 {real_max} > 上限 {h.max_prompt_length}，"
              f"截断会砍断原文导致 R_grounding 全池短路 → 自动放宽到 {new_len}")
        h.max_prompt_length = new_len
    print(f"prompt 长度: 最长 {real_max} / 均值 {sum(lens)//max(1,len(lens))} "
          f"→ 上限 {h.max_prompt_length}")

    # TRL 版本差异：tokenizer= 已改名 processing_class=
    rows = build_dataset_rows(records)
    train_dataset = Dataset.from_list(rows)
    print(f"训练集列: {train_dataset.column_names}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    cfg_kwargs = dict(
        output_dir=args.output_dir,
        run_name=opts["wandb_run_name"],
        learning_rate=h.lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=h.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=torch.cuda.is_available(),
        per_device_train_batch_size=h.per_device,
        gradient_accumulation_steps=h.grad_accum,
        num_generations=h.num_generations,
        max_prompt_length=h.max_prompt_length,
        max_completion_length=h.max_completion_length,
        num_train_epochs=h.num_train_epochs,
        beta=h.beta_kl,                       # token 级常数 KL (三闸①)
        max_grad_norm=args.max_grad_norm,
        save_strategy="steps" if args.save_steps > 0 else args.save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        seed=h.seed,
        report_to="wandb" if wandb_on else "none",
        log_on_each_node=False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else 500,
        per_device_eval_batch_size=h.per_device,
        use_vllm=args.use_vllm,
    )
    # 可选字段：不同 TRL 版本有无不同，别让它把启动打崩
    _opt = {
        "max_steps": args.max_steps or 0,          # 传 None 会被当成 >0 判断直接 TypeError
        "temperature": h.temperature,               # 采样温度 T
        "loss_type": "grpo",                        # 均值中心化，对齐 group_advantages 公式
        "beta": h.beta_kl,
        "vllm_device": "auto",
        "vllm_gpu_memory_utilization": 0.3,
        "wandb_project": opts["wandb_project"],
    }
    # 用 dataclass 字段表判断字段是否存在（比试构造干净，也不会吞掉真实报错）
    import dataclasses
    cfg_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
    resolved = {k: v for k, v in _opt.items() if k in cfg_fields}
    missing = [k for k in _opt if k not in cfg_fields]
    for k in missing:
        print(f"[WARN] 本机 TRL {info.get('trl')} 的 GRPOConfig 无 {k} 字段 → 用默认值"
              + ("；温度默认若为 0，组内方差恒 0，RL 会退化成更贵的 SFT → 建议 pip install -U trl>=0.19"
                 if k == "temperature" else ""))

    cfg_kwargs.update({k: v for k, v in resolved.items()})
    training_args = GRPOConfig(**cfg_kwargs)

    if wandb_on:
        os.environ.setdefault("WANDB_PROJECT", opts["wandb_project"])
        print(f"[WANDB] 项目={opts['wandb_project']} run={opts['wandb_run_name']}")
    else:
        training_args.report_to = []

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # ---------------- eval 循环 + 对齐税熔断 (TODO 5.19) ----------------
    eval_dataset = None
    if args.eval_file:
        eval_records = load_records(args.eval_file)
        eval_dataset = Dataset.from_list(build_dataset_rows(eval_records))
        print(f"eval 基线集: {args.eval_file} ({len(eval_records)} 条)  "
              f"eval_steps={args.eval_steps}  熔断阈值={args.fuse_threshold}"
              f"{'（硬终止）' if args.fuse_hard else '（仅告警）'}")

    eval_callback = build_fuse_callback(
        eval_on=bool(eval_dataset),
        fuse_threshold=args.fuse_threshold,
        fuse_hard=args.fuse_hard,
        reward_fn=make_reward_fn(h.num_generations),
        g=h.num_generations,
    ) if eval_dataset else None

    trainer = build_trainer(
        GRPOTrainer,
        model=model,
        peft_config=peft_config,
        reward_funcs=make_reward_fn(h.num_generations),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,      # None = 不开 eval（TRL 支持传 None）
        processing_class=tokenizer,     # TRL <0.19 这个参数叫 tokenizer=
    )
    if eval_callback is not None:
        trainer.add_callback(eval_callback)
        eval_callback._attach(trainer)

    print("=" * 74)
    print(f"开始 GRPO 训练 · {b['total_steps']} 步 × {b['trajectories_per_step']} 轨迹/步")
    print("=" * 74)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_prompts": n, "total_steps": b["total_steps"],
        "trajectories_per_step": b["trajectories_per_step"],
        "hyperparams": {**{k: getattr(h, k) for k in
                           ("num_generations", "temperature", "lr", "beta_kl",
                            "per_device", "grad_accum", "max_completion_length",
                            "max_prompt_length", "seed")},
                        "base": args.model_name_or_path,
                        "merge": _merge_evidence(args.model_name_or_path)},
        "elapsed_sec": round(elapsed, 1),
    }
    with open(Path(args.output_dir) / "grpo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n模型已保存: {args.output_dir}")
    print(f"指标已落盘: {Path(args.output_dir) / 'grpo_metrics.json'}")
    print(f"耗时: {elapsed:.1f}s")
    print("=" * 74)
    detach_log(log_handle)


# ============================================================
# 跨版本兼容的两个小装配器
# ============================================================

def is_main_process() -> bool:
    """主进程判定 (accelerate 下看 LOCAL_RANK；直跑时无分布式环境)。"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def decide_wandb(args, opts: Dict[str, Any]) -> bool:
    """wandb 开关：显式关 / 未装 / 非主进程 → 关，且都不报错。"""
    if opts.get("no_wandb") or args.report_to in ("none", "", None, []):
        return False
    if not is_main_process():
        return False                     # 多进程只让主进程建一个 run
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[WARN] 未安装 wandb，本次跳过在线同步（pip install wandb 后重跑可恢复）")
        return False
    return True


def build_trainer(cls, **kwargs):
    """TRL 版本兼容: processing_class= (>=0.19) ↔ tokenizer= (<0.19)。"""
    try:
        return cls(**kwargs)
    except TypeError as e:
        if "processing_class" not in str(e):
            raise
        kwargs["tokenizer"] = kwargs.pop("processing_class")
        print("[WARN] 本机 TRL 用旧参数名 tokenizer=，已自动改传（建议升级 trl>=0.19）")
        return cls(**kwargs)


if __name__ == "__main__":
    main()
