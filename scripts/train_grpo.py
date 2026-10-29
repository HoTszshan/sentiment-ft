#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""舆情负面精判 · Rule-based GRPO 训练主脚本 (与 train_sft_lora.py 同一骨架)

核心特性:
1. 基座: SFT 探针合并后的 Qwen3-4B (须含 merge_manifest.json; 裸基座仅对照实验用)。
2. LoRA 旁路: r=16, alpha=32, 全量线性层 (attention + MLP)，冻结基座。
3. Rule-based Reward (lib/reward_fns.py): 四维打分 —— 门禁短路 (R_format /
   R_grounding 只给 {0,-2.5}) + 业务判别 (R_accuracy 漏报-2.0/误报-1.0/全中+1.0)
   + CoT 长度奖励 (R_length 仅判对激活)，总值域 [-2.5, +2.0] 严格单调。
4. scale_rewards="none": 优势只减均值、不除 std —— 固定离散阶梯分除 std 会放大
   噪声组；同分组清零语义由 assert_zam_equivalence 开训前自检把守。
5. 训练中监控: eval reward 熔断 (lib/grpo_eval_fuse.py，可选) —— eval 集上 reward
   均值相对 pre-train baseline 跌幅超阈值时告警/停车，防「对齐税」退化。默认关闭，
   传 --eval_file 才启用。
6. 训练纪律: 不做自动早停 (熔断是退化保险丝，不是收敛判据)；探针口径 G=4 /
   T=0.7 / lr=2e-5 (LoRA 口径) / β=0.04 常数 KL，见 lib/grpo_common.GrpoHyper。

主流程 (与 SFT 侧序号对齐):
  1. 配置训练参数 → 2. 数据加载与体检 (整除校验/口径断言/预算打印) →
  3. 加载模型 → 4. 创建 Trainer (挂 reward + 可选熔断 callback) →
  5. 开始训练 → 6. 保存模型与训练血缘 (grpo_metrics.json)

用法:
  干跑体检 (无 GPU 可跑):
      python3 scripts/train_grpo.py --dry_run
  真实训练 (推荐走启动脚本，含依赖自检/拓扑选择/nohup):
      ./scripts/run_grpo_train.sh
  冒烟:
      GRPO_ARGS="--limit 8 --max_steps 3" ./scripts/run_grpo_train.sh
"""

import argparse
import inspect
import json
import math
import os
import sys
import time
import textwrap
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from grpo_common import (  # noqa: E402
    GrpoHyper, assert_zam_equivalence, budget_note, validate_generation_shape,
)
from grpo_dataset import (  # noqa: E402
    build_dataset_rows, build_eval_baseline, load_records)
from grpo_probe_core import extract_completion_text  # noqa: E402
from log_tee import attach_log, detach_log  # noqa: E402
from reward_fns import compute_reward  # noqa: E402
from grpo_eval_fuse import build_fuse_callback  # noqa: E402


def make_reward_fn(g: int = 4):
    """唯一的 reward 函数：格式/溯源失败必须短路，不可拆成独立奖励相加。

    TRL 侧契约: TRL 每步把整批 completion 连同数据集各列（gold_* / input_text）
    平铺传入，本函数负责「批格式 → 单条打分」的翻译。
    gold 列与 completion 数量不齐直接 raise——错位打分比不打分可怕。
    """
    def reward_fn(completions: Sequence[Any], **kwargs) -> List[float]:
        n = len(completions)
        columns = ("gold_sentiment", "gold_relevance", "gold_hard_case_type",
                   "gold_risk_json", "gold_evidence", "input_text")
        for key in columns:
            values = kwargs.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != n:
                raise ValueError(f"reward 列 {key} 须按 completion 对齐：期望 {n} 条，"
                                 f"实际 {len(values) if isinstance(values, (list, tuple)) else type(values).__name__}")

        scores = []
        for i, completion in enumerate(completions):
            gold = {
                "sentiment": kwargs["gold_sentiment"][i],
                "relevance": kwargs["gold_relevance"][i],
                "hard_case_type": kwargs["gold_hard_case_type"][i],
                "risk_categories": json.loads(kwargs["gold_risk_json"][i]),  # 落盘是 JSON 串，还原成 list
                "evidence": kwargs["gold_evidence"][i],
            }
            scores.append(compute_reward(
                # 剥壳: TRL 各版本给 str/dict/list[dict] 形态不一，统一成纯文本
                extract_completion_text(completion), kwargs["input_text"][i] or "", gold
            )["total"])
        return scores

    # g 参数仅为签名占位（组内对比/优势计算由 TRL 做，打分是逐条独立的）
    return reward_fn


def _patch_scale_rewards_none(trainer, g: int) -> None:
    """运行时把 trl 0.14 GRPOTrainer.compute_loss 的优势计算改为「只减均值、不除 std」。

    trl 0.14 源文件中该行唯一出现，直接对源码做锚点替换后重新编译绑定到实例/类。
    若未来 trl 升级导致锚点失配（find 返回 -1），立即 raise——scale_rewards 语义
    是奖励契约的一部分（见 d371a25），静默回退到「除 std」会悄悄改变实验口径。
    """
    import types
    import trl.trainer.grpo_trainer as _gt

    anchor = "advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)"
    replacement = "advantages = rewards - mean_grouped_rewards  # [scale_rewards=none patch]"
    src = inspect.getsource(_gt.GRPOTrainer.compute_loss)
    if src.count(anchor) != 1:
        raise RuntimeError(
            f"trl 0.14 advantage 锚点失配（出现 {src.count(anchor)} 次，期望 1 次）。"
            "请检查新版 trl 的 scale_rewards 参数是否已支持 \"none\"，"
            "若有则删除本 patch 改回配置式；否则更新锚点。")
    patched = textwrap.dedent(src).replace(anchor, replacement)
    # super() 零参形式依赖 __class__ 闭包 cell，exec 出来的顶层函数没有它，
    # 走到 super()._prepare_inputs 时会炸 "super(): __class__ cell not found"。
    # compute_loss 里该调用仅 2 处且都是 Trainer 方法（单继承链），直接显式化：
    # super()._prepare_inputs(x) → Trainer._prepare_inputs(self, x)。
    # 注意必须落到 Trainer（基类实现：转 tensor + 搬 GPU），不能是 GRPOTrainer——
    # GRPOTrainer 自己 override 的 _prepare_inputs 是「原样 return」，
    # 用它 prompt_inputs 留在 CPU，后续与 cuda 采样结果 cat 时 device 不匹配。
    patched = patched.replace("super()._prepare_inputs(",
                              "Trainer._prepare_inputs(self, ")
    ns = {}
    exec(compile(patched, inspect.getsourcefile(_gt.GRPOTrainer.compute_loss), "exec"),
         _gt.__dict__, ns)
    trainer.compute_loss = types.MethodType(ns["compute_loss"], trainer)
    print("[PATCH] scale_rewards=none 已生效（优势 = r - 组均值，不除 std）")


def load_merge_manifest(model_path: str):
    """只读：抽取起点模型的 SFT 合并证据（不动 manifest 本身），进 metrics 留血缘。
    真正执行合并的是 merge_sft_adapter.py，此处只读取其产物。"""
    manifest = Path(model_path) / "merge_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"error": "merge_manifest.json 解析失败"}
    return {key: data.get(key) for key in ("base", "adapter", "merged_at", "lora", "parity")}


def print_budget(h: GrpoHyper, n: int, use_vllm: bool) -> None:
    budget = h.budget(n)
    print("=" * 74)
    print(f"数据 {n} 题；每步 {budget['prompts_per_step']} 题 × G={h.num_generations} "
          f"= {budget['trajectories_per_step']} 轨迹；预计 {budget['total_steps']} 步")
    print(f"T={h.temperature} lr={h.lr} KL β={h.beta_kl}；"
          "loss_type=grpo；scale_rewards=none（只减均值、不除 std）")
    print(f"prompt 上限 {h.max_prompt_length}；{budget_note(h)}")
    print(f"vLLM: {'开' if use_vllm else '关（HF generate）'}")
    print("=" * 74)


def print_hyperparams(h: GrpoHyper, args: argparse.Namespace, config) -> None:
    """训练前全量落盘超参——一个 run 目录拷走即可复现，不用翻代码。
    在 GRPOConfig 构造后、trainer.train() 前调用。"""
    import torch
    print("=" * 74)
    print("[HP] 全量超参（*= 本次命令行覆盖或非常规项）")
    print("-" * 74)
    print(f"[HP] 采样/rollout: G={h.num_generations} T={h.temperature} "
          f"max_prompt={h.max_prompt_length} max_completion={h.max_completion_length}")
    print(f"[HP] 优化器: lr={h.lr} scheduler=cosine warmup_ratio={h.warmup_ratio} "
          f"adam_beta1=0.9 adam_beta2=0.99 weight_decay=0.01 "
          f"max_grad_norm={args.max_grad_norm} seed={h.seed}")
    print(f"[HP] RL: β_KL={h.beta_kl} scale_rewards=none* (patch, 只减均值不除std) "
          f"epsilon={h.epsilon}* (仅文档口径, trl 0.14 无clip)")
    print(f"[HP] 批次拓扑: per_device={h.per_device} grad_accum={h.grad_accum} "
          f"→ {h.per_device * h.grad_accum} prompt/步 × G{h.num_generations} "
          f"= {h.per_device * h.grad_accum * h.num_generations} 轨迹/步 "
          f"epochs={h.num_train_epochs}")
    print(f"[HP] LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
          f"target=q/k/v/o/gate/up/down_proj(7) bias=none")
    print(f"[HP] vLLM: use={args.use_vllm} "
          f"utilization={config.vllm_gpu_memory_utilization} "
          f"max_model_len={h.max_prompt_length + h.max_completion_length} (patch④注入) "
          f"device={args.vllm_device or '自动(双卡=cuda:1)'}")
    print(f"[HP] 精度: bf16={torch.cuda.is_available()} "
          f"梯度检查点={getattr(config, 'gradient_checkpointing', False)} "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '默认')}")
    print(f"[HP] eval/熔断: eval_file={args.eval_file or '训练池自动抽'} "
          f"eval_n={args.eval_n} eval_steps={args.eval_steps} "
          f"fuse_threshold={args.fuse_threshold} fuse_hard={args.fuse_hard}")
    print(f"[HP] 保存: strategy={config.save_strategy} steps={config.save_steps} "
          f"output_dir={args.output_dir}")
    print(f"[HP] 版本: trl={trl_ver} transformers={tf_ver} torch={torch.__version__}")
    print("=" * 74)


def resolve_prompt_limit(real_max: int, configured: int, hard_cap: int = 4096) -> int:
    """保留 prompt 原文；超出硬上限就拒绝启动，不允许静默截断证据来源。
    截断 prompt 会截掉待判原文 → R_grounding 的「evidence ⊂ 原文」断言必崩，
    所以宁可放宽/拒绝，绝不截断。"""
    if real_max <= configured:
        return configured
    if real_max > hard_cap:
        raise ValueError(f"最长 prompt {real_max} 超过可接受硬上限 {hard_cap}")
    return min(hard_cap, int(real_max * 1.05) + 8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based GRPO 训练")
    parser.add_argument("--data", default="data/probe_test_grpo_train_synth.jsonl")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct")
    parser.add_argument("--output_dir", default="output/grpo_lora")
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--wandb_project", default="sentiment-grpo-probe")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="激活检查点换显存：前向不存激活、反向重算。"
                             "省 ~30-40%% 激活显存，换 ~20-30%% 训练减速。"
                             "LoRA 下基座权重已冻结，配合 use_reentrant=False 安全。")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_file", default=None,
                        help="可选 eval jsonl；开启每 eval_steps 步的 reward 监控")
    parser.add_argument("--eval_n", type=int, default=8,
                        help="未传 --eval_file 时，从训练池按 (极性,难点) 去重自动抽 "
                             "eval_n 条作基线集（0 = 不自动生成）")
    parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--fuse_threshold", type=float, default=0.02)
    parser.add_argument("--fuse_hard", action="store_true",
                        help="eval reward 跌幅超阈值时停止训练（默认只告警）")
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_strategy", default="no")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_device", default=None,
                        help="trl 0.14: vLLM 采样卡, 如 cuda:1 (双卡 colocate); 不传走 auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def wandb_enabled(args) -> bool:
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) != 0:
        return False
    from wandb_setup import setup_wandb
    return setup_wandb(args.wandb_project, args.wandb_run_name,
                       args.no_wandb, user_report_to=args.report_to)


def main():
    args = build_parser().parse_args()
    args.wandb_run_name = args.wandb_run_name or f"grpo_{time.strftime('%m%d_%H%M%S')}"
    # DDP 多进程时只有 rank0 写日志文件，其余 rank 只打终端
    # （否则 N 个进程向同一文件追加会交错重复，同 train_sft_lora.py 的 rank0 门禁）
    is_rank0 = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    log_handle = attach_log(args.log_file if is_rank0 else None)
    try:
        h = GrpoHyper(
            num_generations=args.num_generations, temperature=args.temperature,
            lr=args.learning_rate, beta_kl=args.beta_kl,
            per_device=args.per_device_train_batch_size,
            grad_accum=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs, warmup_ratio=args.warmup_ratio,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length, seed=args.seed,
        )

        # 1. 加载数据集
        records = load_records(args.data)
        if args.limit:
            records = records[:args.limit]
        if not records:
            raise SystemExit("[FATAL] --limit 后数据为空")
        error = validate_generation_shape(
            len(records), h.num_generations, h.per_device,
            int(os.environ.get("WORLD_SIZE", "1")))
        if error:
            raise SystemExit(f"[FATAL] {error}")
        rows = build_dataset_rows(records)
        eval_rows = build_dataset_rows(load_records(args.eval_file)) if args.eval_file else None
        if eval_rows is None and args.eval_n > 0:
            # 未显式给 eval_file：从训练池按 (极性,难点) 去重自动抽基线集，
            # 链路同 grpo_dataset.build_eval_baseline（与训练池同源，仅熔断冒烟口径）
            eval_rows = build_dataset_rows(build_eval_baseline(records, args.eval_n))
            print(f"eval 基线集: 未指定 --eval_file，已从训练池自动抽取 "
                  f"{len(eval_rows)} 条（--eval_n 0 可关闭）")
        print(f"数据: {args.data}（{len(rows)} 条）")
        if eval_rows:
            print(f"eval 基线集: {args.eval_file}（{len(eval_rows)} 条；阈值 {args.fuse_threshold:.0%}；"
                  f"{'硬终止' if args.fuse_hard else '仅告警'}）")
            if {row["prompt"] for row in eval_rows} & {row["prompt"] for row in rows}:
                print("[WARN] eval prompts 与训练集有重叠；熔断只验证链路，不代表独立泛化/对齐税评估")
        print_budget(h, len(rows), args.use_vllm)
        assert_zam_equivalence()
        if args.dry_run:
            print(f"数据契约: {list(rows[0])}；input_text 为空: "
                  f"{sum(not row['input_text'] for row in rows)} 条")
            if eval_rows:
                print(f"eval 数据契约: {len(eval_rows)} 条；熔断 callback 将在真实训练时注册")
            print("[DRY RUN] 未加载模型/TRL；真实训练请去掉 --dry_run")
            return
        from datasets import Dataset
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 环境怪癖补丁: 本环境 (trl 0.14 + transformers 5.x) 下 trl 检测 vllm
        # 返回 (bool, version) 元组而非 bool, 恒 truthy → 未装 vllm 时
        # grpo_trainer 无条件 import vllm 而崩。import trl 前把元组修回 bool。
        try:
            import trl.import_utils as _trl_iu
            if isinstance(getattr(_trl_iu, "_vllm_available", None), tuple):
                _trl_iu._vllm_available = _trl_iu._vllm_available[0]
        except Exception:
            pass
        # 环境怪癖补丁③: vllm 0.7.3 读 tokenizer.all_special_tokens_extended,
        # transformers 5.x 已删除该属性 (实测 Qwen2Tokenizer AttributeError)。
        # vLLM 只做缓存/集合运算, 挂基类 property 返回 str 列表语义等价。
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
            if not hasattr(_PTB, "all_special_tokens_extended"):
                _PTB.all_special_tokens_extended = property(
                    lambda self: list(self.all_special_tokens))
        except Exception:
            pass
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig, TaskType
        dataset = Dataset.from_list(rows)

        # 2. 配置训练参数：先读取 tokenizer 并实测 prompt，避免把原文截断。
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lens = [len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
                for row in records]
        real_max = max(lens)
        try:
            h.max_prompt_length = resolve_prompt_limit(real_max, h.max_prompt_length)
        except ValueError as exc:
            raise SystemExit(f"[FATAL] {exc}") from exc
        print(f"prompt 长度：最长 {real_max} / 均值 {sum(lens)//len(lens)} / 上限 {h.max_prompt_length}")
        wandb_on = wandb_enabled(args)   # setup_wandb 已注入 WANDB_PROJECT/NAME
        # TRL 1.16 的 GRPOConfig 只收 warmup_steps；按 ratio 折算成绝对步数
        steps_per_epoch = math.ceil(len(records) / (h.per_device * h.grad_accum))
        total_steps = args.max_steps or steps_per_epoch * h.num_train_epochs
        warmup_steps = int(h.warmup_ratio * total_steps)
        config = GRPOConfig(
            output_dir=args.output_dir, run_name=args.wandb_run_name,
            learning_rate=h.lr, adam_beta1=0.9, adam_beta2=0.99,
            weight_decay=0.01, warmup_steps=warmup_steps,
            lr_scheduler_type="cosine", logging_steps=1,
            bf16=torch.cuda.is_available(),
            per_device_train_batch_size=h.per_device,
            gradient_accumulation_steps=h.grad_accum,
            num_generations=h.num_generations,
            max_completion_length=h.max_completion_length,
            num_train_epochs=h.num_train_epochs, beta=h.beta_kl,
            max_grad_norm=args.max_grad_norm,
            # 激活检查点（--gradient_checkpointing 开启）：LoRA 冻结基座下安全，
            # 显存紧（如 24G 卡跑 4B+GRPO）时的首选开关
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
                if args.gradient_checkpointing else None,
            save_strategy="steps" if args.save_steps > 0 else args.save_strategy,
            save_steps=args.save_steps if args.save_steps > 0 else 500,
            seed=h.seed, report_to="wandb" if wandb_on else "none",
            # 优势口径 scale_rewards="none"（只减均值、不除 std）不在 config 传——
            # trl 0.14 无此参数，由 _patch_scale_rewards_none 在 trainer 构造后打补丁。
            # trl 0.14 已移除 max_prompt_length / vllm_mode：prompt 截断由数据侧
            # （resolve_prompt_limit 已实测并对齐）保证，vllm_device 由 --vllm_device 接管
            log_on_each_node=False, use_vllm=args.use_vllm,
            # eval_strategy 必须为 no：Trainer 内置 eval 走 prediction_step，
            # GRPOTrainer 未 override，list 形态 eval batch 会炸（见 trainer 构造处注释）。
            # reward 监控/熔断由 fuse_callback 的 on_evaluate 自行采样实现。
            eval_strategy="no",
            eval_steps=args.eval_steps if eval_rows else 500,
            per_device_eval_batch_size=h.per_device,
            # vLLM 独占 cuda:1（双卡拓扑）：0.9 让 KV cache ~12.3G、并发 ~29x，
            # 32 条 rollout 一波发完（0.7 只有 ~21x，要分两波）；
            # 单卡 colocate 同卡模式才需要 0.3（给训练留 70%）
            vllm_gpu_memory_utilization=0.9,
            max_steps=args.max_steps or 0, temperature=h.temperature,
            # ε/clip：trl 0.14 无 epsilon 参数，其 per_token_loss 直接用
            # exp(logps - logps.detach())（无截断的 policy-gradient 形式，等价 ε→∞），
            # 概率比截断由 β=0.04 的常数 KL 承担。h.epsilon 保留仅作文档口径。
            # 注：0.14 该形式实为 importance-ratio 无 clip 版本，升级 trl 后可恢复 epsilon=h.epsilon。
        )

        # 3. 加载模型；tokenizer 已在步骤 2 加载并测长。
        if not (Path(args.model_name_or_path) / "merge_manifest.json").exists():
            print("[WARN] 未发现 SFT 合并记录；请确认从裸基座起训是有意的对照实验")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True)
        # 环境怪癖补丁②: trl 0.14 写 model.warnings_issued[...] (transformers
        # 5.x 已删该属性 → AttributeError)。预挂空 dict, 仅影响警告抑制。
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        # 环境怪癖补丁④: trl 0.14 内部 LLM(...) 不传 max_model_len → vLLM 按
        # 模型原始上下文 262144 做 profiling dummy run，4D causal mask 要 128GiB
        # 直接 OOM（trl 0.14 的 GRPOConfig 也没有 vllm_max_model_len 旋钮）。
        # 包装 vllm.LLM：未显式指定时注入 max_model_len = prompt 上限 + 输出预算
        # （本任务 2048+512=2560，实测 max prompt 795，余量充足）。
        if args.use_vllm:
            import vllm as _vllm
            _orig_llm_init = _vllm.LLM.__init__
            _vllm_max_len = h.max_prompt_length + h.max_completion_length

            def _patched_llm_init(self, *a, **kw):
                if kw.get("max_model_len") is None and "max_model_len" not in kw:
                    kw["max_model_len"] = _vllm_max_len
                    print(f"[PATCH] vLLM max_model_len 注入 {_vllm_max_len} "
                          f"(prompt {h.max_prompt_length} + completion {h.max_completion_length})")
                return _orig_llm_init(self, *a, **kw)
            _vllm.LLM.__init__ = _patched_llm_init

        # 4. 创建 Trainer（可选注册评估熔断 callback）
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"], bias="none")
        trainer = GRPOTrainer(
            model=model, peft_config=peft_config,
            reward_funcs=make_reward_fn(h.num_generations), args=config,
            train_dataset=dataset,
            # eval 熔断不走 Trainer 内置 eval：eval_dataset 传 None、eval_strategy=no。
            # 原因：GRPOTrainer 未 override prediction_step，Trainer 的 evaluation_loop
            # 会拿 eval batch（list 形态、含 gold list 列）喂 model.forward ——
            # 'list' object has no attribute 'get'（实测 run_20261028_114227，第 2 步
            # eval_steps 触发）。reward 监控由 fuse_callback 自行采样打分（见下），
            # 与 Trainer 的 eval 机制完全解耦。
            eval_dataset=None,
            processing_class=tokenizer)

        # trl 0.14 无 scale_rewards 开关：compute_loss 硬编码优势 = (r-组均值)/(组std+1e-4)。
        # 我们要「只减均值、不除 std」（固定离散阶梯分除 std 会放大噪声组，
        # 见 lib/reward_fns.group_advantages 注释；reward 侧预乘在代数上无法精确抵消——
        # 分子分母被同时缩放，eps 项权重反而失真，已实测否决）。
        # → 运行时 monkeypatch compute_loss：源码中该行仅一处
        #   advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # 改为 advantages = rewards - mean_grouped_rewards（等价新版 scale_rewards="none"）。
        # 同分/全对/全错组中心化后全 0，优势自然清零，与 ZAM 语义一致。
        # 升级 trl 后此 patch 的锚字符串可能失配 → fail-loud，不许静默跳过。
        _patch_scale_rewards_none(trainer, h.num_generations)

        # 训练前全量落盘超参（config 已构造、patch 已生效，此时打印的是真实值）
        import trl as _trl, transformers as _tf
        global trl_ver, tf_ver
        trl_ver, tf_ver = _trl.__version__, _tf.__version__
        print_hyperparams(h, args, config)

        if eval_rows:
            fuse_callback = build_fuse_callback(
                fuse_threshold=args.fuse_threshold, fuse_hard=args.fuse_hard,
                reward_fn=make_reward_fn(h.num_generations), g=h.num_generations,
                eval_rows=eval_rows)
            fuse_callback._attach(trainer)
            fuse_callback.initialize_baseline(trainer)
            trainer.add_callback(fuse_callback)

        # 5. 开始训练
        budget = h.budget(len(records))
        print(f"开始 GRPO 训练：预计 {budget['total_steps']} 步 × "
              f"{budget['trajectories_per_step']} 轨迹/步")
        start = time.time()
        trainer.train()
        elapsed = time.time() - start

        # 6. 保存模型和训练血缘
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output))
        metrics = {
            "n_prompts": len(records), "total_steps": budget["total_steps"],
            "trajectories_per_step": budget["trajectories_per_step"],
            "hyperparams": {**{key: getattr(h, key) for key in (
                "num_generations", "temperature", "lr", "beta_kl", "per_device",
                "grad_accum", "max_completion_length", "max_prompt_length", "seed")},
                "base": args.model_name_or_path,
                "merge": load_merge_manifest(args.model_name_or_path)},
            "elapsed_sec": round(elapsed, 1),
        }
        (output / "grpo_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"模型已保存: {output}；指标: {output / 'grpo_metrics.json'}；耗时: {elapsed:.1f}s")
    except BaseException as exc:
        # 先在 tee 还挂着时把致命错误写进日志文件，再 detach（detach 后 print 只进终端）
        print(f"[FATAL] 训练异常退出: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        detach_log(log_handle)


if __name__ == "__main__":
    main()
