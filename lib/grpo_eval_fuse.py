#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可选的 GRPO eval reward 熔断 callback。

这是独立扩展模块，训练主脚本不导入、不自动挂载。调用方须显式构造 callback，
并在 Trainer 创建后调用 callback._attach(trainer)。
"""
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_fuse_callback(fuse_threshold: float, fuse_hard: bool, reward_fn, g: int,
                        eval_rows=None):
    """
    建立 eval reward 熔断 callback；超阈值时告警，可选停止训练。

    用法（三步）:
        cb = build_fuse_callback(0.02, True, reward_fn, g=4)
        trainer.add_callback(cb)
        cb.initialize_baseline(trainer)   # 训练前摸底建基线，推荐
    首次 eval 且无基线时只记录不判定——缺数据不能当退化。
    判定逻辑在 grpo_common.should_fuse（纯函数），本模块只负责接线。
    """
    from transformers import TrainerCallback
    from grpo_common import should_fuse

    fuse_state: Dict[str, Any] = {"baseline": None}   # 闭包状态：基线分
    holder: Dict[str, Any] = {"trainer": None}        # 闭包状态：trainer 引用

    class _FuseCallback(TrainerCallback):
        # 用 on_step_end 周期触发而不是 on_evaluate：Trainer 内置 eval 已关
        # （eval_strategy=no，prediction_step 会炸 list batch），on_evaluate 永远
        # 不会被调用。本 callback 自带采样打分，按 state.global_step % eval_steps 节流。
        def on_step_end(self, args, state, control, **kwargs):
            trainer = holder.get("trainer")
            if trainer is None or state.global_step == 0:
                return control
            eval_steps = getattr(args, "eval_steps", 0) or 0
            if eval_steps <= 0 or state.global_step % eval_steps != 0:
                return control
            try:
                current = _eval_reward_mean(trainer, reward_fn, g, eval_rows)
            except Exception as exc:
                print(f"[WARN] eval 打分失败，跳过本轮熔断判定: {exc}")
                return control

            if fuse_state["baseline"] is None:
                # 未显式初始化 pre-train baseline 时，以首次 eval 建立基线并结束本回调。
                fuse_state["baseline"] = current
                print(f"[EVAL] baseline reward 均值 = {current:.4f}")
                return control

            baseline = fuse_state["baseline"]
            if should_fuse(baseline, current, fuse_threshold):
                print(f"[FUSE] eval reward 均值 {current:.4f} 相对 baseline "
                      f"{baseline:.4f} 跌幅超 {fuse_threshold:.0%}")
                if fuse_hard:
                    control.should_training_stop = True   # 硬熔断：直接停车
            else:
                print(f"[EVAL] reward 均值 = {current:.4f}（baseline {baseline:.4f}）")
            return control

    callback = _FuseCallback()

    def initialize_baseline(trainer):
        """训练前用当前策略评估一次，避免把首个训练中 eval 错当基线。"""
        holder["trainer"] = trainer
        baseline = _eval_reward_mean(trainer, reward_fn, g, eval_rows)
        fuse_state["baseline"] = baseline
        print(f"[EVAL] pre-train baseline reward 均值 = {baseline:.4f}")
        return baseline

    callback._attach = lambda trainer: holder.__setitem__("trainer", trainer)
    callback.initialize_baseline = initialize_baseline
    return callback


def _eval_reward_mean(trainer, reward_fn, g: int, eval_rows=None) -> float:
    """
    对 eval 集逐题采一条 completion，用 reward_fn 打分取均值。

    与训练同一个 reward_fn —— 「退化」的定义就是奖励分本身在掉。
    每题只采 1 条（eval 求稳定均值，无需 G 组内对比）。

    eval_rows: 元数据行列表（含 gold_* 列），必传——trainer.eval_dataset 已置 None
    （Trainer 内置 eval 走 prediction_step 会炸 list batch，见 train_grpo 注释），
    prompt 与 gold 全部从这里拿。
    """
    import torch

    tokenizer = trainer.processing_class
    cfg = trainer.args
    max_new = getattr(cfg, "max_completion_length", 200) or 200
    temperature = getattr(cfg, "temperature", 0.7) or 0.7
    model = getattr(trainer, "model_wrapped", None) or trainer.model
    was_training = model.training
    model.eval()
    totals = []
    try:
        with torch.no_grad():
            if not eval_rows:
                raise ValueError("eval_rows 为空：独立 eval callback 需要 eval_rows "
                                 "(pre-train baseline 或 eval_file)，不能依赖 "
                                 "trainer.eval_dataset（已置 None）")
            meta_rows = eval_rows
            for row in meta_rows:
                prompt = row["prompt"]
                if not isinstance(prompt, str):
                    raise TypeError("独立 eval callback 只接受字符串 prompt")
                encoded = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=getattr(cfg, "max_prompt_length", 2048) or 2048)
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                output = model.generate(
                    **encoded, do_sample=True, temperature=temperature,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                # 只 decode 新生成部分（跳过 prompt 前缀）
                text = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=False)
                reward_kwargs = {key: [row.get(key)] for key in (
                    "gold_sentiment", "gold_relevance", "gold_hard_case_type",
                    "gold_risk_json", "gold_evidence", "input_text")}
                totals.append(reward_fn([text], **reward_kwargs)[0])
    finally:
        # eval 完必须恢复训练态，否则 dropout 等行为在后续训练中出错
        model.train(was_training)
    return sum(totals) / max(1, len(totals))
