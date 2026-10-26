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


def build_fuse_callback(fuse_threshold: float, fuse_hard: bool, reward_fn, g: int):
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
        def on_evaluate(self, args, trainer_state, control, **kwargs):
            trainer = holder.get("trainer")
            if trainer is None:
                return control
            try:
                current = _eval_reward_mean(trainer, reward_fn, g)
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
        baseline = _eval_reward_mean(trainer, reward_fn, g)
        fuse_state["baseline"] = baseline
        print(f"[EVAL] pre-train baseline reward 均值 = {baseline:.4f}")
        return baseline

    callback._attach = lambda trainer: holder.__setitem__("trainer", trainer)
    callback.initialize_baseline = initialize_baseline
    return callback


def _eval_reward_mean(trainer, reward_fn, g: int) -> float:
    """
    对 eval_dataset 逐题采一条 completion，用 reward_fn 打分取均值。

    与训练同一个 reward_fn —— 「退化」的定义就是奖励分本身在掉。
    每题只采 1 条（eval 求稳定均值，无需 G 组内对比）。
    """
    import torch

    tokenizer = trainer.processing_class
    dataset = trainer.eval_dataset
    cfg = trainer.args
    max_new = getattr(cfg, "max_completion_length", 200) or 200
    temperature = getattr(cfg, "temperature", 0.7) or 0.7
    model = getattr(trainer, "model_wrapped", None) or trainer.model
    was_training = model.training
    model.eval()
    totals = []
    try:
        with torch.no_grad():
            for row in dataset:
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
