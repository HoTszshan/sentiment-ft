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


def _log_eval_metric(trainer, value: float, is_baseline: bool = False):
    """把 eval reward 均值推到 trainer 日志（wandb/tensorboard 均可收）。

    on_step_end 里 trainer.state.global_step 已是当前步，直接标该步；
    baseline（训练前）标 step 0。报告失败但不抛——熔断主链路不因日志而断。
    """
    try:
        step = trainer.state.global_step if trainer is not None else 0
        trainer.log({"eval/reward_mean_baseline" if is_baseline else "eval/reward_mean": value}, step)
    except Exception as exc:
        print(f"[WARN] eval 指标推送日志失败（不影响训练）: {exc}")


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

    # ── 数值健康熔断 (2026-11-03, G2 坍塌实证) ──
    # G2 (β=0) 实测: KL 步间跳变是奖励坍塌唯一有效前兆（0.002→1.45），NaN 出现时
    # 训练已死 145 步。判据（踩坑笔记「三看」）:
    #   1) KL 跳变: 相对上一次记录值膨胀 > kl_jump_factor 且绝对值 > kl_abs_limit
    #      （防早期小基数误报: 0.0005→0.02 是 40× 但绝对值无害）;
    #   2) KL 绝对值连续持续超限（指数发散期 1e8 级别）;
    #   3) grad_norm/loss NaN/inf 连续 ≥ nan_patience 步（已死即停，省卡时）。
    # env 开关: GRPO_NAN_FUSE=1 启用（默认关，保持旧行为）；fuse_hard 控制是否真停车。
    nan_state: Dict[str, Any] = {"last_kl": None, "nan_run": 0, "kl_high_run": 0}
    kl_jump_factor = float(os.environ.get("GRPO_KL_JUMP_FACTOR", "100"))
    kl_abs_limit = float(os.environ.get("GRPO_KL_ABS_LIMIT", "10"))
    nan_patience = int(os.environ.get("GRPO_NAN_PATIENCE", "3"))
    nan_fuse_enabled = os.environ.get("GRPO_NAN_FUSE", "0") == "1"

    def _check_numeric_health(trainer, state, control):
        """每步检查 grad_norm/loss 的 NaN 与 KL 发散；返回 True 表示已请求停车。"""
        if not nan_fuse_enabled:
            return False
        metrics = getattr(state, "log_history", [])
        if not metrics:
            return False
        last = metrics[-1]
        grad = last.get("grad_norm")
        loss = last.get("loss")
        # 1) NaN/inf 早停: 连续 nan_patience 步非有限值 → 停车
        bad = (grad is not None and not (abs(grad) < float("inf"))) or \
              (loss is not None and not (abs(loss) < float("inf")))
        if bad:
            nan_state["nan_run"] += 1
        else:
            nan_state["nan_run"] = 0
        if nan_state["nan_run"] >= nan_patience:
            print(f"[FUSE-NAN] grad_norm/loss 连续 {nan_state['nan_run']} 步非有限值 "
                  f"(grad={grad}, loss={loss})——训练已数值死亡，停车")
            control.should_training_stop = True
            return True
        # 2) KL 前兆: 跳变或持续超限
        kl = last.get("kl")
        if kl is not None and isinstance(kl, (int, float)) and kl == kl:
            prev = nan_state["last_kl"]
            nan_state["last_kl"] = kl
            if prev is not None:
                jumped = (kl > kl_abs_limit and prev <= kl_abs_limit) or \
                         (prev > 0 and kl / prev > kl_jump_factor and kl > kl_abs_limit)
                if jumped:
                    # 记录嫌疑但不立即停车——G1'' 实测存在孤立尖峰 (step 362 冲 1285 后回落)，
                    # 立即杀会把「健康训练的单步毛刺」误判成坍塌。
                    # 连续 kl_high_run 次 > 限值才停（见下方持续超限分支），孤峰自愈不罚。
                    print(f"[WARN-KL] KL 步间跳变 {prev:.4g} → {kl:.4g} "
                          f"(>x{kl_jump_factor:.0f} 且绝对值超 {kl_abs_limit})——坍塌前兆嫌疑，"
                          f"观察 {nan_patience} 步")
            # 持续超限 (跳变 + 连续在高位 = 真发散; 孤立尖峰回落即清零)
            nan_state["kl_high_run"] = nan_state["kl_high_run"] + 1 if kl > kl_abs_limit else 0
            if nan_state["kl_high_run"] >= nan_patience:
                print(f"[FUSE-KL] KL 连续 {nan_state['kl_high_run']} 步 > {kl_abs_limit} "
                      f"(当前 {kl:.4g})——指数发散期, 停车")
                if fuse_hard:
                    control.should_training_stop = True
                    return True
        return False

    class _FuseCallback(TrainerCallback):
        # 用 on_step_end 周期触发而不是 on_evaluate：Trainer 内置 eval 已关
        # （eval_strategy=no，prediction_step 会炸 list batch），on_evaluate 永远
        # 不会被调用。本 callback 自带采样打分，按 state.global_step % eval_steps 节流。
        def on_step_end(self, args, state, control, **kwargs):
            trainer = holder.get("trainer")
            if trainer is None or state.global_step == 0:
                return control
            # 数值健康检查每步都做（NaN/KL 坍塌不等 eval 周期——G2 教训:
            # eval 间隔内可以走完 0.0005→1e34 的全部发散）
            if _check_numeric_health(trainer, state, control):
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
                _log_eval_metric(trainer, current, is_baseline=True)
                return control

            baseline = fuse_state["baseline"]
            if should_fuse(baseline, current, fuse_threshold):
                print(f"[FUSE] eval reward 均值 {current:.4f} 相对 baseline "
                      f"{baseline:.4f} 跌幅超 {fuse_threshold:.0%}")
                _log_eval_metric(trainer, current)
                if fuse_hard:
                    control.should_training_stop = True   # 硬熔断：直接停车
            else:
                print(f"[EVAL] reward 均值 = {current:.4f}（baseline {baseline:.4f}）")
                _log_eval_metric(trainer, current)
            return control

    callback = _FuseCallback()

    def initialize_baseline(trainer):
        """训练前用当前策略评估一次，避免把首个训练中 eval 错当基线。"""
        holder["trainer"] = trainer
        baseline = _eval_reward_mean(trainer, reward_fn, g, eval_rows)
        fuse_state["baseline"] = baseline
        print(f"[EVAL] pre-train baseline reward 均值 = {baseline:.4f}")
        _log_eval_metric(trainer, baseline, is_baseline=True)
        return baseline

    callback._attach = lambda trainer: holder.__setitem__("trainer", trainer)
    callback.initialize_baseline = initialize_baseline
    return callback


def resolve_eval_max_new(cfg) -> int:
    """
    eval生成长度和rollout统一为max_completion=512。
    旧兜底200会截断CoT+JSON，格式校验误判、奖励锁死-2.5，熔断失效；改为兜底512，宁长勿截。
    """
    return getattr(cfg, "max_completion_length", None) or 512


def _eval_reward_mean(trainer, reward_fn, g: int, eval_rows=None) -> float:
    """
    对 eval 集逐题采一条 completion，用 reward_fn 打分取均值。

    与训练同一个 reward_fn —— 「退化」的定义就是奖励分本身在掉。
    每题只采 1 条（eval 求稳定均值，无需 G 组内对比）。

    eval_rows: 元数据行列表（含 gold_* 列），必传——trainer.eval_dataset 已置 None
    （Trainer 内置 eval 走 prediction_step 会炸 list batch，见 train_grpo 注释），
    prompt 与 gold 全部从这里拿。

    生成路径优先级：
      1. vLLM（trainer.llm 存在且可用）——G1 实测教训：HF generate 逐题串行 8 题
         ~28 min/次，16 次 eval 烧掉 8.8h 里的 7.6h；vLLM 同路径批量 greedy 量级
         分钟内。weight 同步复用 trl GRPOTrainer.compute_loss 的
         `llm_engine.model_executor.driver_worker.model_runner.model.load_weights`
         调用链（trl 0.14 源码实测），本步 eval 前把当前策略权重推进 vLLM。
      2. 兜底 HF generate（trainer.llm 缺失 = 未开 vLLM 的单卡拓扑）——保持旧路径。
    """
    import torch

    tokenizer = trainer.processing_class
    max_new = resolve_eval_max_new(trainer.args)
    # # [P0 20261030] 
    # eval改用greedy解码：do_sample=True带来的采样噪声会淹没2%熔断阈值，
    # 导致判定失效；greedy保证分数稳定，仅存浮点噪声。探测脚本run_grpo_probe.py保留采样，用于衡量探索多样性。

    llm = getattr(trainer, "llm", None)
    if llm is not None:
        return _eval_reward_mean_vllm(
            trainer, llm, tokenizer, reward_fn, eval_rows, max_new)
    return _eval_reward_mean_hf(
        trainer, tokenizer, reward_fn, eval_rows, max_new)


def _eval_reward_mean_vllm(trainer, llm, tokenizer, reward_fn, eval_rows,
                           max_new) -> float:
    """vLLM 路径：先把当前策略权重同步进 vLLM 引擎，再批量 greedy 采样打分。

    ⚠️ [P0 20261103] PEFT 权重同步修复（G2 实证 eval 盲区）：
    PEFT 模型的 state_dict() 里 base_layer = 冻结基座原值，LoRA 增量在独立的
    lora_A/B 键；vLLM load_weights() 按名字加载，lora_A/B 无对应参数被静默忽略
    → 旧链路同步进 vLLM 的永远是裸基座，eval 从头到尾量的是同一个冻结策略
    （G1/G1''/G2 三轮 run 内 eval 恒定为铁证）。
    修复：load_weights 前 merge_adapter() 让 base_layer 真正带上 LoRA 增量，
    同步完再 unmerge_adapter() 还原训练态。unwrap 传 is_peft_model=False
    （我们自己管理 merge/unmerge，避免 accelerate 的 disable_adapter 干扰）。
    """
    import torch
    from vllm import SamplingParams
    from trl.models import unwrap_model_for_generation

    model = getattr(trainer, "model_wrapped", None) or trainer.model
    was_training = model.training
    model.eval()
    totals = []
    merged = False
    try:
        with torch.no_grad():
            if not eval_rows:
                raise ValueError("eval_rows 为空：独立 eval callback 需要 eval_rows")
            # 1) 权重同步——先合并 LoRA 增量进 base_layer，再取 state_dict
            peft_model = model
            try:
                peft_model.merge_adapter()
                merged = True
            except AttributeError:
                merged = False   # 非 PEFT 模型（全参训练），直接走旧链路
            try:
                with unwrap_model_for_generation(model, trainer.accelerator,
                                                 is_peft_model=False) as unwrapped:
                    state_dict = unwrapped.state_dict()
                llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            finally:
                if merged:
                    peft_model.unmerge_adapter()   # 还原训练态，防梯度污染
            # 2) 批量 greedy（64 题 → 1 次 generate；greedy 与温度无关）
            prompts = [row["prompt"] for row in eval_rows]
            sp = SamplingParams(temperature=0.0, max_tokens=max_new)
            outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            # 3) 打分（与 HF 路径同一 reward_fn 契约）
            for row, out in zip(eval_rows, outputs):
                # skip_special_tokens=True对齐训练解码逻辑，避免特殊token判定为垃圾，造成奖励偏差、干扰熔断
                text = tokenizer.decode(
                    out.outputs[0].token_ids, skip_special_tokens=True)
                reward_kwargs = {key: [row.get(key)] for key in (
                    "gold_sentiment", "gold_relevance", "gold_hard_case_type",
                    "gold_risk_json", "gold_evidence", "input_text")}
                totals.append(reward_fn([text], **reward_kwargs)[0])
    finally:
        # eval 完恢复训练态（与 HF 路径同款纪律）
        model.train(was_training)
    return sum(totals) / max(1, len(totals))


def _eval_reward_mean_hf(trainer, tokenizer, reward_fn, eval_rows, max_new) -> float:
    """HF generate 兜底路径（未开 vLLM 的拓扑）：逐题贪心，行为与 G1 前一致。"""
    import torch

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
            max_prompt = getattr(trainer.args, "max_prompt_length", 2048) or 2048
            meta_rows = eval_rows
            for row in meta_rows:
                prompt = row["prompt"]
                if not isinstance(prompt, str):
                    raise TypeError("独立 eval callback 只接受字符串 prompt")
                encoded = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt)
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                output = model.generate(
                    **encoded, do_sample=False,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                # 只 decode 新生成部分（跳过 prompt 前缀）
                # skip_special_tokens=True 对齐训练 reward 路径
                # (trl grpo_trainer decode 同款)，口径统一防熔断阈值偏差
                text = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                reward_kwargs = {key: [row.get(key)] for key in (
                    "gold_sentiment", "gold_relevance", "gold_hard_case_type",
                    "gold_risk_json", "gold_evidence", "input_text")}
                totals.append(reward_fn([text], **reward_kwargs)[0])
    finally:
        # eval 完必须恢复训练态，否则 dropout 等行为在后续训练中出错
        model.train(was_training)
    return sum(totals) / max(1, len(totals))
