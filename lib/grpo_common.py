#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 共享层 · 纯 Python (零 torch/trl 依赖，可独立单测)

职责:
  1. k 分桶: 同题 G=4 采样 → 数出「判对且溯源过门禁」的轨迹数 k ∈ {0..4}
  2. 训练预算: G / T / lr / β / 有效 batch / 步数的推导与打印 (手册 1.3 口径)
  3. 防御自检与熔断: 优势口径回归闸 (ZAM) / 对齐税熔断 / TRL 形状约束校验

数据契约 (加载/校验/展平 TRL 训练行) 已拆到 grpo_dataset.py —— 本模块不再持有。

真理源:
  MVP/notes/GRPO训练实战手册.md §1.1 LoRA 旁路 / §1.3 超参 / §1.4 闭环 / §1.5 三闸
  MVP/notes/舆情负面精判_GRPO数据准备实战手册.md §四 难度分流 (只留 k∈{1,2,3})
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_fns import (
    R_GATE_SHORT_CIRCUIT, compute_reward, zero_advantage_masking,
)  # noqa: E402

# 唯一可留进训练池的 k 桶 (数据手册 §四)
KEEP_K = (1, 2, 3)

# ============================================================
# 1. k 分桶
# ============================================================

def group_totals(totals: Sequence[float], g: int) -> List[List[float]]:
    """把一维 (n*G) 分数切成 n 个组。长度不整除直接报错。"""
    vals = list(totals)
    if g <= 0:
        raise ValueError("G 必须 ≥ 1")
    if len(vals) % g:
        raise ValueError(f"轨迹数 {len(vals)} 不能被 G={g} 整除")
    return [vals[i:i + g] for i in range(0, len(vals), g)]


def score_one(completion: str, input_text: str, gold: Dict[str, Any]) -> Dict[str, Any]:
    """单条 completion → 四维总分 + 明细 (薄封装，方便打日志)。"""
    return compute_reward(completion, input_text, gold)


def k_of(details: Iterable[Dict[str, Any]]) -> int:
    """
    k = 组内「判对 且 溯源过门禁」的轨迹数。
    只有 compute_reward 的 detail.get('accuracy_pass') 为 True 才算判对
    (门禁短路的轨迹 detail 里没有 accuracy_pass → 记 0)。
    """
    return sum(1 for d in details if isinstance(d, dict) and d.get("accuracy_pass"))


def bucket_label(k: int, g: int = 4) -> str:
    """k → 数据手册 §四 的分流处置。"""
    if k <= 0:
        return "drop_k0"          # 全错: RL 无米下锅，退回 SFT 补课
    if k >= g:
        return "drop_k_max"       # 全对: 组内方差为 0，白烧卡时
    return "keep" if k in KEEP_K else "drop"


def summarize_buckets(rows: Sequence[Dict[str, Any]], g: int = 4) -> Dict[str, Any]:
    """
    rows: 每行 {k, ...meta} → 分桶统计 (训练池规模 + 各难点桶健康度)。
    """
    hist = {i: 0 for i in range(g + 1)}
    keep_by_hc: Dict[str, int] = {}
    keep_by_sent: Dict[str, int] = {}
    for r in rows:
        k = int(r.get("k", 0))
        hist[k] = hist.get(k, 0) + 1
        if bucket_label(k, g) == "keep":
            hc = r.get("hard_case_type", "?")
            s = r.get("sentiment", "?")
            keep_by_hc[hc] = keep_by_hc.get(hc, 0) + 1
            keep_by_sent[s] = keep_by_sent.get(s, 0) + 1
    keep = sum(hist[i] for i in KEEP_K if i <= g)
    return {
        "n": len(rows),
        "g": g,
        "hist": {str(k): hist.get(k, 0) for k in range(g + 1)},
        "keep": keep,
        "drop_k0": hist.get(0, 0),
        "drop_k_max": hist.get(g, 0),
        "keep_ratio": round(keep / len(rows), 4) if rows else 0.0,
        "keep_by_hard_case": dict(sorted(keep_by_hc.items(), key=lambda x: -x[1])),
        "keep_by_sentiment": dict(sorted(keep_by_sent.items(), key=lambda x: -x[1])),
    }


# ============================================================
# 2. 训练预算 (手册 §1.3 的每个数都要能算出来源)
# ============================================================

@dataclass
class GrpoHyper:
    """GRPO 特有超参。✅=手册标定 / [设计值]=有推导未跑。"""
    num_generations: int = 4                 # G ✅ 手册 1.3
    temperature: float = 0.7                 # T ✅ 手册 1.3
    lr: float = 2e-5                         # [设计值] LoRA 口径：≈ SFT 1e-4 的 1/5；全参 RLHF 的 1e-6~5e-6 不适用
    beta_kl: float = 0.04                    # β ✅ 手册 1.3 常数 KL
    per_device: int = 4                      # 一步内同时进来的 prompt 数
    grad_accum: int = 2                      # → 8 题/步 × G=4 = 32 条轨迹/步 ✅
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    max_prompt_length: int = 2048            # 2560(SFT max_seq) - 512(completion)；实测 max=774，2.6× 余量
    max_completion_length: int = 512         # 实测整段输出 max=383；见下方 budget_note()
    epsilon: float = 0.2                     # PPO 截断替代项，裁概率比 r 不裁优势 A ✅ TRL 默认显式化
    seed: int = 42

    @property
    def prompts_per_step(self) -> int:
        return self.per_device * self.grad_accum

    @property
    def trajectories_per_step(self) -> int:
        """有效 batch = 题数 × G。"""
        return self.prompts_per_step * self.num_generations

    def steps(self, n_prompts: int) -> int:
        """总更新步数 = ceil(题数/每步题数) × epoch 数（epoch 取整，与 Trainer 一致）。"""
        if n_prompts <= 0:
            return 0
        return math.ceil(n_prompts / self.prompts_per_step) * math.ceil(self.num_train_epochs)

    def budget(self, n_prompts: int) -> Dict[str, Any]:
        per_ep = math.ceil(n_prompts / self.prompts_per_step)
        total = per_ep * math.ceil(self.num_train_epochs)
        return {
            "n_prompts": n_prompts,
            "prompts_per_step": self.prompts_per_step,
            "num_generations": self.num_generations,
            "trajectories_per_step": self.trajectories_per_step,
            "steps_per_epoch": per_ep,
            "total_steps": total,
            "warmup_steps": int(total * self.warmup_ratio),
            "lr": self.lr, "temperature": self.temperature, "beta_kl": self.beta_kl,
            "epsilon": self.epsilon,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
        }


def budget_note(h: GrpoHyper) -> str:
    """
    输出预算怎么定的（实测数据，不是拍脑袋）:
      SFT 73 条真分词：整段输出 p50 263 / p95 335 / p99 383 / max 383 tokens。
      整段 = CoT + JSON。其中 JSON 段约 170 字符 ≈ 81 tokens（落在 65~105 口径内），
      CoT 段 p50 约 177 / max 约 257 tokens。

      ⚠️ 两个曾经踩过的坑：
      1. 把手册「65~105 tokens」当成整段预算 → 按 105 或 200 截 → **99% 的样本被砍在
         CoT 或 JSON 中间** → json.loads 失败 → R_format 全池 -2.5 短路 → 训练废掉。
         那一格说的是**扁平 JSON 段**的长度，不含 CoT。
      2. 以为「CoT ≤120」→ 实际 CoT p50 已 177。奖励阈值另议，但截断线必须按实测定。

      现任 512 = 实测 max 383 × 1.34，留出生成抖动余量；生成遇 EOS 即停，
      放大上限对短输出零成本，只抬高长尾的天花板。
    """
    return (f"输出预算 {h.max_completion_length}（实测整段 max 383 = CoT ~257 + JSON ~81，留余量）；"
            f"65~105 是 JSON 段口径，不是整段")


# ============================================================
# 3. 防御自检与熔断
# ============================================================

def should_fuse(baseline: Optional[float], current: Optional[float],
                threshold: float = 0.02) -> bool:
    """
    对齐税熔断判定（评估手册模块一④ / GRPO 手册模块 3 闸②）。

    eval 基线集 reward 均值相对首次 eval 的跌幅超过 threshold（默认 2%）→ 熔断。
    None 的语义：baseline 未建立（首轮 eval）或当前 eval 缺数 → 不熔断，
    缺数据不能当成退化，也不能当成健康，只能不动作。
    """
    if baseline is None or current is None:
        return False
    drop = baseline - current
    if baseline <= 0:
        # 基线非正（比如全在罚分区）时跌幅比率无意义，退化为绝对差判定
        return drop > threshold
    # 浮点噪声容差：跌幅恰在阈值 ±1e-9 内不熔断
    return drop / abs(baseline) > threshold + 1e-9


def assert_zam_equivalence() -> bool:
    """
    启动自检: 优势归一化约定 + Zero-Advantage Masking 的处置语义。

    ⚠️ 边界说清楚: 本函数**不 import TRL**, 因此不是「与 TRL 对拍」, 它也做不到——
    它断言的是我方口径本身。真正让 TRL 走对口径的是 `GRPOConfig(scale_rewards="none")`
    那个显式传参；若字段缺失会由 GRPOConfig 构造直接报错，不做静默回退。

    断言三件事:
      ① 同分组 (全对 / 全错 / 任意同分) → 整组优势清零
      ② 非同分组 → 不得清零 (否则 R_length 的部分得分区失去意义)
      ③ 优势必须是**减均值后的原值**, 不是再除以 std 的标准化值;
         且「只差一档」的垃圾组幅度必须小于真有判别信息的组 —— 这条是口径回归闸。

    失败时抛 AssertionError (不是返回 False), 调用方不判返回值。
    """
    # ① 同分组清零 (R_GATE_SHORT_CIRCUIT 是最低分, 全错组必然是四条 -2.5)
    for v in (R_GATE_SHORT_CIRCUIT, 1.0, 0.0, -1.0, 2.0):
        assert zero_advantage_masking([v] * 4) == [0.0] * 4, f"同分组 {v} 必须清零"
    # ② 非同分组不得清零
    assert zero_advantage_masking([-2.5, -1.0, -2.0, 1.0]) != [0.0] * 4, "非同分组不得清零"

    # ③-a 口径: 优势 == 减均值原值。若有人把默认改回「除 std」, 这里立刻红。
    vals = [2.0, 1.0, -1.0, -2.0]
    mean = sum(vals) / len(vals)
    expected = [round(v - mean, 6) for v in vals]
    got = zero_advantage_masking(vals)
    assert got == expected, (
        f"优势口径被改动了: 期望减均值原值 {expected}, 实际 {got}。"
        f"若实际值是被 std 归一化过的, 说明 scale_rewards 退回默认 'group' 了")

    # ③-b 反向不变量: 垃圾组(仅差一档)不得压过真信息组 —— 这正是不除 std 的目的
    junk = zero_advantage_masking([-2.5, -2.5, -2.5, -2.0])
    real = zero_advantage_masking(vals)
    assert max(map(abs, junk)) < max(map(abs, real)), \
        "垃圾组(仅差一档)的优势幅度必须小于真有判别信息的组"
    return True



def validate_generation_shape(n_rows: int, num_generations: int,
                              per_device: int, world_size: int = 1) -> Optional[str]:
    """
    TRL 硬约束: per_device × world_size 必须能被 num_generations 整除
    (G 份 rollout 必须整批同进同出)。返回 None=合法，否则返回报错文本。
    """
    if num_generations <= 0:
        return f"G 必须 ≥1，拿到 {num_generations}"
    batch = per_device * max(1, world_size)
    if batch % num_generations:
        return (f"per_device({per_device})×world_size({world_size})={batch} 不能被 "
                f"G={num_generations} 整除 → TRL 会直接抛错。改 per_device 为 "
                f"{num_generations} 的倍数。")
    return None
