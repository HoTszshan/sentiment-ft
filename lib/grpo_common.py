#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 共享层 · 纯 Python (零 torch/trl 依赖，可独立单测)

职责:
  1. 数据契约: {prompt, gold} → GRPOTrainer 吃的**标量列** (dict 列在不同
     datasets 版本里行为不一致，展平成标量最稳)
  2. k 分桶: 同题 G=4 采样 → 数出「判对且溯源过门禁」的轨迹数 k ∈ {0..4}
  3. 训练预算: G / T / lr / β / 有效 batch / 步数的推导与打印 (手册 1.3 口径)
  4. 日志双写 (tee)

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

from reward_fns import compute_reward, zero_advantage_masking   # noqa: E402

# ============================================================
# 常量
# ============================================================

# gold 的 5 个字段（**没有 subject** —— subject 由 rollout 时从 prompt 回显）
GOLD_FIELDS = ("sentiment", "relevance", "hard_case_type", "risk_categories", "evidence")

# 唯一可留进训练池的 k 桶 (数据手册 §四)
KEEP_K = (1, 2, 3)


# ============================================================
# 1. 数据契约: gold → 标量列
# ============================================================

def flatten_gold(gold: Dict[str, Any]) -> Dict[str, Any]:
    """gold dict → 标量列 (risk_categories 是 list，JSON 序列化成字符串)。"""
    if not isinstance(gold, dict):
        raise ValueError(f"gold 必须是 dict，拿到 {type(gold).__name__}")
    out = {}
    for f in GOLD_FIELDS:
        if f not in gold:
            raise ValueError(f"gold 缺字段 {f}")
    out["gold_sentiment"] = str(gold["sentiment"])
    out["gold_relevance"] = str(gold["relevance"])
    out["gold_hard_case_type"] = str(gold["hard_case_type"])
    out["gold_risk_json"] = json.dumps(gold["risk_categories"] or [], ensure_ascii=False)
    out["gold_evidence"] = str(gold["evidence"])
    return out


def rebuild_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    """标量列 → gold dict (打分时还原)。"""
    rk = row.get("gold_risk_json", "[]")
    if isinstance(rk, str):
        rk = json.loads(rk or "[]")
    return {
        "sentiment": row.get("gold_sentiment"),
        "relevance": row.get("gold_relevance"),
        "hard_case_type": row.get("gold_hard_case_type"),
        "risk_categories": list(rk),
        "evidence": row.get("gold_evidence"),
    }


def build_dataset_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    grpo_prompts_*.jsonl 每行 {prompt, gold, [input_text], [meta]} → Trainer 数据行。
    额外带 input_text 列: 打分器要拿它做 evidence ⊂ 原文 断言，
    **不依赖 prompt 列的格式**（TRL 版本间 prompt 传 str 还是 messages 不稳定）。

    原文取值优先级（评估手册缺口 16：训练侧与评测侧原文同源）:
      ① 数据行自带的 `input_text`（build_grpo_probe 默认落盘，与评测侧同一份原文）
      ② 反解 prompt 兜底（旧数据 / --two_field 交付的严格两字段版本）
    """
    from reward_fns import extract_input_text
    rows = []
    for i, rec in enumerate(records):
        prompt = rec.get("prompt")
        if not prompt:
            raise ValueError(f"第 {i} 行缺 prompt")
        input_text = rec.get("input_text")
        if not isinstance(input_text, str) or not input_text.strip():
            input_text = extract_input_text(prompt)
        rows.append({
            "prompt": prompt,
            "input_text": input_text,
            **flatten_gold(rec["gold"]),
        })
    return rows


# ============================================================
# 2. k 分桶
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
# 3. 训练预算 (手册 §1.3 的每个数都要能算出来源)
# ============================================================

@dataclass
class GrpoHyper:
    """GRPO 特有超参。✅=手册标定 / [设计值]=有推导未跑。"""
    num_generations: int = 4                 # G ✅ 手册 1.3
    temperature: float = 0.7                 # T ✅ 手册 1.3
    lr: float = 2e-6                         # [设计值] 1e-6~5e-6 区间中值，比 SFT 2e-5 低一个量级
    beta_kl: float = 0.04                    # β ✅ 手册 1.3 常数 KL
    per_device: int = 4                      # 一步内同时进来的 prompt 数
    grad_accum: int = 2                      # → 8 题/步 × G=4 = 32 条轨迹/步 ✅
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    max_prompt_length: int = 1024            # 含行业风险词表，实测可能到 700~900
    max_completion_length: int = 200         # 见下方 budget_note()
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
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
        }


def budget_note(h: GrpoHyper) -> str:
    """
    为什么 max_completion_length=200 而不是手册 65~105:
      手册 1.3 那一格写的是**CoT 档位**（R_length 只量 CoT，40/65/120 三档），
      不是整段输出预算。整段 = CoT(≤120) + JSON 骨架(~70 固定开销) + 余量。
      若按 105 硬截 → 每条都砍在 JSON 中间 → R_format 全池 -2.5 短路 → 训练废掉。
      CoT 本身超长由 R_length 自然罚到 0，不需要靠截断来治。
    """
    return (f"输出预算 {h.max_completion_length} = CoT 上限 120 + JSON 骨架 ~70 + 余量；"
            f"CoT 超长由 R_length 治理（≥120 归 0），不靠截断")


def assert_zam_equivalence() -> bool:
    """
    Zero-Advantage Masking 与 TRL 组内归一化的自检:
      我们的实装条件是「整组都触顶门禁罚」→ 4 个分数完全相同 → 方差 0。
      TRL 的 (r - mean)/std 对同分组天然返回 0（除以 eps），**语义重合**，
      因此不需要 hack TRL 私有 API 去改 Trainer。
    """
    assert zero_advantage_masking([-2.5] * 4) == [0.0] * 4, "全错组必须清零"
    assert zero_advantage_masking([1.0] * 4) == [0.0] * 4, "全对组方差 0 → 清零"
    for v in (0.0, -1.0, 2.0):
        assert zero_advantage_masking([v] * 4) == [0.0] * 4, f"同分组 {v} 必须清零"
    assert zero_advantage_masking([-2.5, -1.0, -2.0, 1.0]) != [0.0] * 4, "非同分组不得清零"
    # TRL 侧等价: 同分组 mean-center 后全 0
    grp = [-2.5] * 4
    mean = sum(grp) / len(grp)
    assert all(abs(v - mean) < 1e-9 for v in grp)
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


# ============================================================
# 4. 日志双写 (与 SFT 轮 --log_file 同语义)
# ============================================================

class Tee:
    """stdout/stderr 同时写终端与日志文件。"""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def attach_log(path: Optional[str]) -> Optional[Any]:
    """启用日志双写，返回原 stdout（供收尾恢复）。"""
    if not path:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, fh)
    sys.stderr = Tee(old_err, fh)
    return (old_out, old_err)


def detach_log(handle) -> None:
    if not handle:
        return
    sys.stdout, sys.stderr = handle
    try:
        sys.stdout.flush()
    except Exception:
        pass
