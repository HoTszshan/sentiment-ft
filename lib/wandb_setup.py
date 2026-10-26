#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wandb 接入统一接口 (零 wandb 依赖——本模块只在探测时 import 它)

职责: 「能不能开 wandb」判定 + env 注入三件套，SFT/GRPO 两个训练脚本共用。
report_to 的最终形态不在这里落: SFT 用 list (TrainingArguments.report_to=["wandb"])，
GRPO 用 str (GRPOConfig.report_to="wandb")，由调用方按各自 Config 类型赋值。

真理源:
  train_sft_lora.py 原 §3 wandb 接入段 (手动 wandb.init 会与 Trainer callback
  lifecycle 打架 → 一律靠 env + report_to 让 callback 自己 init/finish)
"""

import os


def setup_wandb(project: str, run_name: str, no_wandb: bool = False,
                user_report_to=None) -> bool:
    """
    判定 wandb 是否启用，ready 时注入 env 三件套。返回 ready: bool。

      - no_wandb=True 或 user_report_to=="none" (用户显式禁用) → False，不动 env
      - wandb 未安装 → WARN + False (降级本地日志，不致命)
      - ready → setdefault 注入 WANDB_PROJECT / WANDB_NAME / WANDB_LOG_MODEL=false
        (探针口径: 只传指标不传权重; setdefault 不覆盖用户显式 env)

    调用方拿到 False 后自行把 report_to 落成禁用形态。
    """
    # 用户显式给的空形态一律视为禁用 ("none" / "" / [] / [""])；
    # None = 调用方未传 report_to（如 SFT 没扫到 --report_to），不算禁用
    _rt_forms = (user_report_to, ) if not isinstance(user_report_to, list) else tuple(user_report_to)
    _user_off = any(v in ("none", "") for v in _rt_forms)
    if no_wandb or _user_off:
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[WARN] 未安装 wandb，本次跳过在线同步（pip install wandb 后重跑可恢复）")
        return False
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_NAME", run_name)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    return True
