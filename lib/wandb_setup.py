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
    # list 形态的"空"也是禁用: [] / [""] (调用方显式传了空列表 = 不要上报)
    if isinstance(user_report_to, list) and not any(v for v in user_report_to):
        _user_off = True
    if no_wandb or _user_off:
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[WARN] 未安装 wandb，本次跳过在线同步（pip install wandb 后重跑可恢复）")
        return False
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode == "offline":
        # 离线模式: 指标落本地 wandb/offline-run-*, 事后 wandb sync 补传。
        # 用途: api.wandb.ai 被 WAF 拦 (403) 时冒烟仍要全链路跑通 ——
        # Trainer 的 wandb callback 在 on_train_begin 才 init, 在线模式炸了
        # 会连累训练一起 FATAL (run_20261031_230711 实测)。
        os.environ.setdefault("WANDB_PROJECT", project)
        os.environ.setdefault("WANDB_NAME", run_name)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print("[WANDB] WANDB_MODE=offline: 指标落本地, 事后 wandb sync 补传")
        return True
    if mode == "online":
        # 在线模式先做 5s 连通性预检, 不通则降级 offline 而不是让训练 FATAL
        import urllib.request, urllib.error
        try:
            resp = urllib.request.urlopen("https://api.wandb.ai", timeout=5)
            print(f"[WANDB] api.wandb.ai 可达 (HTTP {resp.status}), 保持在线同步")
        except urllib.error.HTTPError as exc:
            # WAF 拦截页是 text/html 的 403 (AutoDL 出口实测); API 正常拒绝
            # (鉴权失败等) 是 JSON/纯文本——用 content-type 区分可达与被墙
            ctype = exc.headers.get("content-type", "")
            if exc.code == 403 and "text/html" in ctype:
                print(f"[WANDB] api.wandb.ai 被 WAF 拦截 (403 HTML), "
                      "自动降级 WANDB_MODE=offline, 事后 wandb sync 补传")
                os.environ["WANDB_MODE"] = "offline"
            else:
                print(f"[WANDB] api.wandb.ai 可达 (HTTP {exc.code}), 保持在线同步")
        except Exception as exc:
            print(f"[WANDB] api.wandb.ai 不可达 ({exc.__class__.__name__}), "
                  "自动降级 WANDB_MODE=offline, 事后 wandb sync 补传")
            os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_NAME", run_name)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    return True
