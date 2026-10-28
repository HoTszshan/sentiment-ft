#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
格式自检（format canary）· 与训练脚本解耦

训练过程中定期在验证集上测「模型现在会不会输出规定格式」。不参与梯度，只做观测。

存在的理由：**loss 曲线判断不了格式是否学会**。实测两轮 `train_loss` 只差 4%，
格式合规率却是 100% 与 0% 的差别；没有本模块就只能等训练 + 推理 + 评测跑完才发现。

四个指标：
  p_close   teacher-forced 下 `</think>` 的平均概率（在真实闭合位置）
  p_next    `</think>` 之后首个 token 的平均概率（读**真实 token id**，不写死 '{'）
  p_stop    序列末尾 `<|im_end|>` 的平均概率
  greedy_ok 贪心生成中格式全合规的条数

用法：
  训练内挂载 —— `build_callback(tok, rows, ...)` 返回 TrainerCallback，交给 Trainer
  独立检查   —— `run_canary(model, tok, rows)` 直接返回指标 dict（见 scripts/check_format.py）

协议 token 由 tokenizer 解析，不硬编码 id：换基座只要词表里有这几个 token 就能用。
"""

from typing import Any, Dict, List, Optional, Sequence

import prompt_template as pt

PROTOCOL_OPEN = "<think>"
PROTOCOL_CLOSE = "</think>"
STOP_TOKEN = "<|im_end|>"


# ============================================================
# 协议 token 解析
# ============================================================

def resolve_protocol_ids(tokenizer: Any) -> Dict[str, int]:
    """从 tokenizer 解析协议 token 的 id。

    不写死数字：Qwen 系这两个 token 是 151667 / 151668，换基座会变。
    词表里没有时直接报错 —— 静默退化成 unk 会让 canary 读数恒为 0，
    比直接失败更难排查。
    """
    ids: Dict[str, int] = {}
    for key, token in (("open", PROTOCOL_OPEN),
                       ("close", PROTOCOL_CLOSE),
                       ("stop", STOP_TOKEN)):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(
                f"词表中找不到协议 token {token!r}（解析结果 {tid}）。"
                f"当前基座的输出协议与本模块不匹配。")
        ids[key] = int(tid)
    return ids


# ============================================================
# 纯文本层判定（可脱离 torch 单测）
# ============================================================

def is_format_compliant(text: str) -> bool:
    """生成文本是否满足「<think>…</think> + 可解析 JSON」。

    只做结构判定，不校验 schema 字段（那是 evaluate_sft 的事）。
    """
    import json

    if PROTOCOL_OPEN not in text or PROTOCOL_CLOSE not in text:
        return False
    after = text.split(PROTOCOL_CLOSE)[-1].lstrip()
    if "}" not in after:
        return False
    try:
        pred = json.loads(after[: after.rfind("}") + 1])
    except Exception:
        return False
    return isinstance(pred, dict)


# ============================================================
# 指标计算
# ============================================================

def measure_teacher_forced(model, tokenizer, rows: Sequence[Dict[str, Any]],
                           ids: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """在标注位置读模型给出的概率（teacher-forced，不做生成）。

    取每个样本监督区内真实 `</think>` / `</think>` 后首 token / 末位停笔 token 的概率，
    对多条样本取均值。这些位置是「格式协议的三个关键决策点」。
    """
    import torch

    ids = ids or resolve_protocol_ids(tokenizer)
    device = next(model.parameters()).device
    ps_close, ps_next, ps_stop = [], [], []

    with torch.no_grad():
        for r in rows:
            prompt_text = pt.render_prompt(tokenizer, r["instruction"], r["input"])
            body = pt._strip_think_prefix(r["output"])
            full_text = prompt_text + body + pt.ASSISTANT_SUFFIX
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            plen = len(tokenizer.encode(prompt_text, add_special_tokens=False))

            inp = torch.tensor([full_ids], device=device)
            logits = model(input_ids=inp,
                           attention_mask=torch.ones_like(inp),
                           logits_to_keep=0).logits[0]
            probs = torch.softmax(logits.float(), -1)

            # 位置 t 的 token 由 logits[t-1] 预测
            for t in range(plen, len(full_ids)):
                if full_ids[t] == ids["close"]:
                    ps_close.append(float(probs[t - 1][ids["close"]]))

            if ids["close"] in full_ids[plen:]:
                j = full_ids.index(ids["close"], plen)
                if j + 1 < len(full_ids):
                    # 读真实下一个 token，不写死 ord("{")：
                    # Qwen 会把 '{"' 合成一个 token（id 4913），'{' 单独是 id 90
                    ps_next.append(float(probs[j][full_ids[j + 1]]))

            ps_stop.append(float(probs[len(full_ids) - 2][ids["stop"]]))

    return {
        "p_close": sum(ps_close) / max(len(ps_close), 1),
        "p_next": sum(ps_next) / max(len(ps_next), 1),
        "p_stop": sum(ps_stop) / max(len(ps_stop), 1),
    }


def measure_greedy(model, tokenizer, rows: Sequence[Dict[str, Any]],
                   max_new: int = 400) -> int:
    """贪心生成 rows，返回格式全合规的条数。"""
    import torch

    device = next(model.parameters()).device
    ok = 0
    with torch.no_grad():
        for r in rows:
            prompt_text = pt.render_prompt(tokenizer, r["instruction"], r["input"])
            inp = tokenizer(prompt_text, return_tensors="pt",
                            add_special_tokens=False).to(device)
            gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(gen[0][inp["input_ids"].shape[1]:],
                                    skip_special_tokens=False)
            # 前缀预置时 <think> 在 prompt 里、不在生成段，补回再判
            if pt.USE_THINK_PREFIX:
                text = pt.THINK_PREFIX + text
            if is_format_compliant(text):
                ok += 1
    return ok


def run_canary(model, tokenizer, rows: Sequence[Dict[str, Any]],
               n: int = 3, max_new: int = 400,
               ids: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """跑一次完整自检，返回指标 dict。model 需处于 eval 模式调用方的责任。

    返回: {p_close, p_next, p_stop, greedy_ok, greedy_n, n_samples}
    """
    sample = list(rows)[:n] if n > 0 else list(rows)
    ids = ids or resolve_protocol_ids(tokenizer)
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        out: Dict[str, Any] = dict(measure_teacher_forced(model, tokenizer, sample, ids))
        out["greedy_ok"] = measure_greedy(model, tokenizer, sample, max_new)
        out["greedy_n"] = len(sample)
        out["n_samples"] = len(sample)
    finally:
        if was_training:
            model.train()
    return out


# ============================================================
# Trainer 回调
# ============================================================

def build_callback(torch, tokenizer, rows: Sequence[Dict[str, Any]],
                   every: int = 30, n: int = 3, max_new: int = 400):
    """构造一个挂到 Trainer 上的回调，每 every 个优化步跑一次 run_canary。

    依赖由调用方注入（torch / tokenizer / 验证行），本模块不猜基座路径。
    """
    from transformers import TrainerCallback

    sample = list(rows)[:n] if n > 0 else list(rows)
    ids = resolve_protocol_ids(tokenizer)

    class _FormatCanary(TrainerCallback):
        def __init__(self):
            self.trainer = None

        def on_log(self, args, state, control, logs=None, **kw):
            if every <= 0 or state.global_step == 0 or state.global_step % every != 0:
                return
            try:
                model = self.trainer.model
                inner = getattr(model, "module", model)   # DDP 壳
                m = run_canary(inner, tokenizer, sample, n=0,
                               max_new=max_new, ids=ids)
                print(f"[CANARY] step {state.global_step}: "
                      f"p(</think>)={m['p_close']:.3f}  "
                      f"p(next|after</think>)={m['p_next']:.3f}  "
                      f"p(<|im_end|>)={m['p_stop']:.3f}  "
                      f"greedy_ok={m['greedy_ok']}/{m['greedy_n']}")
                if logs is not None:
                    # 注意：本回调注册在 WandbCallback 之后，改 logs 已来不及
                    # （wandb 那份先发出去了），必须自己补 log 才能进 history。
                    logs.update({
                        "canary/p_close": m["p_close"],
                        "canary/p_next": m["p_next"],
                        "canary/p_stop": m["p_stop"],
                        "canary/greedy_ok": m["greedy_ok"],
                    })
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({f"canary/{k}": v for k, v in m.items()},
                                      step=state.global_step)
                    except Exception:
                        pass  # wandb 未启用时静默跳过
            except Exception as e:
                print(f"[CANARY] 失败(不影响训练): {e}")

    return _FormatCanary()
