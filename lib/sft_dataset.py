#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据集与 Collator · 训练 / 推理 / 统计三方共用的唯一实现

定位：
  这是 `prompt_template.py`（外壳）与 `sft_weight_planner.py`（字段权重）之间的装配层。
  它只做三件事，且每件事都不许有第二份实现：
    1. 一行 JSONL → prompt 外壳 + 金标 output（走 prompt_template，不自己拼字符串）
    2. prompt/output → token id、labels（前缀 -100）、loss_weights（字段级）
    3. 变长样本 → 右填充 batch（labels 补 -100、weights 补 0，防 padding 污染分母）

依赖策略（**import 时零重型依赖**）：
  torch 只在 collate 路径惰性导入；本模块顶层不 import torch/transformers。
  这样无 GPU 机器可以直接 `preview()` / `stats()`，把「套模板之后长什么样」印出来，
  不必为了看一眼数据去装 2GB 的 torch。

返回值是 **list 而非 tensor**：与 tokenizer 的原始输出同构，便于 JSON 导出与逐 token 打印。
转 tensor 的责任在 collate 阶段（那时才真正需要 torch）。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompt_template as pt          # noqa: E402
from sft_weight_planner import (     # noqa: E402
    FIELD_WEIGHTS,
    POC_STAGE,
    USE_FIELD_WEIGHTS,
    USE_SKELETON_MASK,
    char_level_encode,
    plan_token_weights,
)

IGNORE_INDEX = -100


# ============================================================
# 数据读取
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读 JSONL（跳过空行）。空文件返回空列表而非报错 —— 由调用方决定是否视为致命。"""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# Dataset
# ============================================================

class SFTConversationDataset:
    """三元组 JSONL → {input_ids, labels, loss_weights}。

    参数:
      data_path: train_sft_81_aligned.jsonl 形态（instruction / input / output）
      tokenizer: 鸭子类型；有 chat_template 就走 chat 模板，否则退手写外壳
      max_len:   截断长度（与 TrainingArguments.max_seq_length 同口径）
      strict_parity: 初始化时校验 chat 渲染 == 手写基准；不一致直接抛错。
                     默认 True —— 宁可跑不起来，也不静默漂移。
      validate_limit: parity 抽查条数
    """

    def __init__(self, data_path: str, tokenizer, max_len: int = 2560, *,
                 strict_parity: bool = True, validate_limit: int = 32):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.uses_chat = pt.has_chat_template(tokenizer)
        # strict_parity=False 的语义：关闭「初始化时的批量校验」与「逐条渲染校验」，
        # 让明知模板漂移的场景（如 inspect 想看 thinking 版数据长什么样）也能取到样本。
        # 此时渲染退回手写基准 —— 训练序列与 chat 渲染可能不同，责任在调用方。
        self.strict_parity = bool(strict_parity)
        self.parity: Optional[Dict[str, Any]] = None

        self.samples = load_jsonl(data_path)

        if self.strict_parity and self.uses_chat:
            self.parity = pt.assert_parity(
                self.tokenizer, self.samples, limit=validate_limit)

    # ---- 基本协议 ----

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ---- 单样本构造 ----

    def __getitem__(self, idx: int) -> Dict[str, List]:
        row = self.samples[idx]
        item = self.build_item(row)
        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "loss_weights": item["loss_weights"],
        }

    def build_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """核心装配：外壳渲染 → tokenize → 前缀掩码 → 字段权重。

        权重口径与 `plan_token_weights` 严格一致：
          - prompt 段权重 0、label -100（不参与梯度，也不参与分母）
          - output 段按 POC 阶段：POC1 等权 1.0 / POC2 骨架 0 / POC3 字段 4.0/3.0/2.0/0.4
          - 尾部 <|im_end|> 未被 span 覆盖 → default 1.0（必须留梯度：学「停笔」）
        """
        instruction = row.get("instruction", "") or ""
        input_text = row.get("input", "") or ""
        output_text = row.get("output", "") or ""

        if self.strict_parity or not self.uses_chat:
            # 正常路径：chat 渲染并逐条验字节等式（漂移在此抛错）
            prompt_text, full_text = pt.render_full(
                self.tokenizer, instruction, input_text, output_text,
            )
        else:
            # 逃生门路径（--no_strict_parity）：明知模板漂移也要取样本。
            # 统一走手写基准 —— 序列形态与训练基准一致，可看可导出，只是没有 chat 校验。
            prompt_text = pt.render_prompt(None, instruction, input_text)
            full_text = prompt_text + pt._strip_think_prefix(output_text) + pt.ASSISTANT_SUFFIX

        # 预置开启时 output 的 "<think>\n" 已在 prompt 掩码区，权重 span 必须
        # 基于同一份 output_body，否则 span 与 token 序列错位。
        output_body = pt._strip_think_prefix(output_text)

        encode = self._encode if self.tokenizer is not None else char_level_encode

        full_ids = encode(full_text)
        truncated = False
        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]
            truncated = True

        prompt_ids = encode(prompt_text)
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        plan = plan_token_weights(
            prompt_text=prompt_text,
            output_text=output_body,
            encode=encode,
            max_len=self.max_len,
        )
        weights = list(plan["weights"])
        # 尾部（<|im_end|>）不在 span 内 → default 权重
        if len(weights) < len(full_ids):
            weights.extend([FIELD_WEIGHTS["default"]] * (len(full_ids) - len(weights)))
        weights = weights[: len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "loss_weights": weights,
            "prompt_text": prompt_text,
            "full_text": full_text,
            "prompt_len": prompt_len,
            "truncated": truncated,
            "spans": plan["spans"],
        }

    # ---- 统计 ----

    def stats(self, limit: int = 0) -> Dict[str, Any]:
        """长度分布（prompt / output / total），单位 = token（或字符，无 tokenizer 时）。"""
        rows = self.samples[:limit] if limit else self.samples
        if not rows:
            return {"n": 0}

        unit = "token" if self.tokenizer is not None else "char"
        encode = self._encode if self.tokenizer is not None else char_level_encode

        prompts, outputs, totals = [], [], []
        for row in rows:
            if self.strict_parity or not self.uses_chat:
                prompt_text, full_text = pt.render_full(
                    self.tokenizer,
                    row.get("instruction", "") or "",
                    row.get("input", "") or "",
                    row.get("output", "") or "",
                )
            else:
                prompt_text = pt.render_prompt(
                    None, row.get("instruction", "") or "",
                    row.get("input", "") or "")
                full_text = prompt_text + (row.get("output", "") or "") + pt.ASSISTANT_SUFFIX
            p, t = len(encode(prompt_text)), len(encode(full_text))
            prompts.append(p)
            totals.append(t)
            outputs.append(t - p)

        def pct(vals: List[int], q: float) -> int:
            s = sorted(vals)
            return s[min(int(len(s) * q / 100), len(s) - 1)]

        return {
            "n": len(rows),
            "unit": unit,
            "prompt": {"p50": pct(prompts, 50), "p95": pct(prompts, 95), "max": max(prompts)},
            "output": {"p50": pct(outputs, 50), "p95": pct(outputs, 95), "max": max(outputs)},
            "total": {"p50": pct(totals, 50), "p95": pct(totals, 95), "max": max(totals)},
            "over_max_len": sum(1 for t in totals if t > self.max_len),
        }

    # ---- 预览（无 torch 可用） ----

    def preview(self, idx: int, *, show_tokens: bool = False,
                show_ignored: bool = False, show_weights: bool = True,
                color: bool = False, max_tokens: int = 400) -> str:
        """把「套进模板之后到底长什么样」逐层印出来（return 字符串，便于测试断言）。"""
        item = self.build_item(self.samples[idx])
        out: List[str] = []
        bar = "=" * 78

        out.append(bar)
        out.append(f"样本 #{idx} / {len(self.samples)}")
        out.append(f"  seq_len={len(item['input_ids'])}  prompt_len={item['prompt_len']}  "
                   f"loss_tokens={sum(1 for l in item['labels'] if l != IGNORE_INDEX)}  "
                   f"truncated={item['truncated']}")
        out.append(f"  POC={POC_STAGE}  骨架掩码={'开' if USE_SKELETON_MASK else '关'}  "
                   f"字段加权={'开' if USE_FIELD_WEIGHTS else '关'}  "
                   f"模板={'chat' if self.uses_chat else '手写'}")

        # ① 原始三元组
        row = self.samples[idx]
        out.append("\n--- ① 原始字段 ---")
        for k in ("instruction", "input", "output"):
            v = row.get(k, "") or ""
            head = v[:200] + ("…" if len(v) > 200 else "")
            out.append(f"[{k}]  ({len(v)} 字)\n{head}")

        # ② messages
        out.append("\n--- ② messages ---")
        row_instruction = row.get("instruction", "") or ""
        row_input = row.get("input", "") or ""
        row_output = row.get("output", "") or ""
        for role, content in (("system", row_instruction),
                              ("user", row_input),
                              ("assistant", row_output)):
            head = content[:160] + ("…" if len(content) > 160 else "")
            out.append(f"  {{{role}}}: {head}")

        # ③ 渲染后的完整文本（模板字符可见）
        out.append("\n--- ③ 模板渲染后完整文本 (add_generation_prompt / <|im_end|> 均可见) ---")
        out.append(item["full_text"])

        # ④ token 级视图
        ids = item["input_ids"][:max_tokens]
        labels = item["labels"][:max_tokens]
        weights = item["loss_weights"][:max_tokens]
        loss_mask = [l != IGNORE_INDEX for l in labels]

        out.append(f"\n--- ④ token 视图 (前 {len(ids)} 个) ---")
        if show_tokens and self.tokenizer is not None:
            toks = self.tokenizer.convert_ids_to_tokens(ids)
            cells = []
            for t, m in zip(toks, loss_mask):
                t = (t or "").replace("\n", "\\n")
                cells.append(_green(t) if (m and color) else (f"[{t}]" if m else t))
            out.append(" ".join(cells))
        else:
            out.append("ids   : " + " ".join(str(i) for i in ids))
            out.append("loss  : " + " ".join("1" if m else "." for m in loss_mask))
        if show_weights:
            out.append("weight: " + " ".join(_fmt_w(w) for w in weights))

        # 字段 span 表
        if item["spans"]:
            out.append("\n  字段 span (token 区间 / 权重 / 覆盖文本):")
            for t_s, t_e, key, w in item["spans"]:
                out.append(f"    {key:<16} [{t_s:>5},{t_e:>5})  {w:<5} "
                           f"{_decode_span(self.tokenizer, item['input_ids'][t_s:t_e])}")

        if show_ignored:
            out.append("\n  不参与 loss 的区段（前缀掩码 + 骨架）:")
            out.append(f"    [0,{item['prompt_len']}) → prompt（label=-100, w=0）")
            # 只统计「输出侧」的 w=0（骨架掩码），prompt 段的 w=0 不算 ——
            # 否则 prompt ~1800 个 0 会把骨架的真实计数（~30）淹没 60 倍
            sk = sum(1 for w in item["loss_weights"][item["prompt_len"]:] if w == 0.0)
            if sk:
                out.append(f"    输出侧 w=0 的 token 共 {sk} 个（骨架掩码）")
            else:
                out.append("    输出侧无 w=0 token（POC1 未启用骨架掩码，属预期）")

        out.append(bar)
        return "\n".join(out)


# ============================================================
# Collator
# ============================================================

class WeightedDataCollator:
    """右填充 collator：labels 补 -100、weights 补 0.0、attention_mask 补 0。

    weights 补 0 是关键：FIELD_W 的分母是「加权有效 token 之和」，
    padding 位置若按 default=1.0 计入分母，每个 batch 的分母都会随 padding 量浮动，
    loss 曲线会出现与学习进度无关的抖动。
    """

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:  # pragma: no cover - 只在误用时触发
            raise ImportError("collate 需要 torch；仅预览/统计场景请直接用 Dataset.preview()")

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        out: Dict[str, List[Any]] = {
            "input_ids": [], "labels": [], "loss_weights": [], "attention_mask": []
        }

        for f in features:
            ids, lbl, w = f["input_ids"], f["labels"], f["loss_weights"]
            pad_len = max_len - len(ids)
            out["input_ids"].append(
                torch.cat([_t(ids), torch.full((pad_len,), pad_id, dtype=torch.long)]))
            out["labels"].append(
                torch.cat([_t(lbl), torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
            out["loss_weights"].append(
                torch.cat([_t(w, torch.float), torch.zeros(pad_len, dtype=torch.float)]))
            out["attention_mask"].append(
                torch.cat([torch.ones(len(ids), dtype=torch.long),
                           torch.zeros(pad_len, dtype=torch.long)]))

        return {k: torch.stack(v) for k, v in out.items()}


def _t(seq: Sequence, dtype=None):
    import torch
    return torch.tensor(list(seq), dtype=dtype or torch.long)


# ============================================================
# 预览辅助
# ============================================================

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _fmt_w(w: float) -> str:
    if w == 0.0:
        return "0"
    if w == 1.0:
        return "1"
    return f"{w:g}"


def _decode_span(tokenizer, ids: Sequence[int], width: int = 40) -> str:
    if not ids:
        return ""
    if tokenizer is None:
        text = "".join(chr(int(i)) for i in ids if int(i) < 0x110000)
    else:
        text = tokenizer.decode(list(ids), skip_special_tokens=False)
    text = text.replace("\n", "\\n")
    return text[:width] + ("…" if len(text) > width else "")


def preview_file(data_path: str, tokenizer=None, *, start: int = 0, n: int = 1,
                 max_len: int = 2560, strict_parity: bool = True,
                 show_tokens: bool = False, show_ignored: bool = False,
                 color: bool = False, max_tokens: int = 400) -> str:
    """一次性预览多条样本（供 CLI 与单测共用）。"""
    ds = SFTConversationDataset(
        data_path, tokenizer, max_len=max_len, strict_parity=strict_parity,
    )
    blocks = []
    end = min(len(ds), start + max(n, 0)) if n else len(ds)
    for i in range(start, end):
        blocks.append(ds.preview(
            i, show_tokens=show_tokens, show_ignored=show_ignored,
            color=color, max_tokens=max_tokens))
    return "\n".join(blocks)
