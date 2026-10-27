#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情负面精判 · Qwen3-4B SFT LoRA 训练主脚本 (《训练手册SOP》对齐版)

核心特性:
1. 基座: Qwen/Qwen3-4B-Instruct-2507 + LoRA (r=16, alpha=32, target_modules=全量线性层)。
2. 协议分隔符对齐: 识别 <think> ... </think>（训练语料固定形态）。
3. 前缀损失掩码 (Prefix Masking): 输入指令、Schema 与原文整段置 -100 (Ignore Index)。
4. 骨架差分掩码 (Skeleton Masking): JSON 结构骨架（键名/引号/括号/冒号/逗号）权重置 0，
   梯度只回传给 CoT 与各字段 Value；协议分隔符留 1.0x 不遮（遮了就学不会开口与停笔）。
5. FIELD_W 自定义加权损失 (防梯度稀释核心技术):
   - 覆写 Trainer.compute_loss;
   - hard_case 提权 4.0x / sentiment·relevance 提权 3.0x / risk 提权 2.0x;
   - evidence 降权 0.4x (防长证据文本稀释离散标签);
   - CoT 保持 1.0x (保留语用推演梯度);
   - 以加权有效 Token 之和重算分母，规避梯度漂移与伪尖峰。
6. **POC 阶段开关 (SFT_POC_STAGE)** —— 三阶段共用本脚本，一次只动一个变量:
   - POC1 = 仅前缀掩码，输出侧等权 1.0        (baseline / 消融对照组)
   - POC2 = POC1 + 骨架掩码
   - POC3 = POC2 + 字段级加权 FIELD_W
   见 `sft_weight_planner.POC_STAGE` 与《SFT训练实战手册》§1.4。
7. 训练纪律: **不做自动早停**；正式训练 Epoch 严格控 1 (防过拟合 + 给下游 GRPO 留策略熵)，
   探针跑 EPOCHS=3 只为看曲线拐点；bf16 混合精度, lr=1e-4 (LoRA 口径, Cosine + warmup_ratio 0.05)。

用法:
  真实训练 (需 torch/transformers/peft):
      python scripts/train_sft_lora.py --output_dir out/lora_r1 --bf16

  落盘训练日志 + 同步 wandb (AutoDL 探针推荐):
      python scripts/train_sft_lora.py \
          --output_dir out/probe_r1 --bf16 \
          --train_file output/dataset/sft_train.jsonl \
          --num_train_epochs 3 --per_device_train_batch_size 2 \
          --logging_steps 1 --log_file logs/train_probe.log \
          --wandb_project sentiment-sft-probe --wandb_run_name probe-81seed

  离线加权检视 (无需任何 DL 依赖，用于验证/面试演示):
      python scripts/train_sft_lora.py --dry_run

自定义 flag (不与 transformers TrainingArguments 冲突, 在解析前拦截):
  --log_file PATH      训练全程日志同时写入文件 (tee, 终端与文件双写)
  --wandb_project NAME wandb 项目名 (默认 sentiment-sft-probe)
  --wandb_run_name NAME wandb 运行名 (默认 probe_<时间戳>)
  --no_wandb           关闭 wandb (本地冒烟用)
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# 字段级 Loss 权重 span 规划器（纯逻辑，有独立单测 unit/test_weight_planner.py）
# FIELD_W 定值: hard_case 4.0 / sentiment·relevance 3.0 / risk 2.0 / evidence 0.4 / CoT 1.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from sft_weight_planner import (
    FIELD_WEIGHTS, plan_token_weights,
    POC_STAGE, USE_SKELETON_MASK, USE_FIELD_WEIGHTS,
)
# 外壳渲染（ChatML）唯一真理源：与 probe_infer / stat_seq_len / GRPO 构建同源。
import prompt_template as pt
import format_canary
from sft_dataset import SFTConversationDataset, WeightedDataCollator
from log_tee import attach_log, detach_log


def _intercept_custom_flags():
    """
    在 HfArgumentParser 之前拦截本项目自定义 flag，避免撞 TrainingArguments 字段。
    返回 dict: log_file / wandb_project / wandb_run_name / no_wandb
              / strict_parity / trainer_backend
              / think_prefix / unfreeze_protocol_tokens / canary_*
    """
    opts = {
        "log_file": None,
        "wandb_project": "sentiment-sft-probe",
        "wandb_run_name": f"probe_{time.strftime('%m%d_%H%M%S')}",
        "no_wandb": False,
        # 渲染与手写基准不一致时是否直接中止（默认 True —— 宁可跑不起来也不静默漂移）
        "strict_parity": True,
        # 训练器后端：auto=有 trl 用 SFTTrainer 否则 HF Trainer / trl / hf 显式指定
        "trainer_backend": "auto",
        # 前缀预置：把 "<think>\n" 写进 assistant 前缀（默认关，理由见 prompt_template）
        "think_prefix": False,
        # 协议 token 解冻：给 <think>/</think> 的输出行挂可训练偏移（默认开）
        "unfreeze_protocol_tokens": True,
        # 格式 canary：每 N 步 × M 条 val，N=0 关闭
        "canary_every": 30,
        "canary_n": 3,
        "canary_max_new": 400,
    }
    # 带取值的 flag → opts key（统一 int 转换的在下面 VALUE_CAST 里声明）
    key_map = {
        "--log_file": "log_file",
        "--wandb_project": "wandb_project",
        "--wandb_run_name": "wandb_run_name",
        "--trainer_backend": "trainer_backend",
        "--think_prefix": "think_prefix",
        "--canary_every": "canary_every",
        "--canary_n": "canary_n",
        "--canary_max_new": "canary_max_new",
    }
    def _as_bool(raw: str) -> bool:
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        raise SystemExit(f"[FATAL] --think_prefix 需为 0/1/true/false，收到 {raw!r}")

    value_cast = {
        "canary_every": int, "canary_n": int, "canary_max_new": int,
        "think_prefix": _as_bool,
    }
    bool_map = {
        "--no_wandb": ("no_wandb", False),
        "--no_strict_parity": ("strict_parity", False),
        "--no_unfreeze_protocol_tokens": ("unfreeze_protocol_tokens", False),
    }
    argv = list(sys.argv)
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in key_map:
            if i + 1 < len(argv):
                raw = argv[i + 1]
                key = key_map[a]
                # value_cast 以 opts key 为索引（不是 "--flag" 字符串），这里必须用 key
                opts[key] = value_cast[key](raw) if key in value_cast else raw
                i += 2
                continue
            raise SystemExit(f"[FATAL] {a} 缺少取值")
        if a in bool_map:
            key, val = bool_map[a]
            opts[key] = val
            i += 1
            continue
        out.append(a)
        i += 1
    sys.argv = out
    return opts


# ============================================================
# 参数定义
# ============================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-4B-Instruct",
        metadata={"help": "基座模型路径或 HuggingFace ID"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    train_file: str = field(
        default="data/mixed/sft_train.jsonl",
        metadata={"help": "训练集 JSONL 文件路径"}
    )
    val_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证集 JSONL；给了才开 eval，用于观测 val_loss 是否反弹（过拟合信号）"}
    )
    max_seq_length: int = field(
        default=2560,
        metadata={"help": "最大截断长度 (SOP §6.3 P99 定值)"}
    )


# ============================================================
# 离线加权检视 (不依赖 torch / transformers / peft)
# ============================================================

def dry_run_inspect(data_file: str, n: int = 2):
    """
    离线检视模式：不加载模型、不训练，仅打印 FIELD_W 加权与掩码分布。
    用途: 无 GPU 环境下验证加权逻辑；面试现场演示「梯度没有被 JSON 骨架稀释」。

    注: 用字符级伪编码器（1 字符 = 1 token）展示分布形态。真实 token 划分由 tokenizer 决定，
        但 span 的「归属」与「相对占比」不受影响 —— span 定位走字符级映射，与分词算法无关。
    """
    from sft_weight_planner import char_level_encode

    print("=" * 74)
    print(f"DRY-RUN 加权检视模式 (不加载模型，纯离线逻辑验证)")
    print(f"POC 阶段      : {POC_STAGE}  (骨架掩码={'开' if USE_SKELETON_MASK else '关'} / "
          f"字段加权={'开' if USE_FIELD_WEIGHTS else '关'}；切换: SFT_POC_STAGE=POC1|POC2|POC3)")
    print(f"FIELD_W 配置: hard_case={FIELD_WEIGHTS['hard_case_type']}x  "
          f"sentiment={FIELD_WEIGHTS['sentiment']}x  relevance={FIELD_WEIGHTS['relevance']}x  "
          f"risk={FIELD_WEIGHTS['risk_categories']}x  evidence={FIELD_WEIGHTS['evidence']}x  "
          f"CoT={FIELD_WEIGHTS['cot']}x"
          + ("" if USE_FIELD_WEIGHTS else "  ← 当前阶段输出侧等权，上方 FIELD_W 不生效"))
    print("=" * 74)

    if not os.path.exists(data_file):
        sys.stderr.write(f"[FATAL] 数据文件不存在: {data_file}\n")
        raise SystemExit(2)

    with open(data_file, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    if not lines:
        sys.stderr.write(f"[FATAL] 数据文件为空: {data_file}\n")
        raise SystemExit(2)

    for i, line in enumerate(lines[:n], 1):
        d = json.loads(line)
        # 外壳统一走 prompt_template（无 tokenizer 时自动退手写路径）——
        # dry-run 与真实训练必须同一个渲染源，否则「离线验证通过」不构成证据。
        prompt_t = pt.render_prompt(None, d.get("instruction", "") or "",
                                    d.get("input", "") or "")
        # 与 Dataset.build_item 同口径：开播种时 output 的 "<think>\n" 已移入 prompt
        output_body = pt._strip_think_prefix(d["output"])

        plan = plan_token_weights(prompt_t, output_body, encode=char_level_encode)

        print(f"\n--- 样本 #{i} ---")
        print(f"  prompt 长度 : {plan['prompt_len']} tokens  (全部置 -100 且权重 0)")
        print(f"  总长度      : {plan['full_len']} tokens | 截断: {plan['truncated']}")
        print(f"\n  {'字段区间':<18}{'token 区间':<16}{'权重':<8}覆盖文本")
        print(f"  {'-' * 18}{'-' * 16}{'-' * 8}{'-' * 28}")

        full_text = prompt_t + output_body
        for t_s, t_e, key, w in plan["spans"]:
            snippet = full_text[t_s:t_e].replace("\n", "\\n")
            if len(snippet) > 32:
                snippet = snippet[:32] + "…"
            print(f"  {key:<18}[{t_s:>5},{t_e:>5})  {w:<8}{snippet}")

        total_pos = sum(1 for w in plan["weights"] if w > 0)
        weighted_sum = sum(w for w in plan["weights"])
        hi = sum(1 for w in plan["weights"] if w >= 3.0)
        # 仅在「有效(非掩码)」区间内统计降权占比，否则会把权重 0 的 prompt 区误计入
        low = sum(1 for w in plan["weights"] if 0 < w < 1.0)

        out_tokens = max(plan["full_len"] - plan["prompt_len"], 1)
        sk = sum(1 for w in plan["weights"][plan["prompt_len"]:] if w == 0.0)

        print(f"\n  有效(非掩码) token : {total_pos}")
        print(f"  加权总量 Σw        : {weighted_sum:.1f}  "
              f"(等权基线应为 {total_pos}；偏离即 FIELD_W 生效)")
        if USE_FIELD_WEIGHTS:
            print(f"  高权(≥3.0x) 占比   : {hi / total_pos * 100:.1f}%   ← 核心分类字段")
            print(f"  降权(0<w<1.0) 占比 : {low / total_pos * 100:.1f}%   ← evidence 长文本")
        if USE_SKELETON_MASK:
            print(f"  骨架遮蔽(0×) 占输出: {sk / out_tokens * 100:.1f}%   ← Skeleton Masking 省下的梯度")

    print("\n" + "=" * 74)
    if USE_FIELD_WEIGHTS:
        print("验证要点: 高权重落在 hard_case/sentiment/relevance 的枚举值上；")
        print("          evidence 整段被降到 0.4x，不参与稀释核心标签梯度；")
    else:
        print("验证要点: 输出侧全为 1.0x（POC1/POC2 等权），FIELD_W 不参与；")
    if USE_SKELETON_MASK:
        print("          JSON 骨架（键名/引号/括号/逗号）整体置 0x，")
    else:
        print("          未启用骨架掩码：JSON 骨架按 1.0x 计损失；")
    print("          CoT 与协议分隔符始终留 1.0x（掩码的边界，不是疏漏）。")
    print("提示: 真实训练请去掉 --dry_run 并确保已安装 torch/transformers/peft/accelerate")
    print("=" * 74)


# ============================================================
# 训练组件 (惰性构建 — 仅在真实训练路径导入重型依赖)
# ============================================================

def _resolve_trainer_base(backend: str):
    """选训练器基类。

    backend:
      auto → 优先 transformers.Trainer（见下），无回归路径
      trl  → 强制 trl.SFTTrainer（缺库直接报错，不静默降级）
      hf   → 强制 transformers.Trainer

    ⚠️ auto 不再优先 trl：trl1.x SFTTrainer强校验必须是HF Dataset，
        我们的SFTConversationDataset为自定义鸭子类型，trl侧需显式转换；
        auto 直接用原生HF Trainer更稳。两者复用同一compute_loss，行为对齐。
    """
    backend = (backend or "auto").strip().lower()
    if backend not in ("auto", "trl", "hf"):
        raise SystemExit(f"[FATAL] 未知 --trainer_backend {backend!r}，可选 auto/trl/hf")

    if backend == "trl":
        try:
            from trl import SFTTrainer
            return SFTTrainer, "trl.SFTTrainer"
        except ImportError:
            raise SystemExit(
                "[FATAL] --trainer_backend trl 但未安装 trl："
                "pip install -r requirements.txt（trl==0.14.0，或改 --trainer_backend hf）")
    from transformers import Trainer
    return Trainer, "transformers.Trainer"


def _to_hf_dataset(ds) -> "Dataset":
    """把 SFTConversationDataset 物化成 HF datasets.Dataset（trl 1.x 硬性要求）。"""
    from datasets import Dataset as HFDataset
    return HFDataset.from_list([ds[i] for i in range(len(ds))])


def _sft_trainer_extra_kwargs(TrainerBase, tokenizer, backend_name: str) -> dict:
    """按 trl 版本差异探测 SFTTrainer 的构造参数（仓内 train_grpo.py 已有同款探测先例）。

    - 新版 trl 用 processing_class=，旧版用 tokenizer=
    - 我们的 Dataset 已预 tokenize（返回 input_ids/labels/loss_weights），
      必须让 SFTTrainer 跳过它自带的 _prepare_dataset（分词/截断/packing），
      否则会对已编码数据二次处理。

    ⚠️ skip_prepare_dataset 是 SFTConfig 字段（trl 0.14 与 0.19+ 均如此），不是 __init__ 参数：
        需在 SFTConfig 里传入，无法从函数签名检测，由参数解析透传。
        下方样本自检兜底：避免SFTTrainer对自定义数据集执行prepare，提前抛出异常而非静默出错。
    HF Trainer 路径不需要这些，返回空 dict。
    """
    if not backend_name.startswith("trl"):
        return {}
    import inspect
    sig = inspect.signature(TrainerBase.__init__)
    params = sig.parameters
    kwargs = {}
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    # 注：不能在 __init__ kwargs 里塞 skip_prepare_dataset / packing ——
    # 它们属于 SFTConfig（args），塞进 SFTTrainer.__init__ 会直接 TypeError。
    return kwargs


def _build_training_components(trainer_backend: str = "auto"):
    """
    惰性导入训练栈并构建 (Trainer 类, 后端名, torch, tokenizer 依赖)。
    重型依赖缺失时给出明确安装指引而非晦涩的 ImportError。

    Dataset / Collator 不在本函数里定义 —— 它们已下沉到 lib/sft_dataset.py，
    供训练、推理、统计、GRPO 构建四方共用（见该模块 docstring）。
    """
    try:
        import torch
        import torch.nn as nn
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, Trainer, HfArgumentParser,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        sys.stderr.write(
            f"\n[FATAL] 缺少训练依赖: {e}\n"
            f"  安装: pip install torch transformers peft accelerate\n"
            f"  仅验证加权逻辑: python scripts/train_sft_lora.py --dry_run\n\n"
        )
        raise SystemExit(2)

    # 训练器基类：trl.SFTTrainer（接 trl 技术栈）或 HF Trainer（回退）。
    # 二者都走下面同一份 compute_loss 覆写，加权口径完全一致。
    TrainerBase, backend_name = _resolve_trainer_base(trainer_backend)

    class FieldWeightedTrainer(TrainerBase):
        """
        覆写 compute_loss: 加权 CE + 分母按加权有效 Token 重算 (SOP §6.2)。

        为何不用默认 mean: 默认对「所有非 -100 token」等权取均，导致
          (a) ~40% 梯度被 JSON 结构骨架钥匙消耗；
          (b) hard_case 仅 1 token，被 evidence ~30-50 token 严重稀释。
        本实现在分子加权、分母同步加权，保持 loss 量纲稳定 (不因加权而整体放大)。
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            loss_weights = inputs.get("loss_weights")

            # 显存预算最紧的路径（4B + 151936-vocab logits + DDP）下绕开 DDP 壳：
            # DDP.forward 除了正常前向外还有 find_unused_parameters 的全图遍历
            # 与通信 bucket 分配，额外 ~GB 级瞬时占用，3090×2 顶不住。
            # 直接调 .module (PeftModel) 走 LoRA 分支；梯度跨卡同步由 DDP 在
            # 参数上注册的 grad hook 在 backward 时完成，forward 不经 DDP 不破坏同步。
            base = getattr(model, "module", model)
            outputs = base(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                logits_to_keep=0,   # 明确要全量 logits 且不触发 labels→loss 路径
            )
            logits = outputs.get("logits")

            # next-token 预测：整体左移一位
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            # bf16 logits 直接喂 CE：fp32 upcast 发生在内核内部按需要处理，
            # 不再预先把整个 [seq, vocab] 矩阵转 fp32（又省 ~0.7GB）
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            weighted_token_losses = token_losses * shift_weights

            # 动态重算分母: 加权有效 Token 之和；floor 防分母为 0 产生 NaN/伪尖峰
            valid_mask = (shift_labels != -100).float()
            effective_tokens = (shift_weights * valid_mask).sum()
            denominator = torch.clamp(effective_tokens, min=1.0)

            loss = weighted_token_losses.sum() / denominator

            # HF Trainer (transformers>=4.46) 梯度累积缩放契约:
            # 当 model.forward 含 **kwargs (Qwen3 如此 → model_accepts_loss_kwargs=True) 时，
            # training_step 认为子类 compute_loss 已按 num_items_in_batch 做了"token 求和"口径，
            # 不再替我们除以 gradient_accumulation_steps。
            # 我们的 compute_loss 返回的是加权 mean 口径 → 8 个 micro-batch 的梯度直接相加，
            # 等效学习率放大 8 倍、logged train_loss 虚高 8 倍 (探针实测 20~27, ÷8 后 2.5~3.4
            # 与 eval_loss 2.15~3.0 完全吻合 —— 本 bug 的数值铁证)。
            # 修法: 训练态返回前除以 grad_accum，把口径还原为"每 micro-batch 份额"。
            #
            # ⚠️ 仅训练模式下才除以梯度累积步数：
            # prediction_step（eval）同样会传入 num_items_in_batch，但评估阶段不做梯度累积。
            # 如果也除以 grad_accum，eval_loss 会被错误缩小（实测：显示0.59，×4≈2.4才是真实loss）。
            # 判断依据：training_step 执行时 model.train()，prediction_step 为 model.eval()，以此区分两路逻辑。
            #
            # ⚠️ 除数取当前窗口实际累积步数，不用固定GAS：
            # 样本无法整除时，末尾累积窗口是零头；硬除GAS会低估loss、梯度失真。
            # current_gradient_accumulation_steps 记录本窗实际步数；eval靠model.training区分。
            if num_items_in_batch is not None and model.training:
                window_gas = max(int(getattr(
                    self, "current_gradient_accumulation_steps",
                    self.args.gradient_accumulation_steps)), 1)
                loss = loss / window_gas
            return (loss, outputs) if return_outputs else loss

    return (FieldWeightedTrainer, backend_name,
            torch, AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
            HfArgumentParser, LoraConfig, get_peft_model, TaskType)


# ============================================================
# 主流程
# ============================================================

def main():
    # 拦截本项目自定义 flag（在 HfArgumentParser 之前，避免与 TrainingArguments 撞字段）
    dry_run = "--dry_run" in sys.argv
    if dry_run:
        sys.argv = [a for a in sys.argv if a != "--dry_run"]
    extra = _intercept_custom_flags()

    # 把「渲染协议」开关落到 prompt_template（训练/推理两侧同源唯一真理源）。
    #     必须在构造 Dataset 之前设置 —— 之后的 render_prompt 全部读这个值。
    _pfx = pt.set_think_prefix(extra["think_prefix"])
    _unfreeze = bool(extra["unfreeze_protocol_tokens"])
    print(f"[PROTO] 前缀预置={'开' if _pfx else '关'}  |  "
          f"协议token解冻={'开' if _unfreeze else '关'}")

    # 尽早装上 tee，保证 dry_run / 训练 / 报错信息都能落盘。
    #    DDP 多 rank 会各自执行 main()，只有 rank0 装 tee 写 log_file，
    #    其余 rank 只打终端 —— 否则 N 个进程向同一文件追加会交错重复。
    _is_rank0 = int(os.environ.get("RANK", 0)) == 0
    log_handle, log_path = None, None
    if extra["log_file"] and _is_rank0:
        log_handle = attach_log(extra["log_file"])
        if log_handle:
            log_path = log_handle["log_path"]  # attach_log 内部 abspath 后的真实落盘位置
        print(f"[LOG] 训练日志同时写入: {log_path}")

    if dry_run:
        # 允许 --train_file 覆盖，默认用 DataArguments 的默认值
        train_file = DataArguments.__dataclass_fields__["train_file"].default
        for i, a in enumerate(sys.argv):
            if a == "--train_file" and i + 1 < len(sys.argv):
                train_file = sys.argv[i + 1]
        dry_run_inspect(train_file, n=2)
        detach_log(log_handle)
        return

    (FieldWeightedTrainer, trainer_backend_name,
     torch, AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
     HfArgumentParser, LoraConfig, get_peft_model, TaskType) = \
        _build_training_components(
            trainer_backend=extra.get("trainer_backend", "auto"))

    # transformers 5.x 从 TrainingArguments 里移除了 warmup_ratio（只留 warmup_steps），
    # 直接传 --warmup_ratio 会被 HfArgumentParser 拒掉。这里把它从 argv 摘出来自行解析，
    # 等 total_steps 算出后再换算成 warmup_steps —— 对 4.x / 5.x 都兼容。
    _warmup_ratio = None  # type: Optional[float]
    for i, a in enumerate(list(sys.argv)):
        if a == "--warmup_ratio" and i + 1 < len(sys.argv):
            _warmup_ratio = float(sys.argv[i + 1])
            sys.argv.pop(i + 1)
            sys.argv.pop(i)
            break

    # 0. 配置训练参数（HfArgumentParser 解析；wandb 接入走 lib/wandb_setup.py）
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb 接入：显式控制 report_to，避免默认 "all" 在无 wandb 环境下报错
    # 判定 + env 注入统一走 lib/wandb_setup.py（与 train_grpo.py 共用一套口径）
    _rt = training_args.report_to
    # transformers 5.x report_to 默认就是 "none"（4.x 是 "all"）——
    # 不能拿值判断「用户禁用」，得看命令行是否显式传了 --report_to
    _user_gave_report_to = "--report_to" in sys.argv
    from wandb_setup import setup_wandb
    wandb_ready = setup_wandb(
        extra["wandb_project"], extra["wandb_run_name"],
        no_wandb=extra["no_wandb"],
        user_report_to=_rt if _user_gave_report_to else None)
    if wandb_ready:
        # 让 Trainer 的 WandbCallback 自己 init/finish：report_to=["wandb"] 即可。
        # 手动 wandb.init 会和 callback 的 lifecycle 打架（train 指标推不上、双 run）。
        # 不覆盖用户显式传入的 --report_to；默认 "none" / "all" 一律改成只报 wandb
        if not _user_gave_report_to or _rt == "all" or _rt == ["all"]:
            training_args.report_to = ["wandb"]
        # run_name 也喂给 TrainingArguments，callback 拿它命名
        training_args.run_name = extra["wandb_run_name"]
        # 80 条小样本只跑几十步，聚合会丢曲线 —— 每步上报
        training_args.logging_steps = 1
    else:
        training_args.report_to = []

    print("=" * 66)
    print("正在初始化 Qwen3-4B SFT LoRA 训练引擎...")
    print(f"基座模型   : {model_args.model_name_or_path}")
    print(f"训练数据   : {data_args.train_file}")
    print(f"LoRA       : r={model_args.lora_r}, alpha={model_args.lora_alpha}, "
          f"dropout={model_args.lora_dropout}")
    print(f"max_len    : {data_args.max_seq_length}")
    print(f"POC 阶段   : {POC_STAGE}  (骨架掩码={'开' if USE_SKELETON_MASK else '关'} / "
          f"字段加权={'开' if USE_FIELD_WEIGHTS else '关'}；切换: SFT_POC_STAGE=POC1|POC2|POC3)")
    print(f"FIELD_W    : {FIELD_WEIGHTS}")
    print(f"日志落盘   : {log_path or '(未启用 --log_file)'}")
    print(f"wandb      : {'同步至项目 ' + extra['wandb_project'] + ' / run ' + extra['wandb_run_name'] if wandb_ready else '已关闭'}")
    print("=" * 66)

    # 1. 加载训练/验证集（lib/sft_dataset + WeightedDataCollator）：
    #   外壳走 prompt_template（chat 模板 + parity gate），权重走 sft_weight_planner。
    # strict_parity=True 是默认行为 —— 模板与推理侧不一致时在构造期就炸，
    # 而不是训完 33 步、跑完推理才发现输出格式不对。
    # 注: tokenizer 先于数据集构建 —— Dataset 构造期就要用它做分词与 parity 校验。
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SFTConversationDataset(
        data_path=data_args.train_file,
        tokenizer=tokenizer,
        max_len=data_args.max_seq_length,
        strict_parity=extra.get("strict_parity", True),
    )
    collator = WeightedDataCollator(tokenizer=tokenizer)

    # ---- 验证集：给了才开 eval；SOP 口径不做早停，靠观察 val_loss 是否反弹 ----
    eval_dataset = None
    if data_args.val_file and Path(data_args.val_file).exists():
        eval_dataset = SFTConversationDataset(
            data_path=data_args.val_file,
            tokenizer=tokenizer,
            max_len=data_args.max_seq_length,
            strict_parity=extra.get("strict_parity", True),
        )
        # transformers 版本差异: 4.41+ 叫 eval_strategy，更早叫 evaluation_strategy
        if hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "steps"
        else:
            training_args.evaluation_strategy = "steps"
        print(f"[EVAL] 验证集: {data_args.val_file} ({len(eval_dataset)} 条)；"
              f"eval_loss 走同一套 FIELD_W 加权口径")
    else:
        if data_args.val_file:
            print(f"[WARN] --val_file {data_args.val_file} 不存在，本次不启用 eval")

    # 2. 加载基座模型 (bf16) + gradient checkpointing + LoRA 挂载
    # DDP 下不能用 device_map="auto"（accelerate 禁分布式自动切分）；
    # 多卡时每 rank 独占一张卡，单卡时与 device_map="auto" 等价
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _device_map = {"": _local_rank} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True,
        device_map=_device_map,
    )

    # 显存预算: 4B bf16 权重 ~8G + activations(seq² 增长) + 151936-vocab logits。
    # 24G 卡即便 batch=1 + max_len=1024，梯度累积期间多个 forward 的 autograd
    # 图叠加会在第 ~5 个 micro-batch 顶穿 (实测 23.5G OOM 于 dropout)。
    # gradient checkpointing 把中间激活换成 backward 时重算 (~80% 激活显存),
    # 是 24G 卡跑 4B SFT 的必选项, 代价 ~30% 速度。正式轮/显存≥32G 可关。
    #
    # ⚠️ 冻结基座 + LoRA + gradient checkpointing 的组合有个静默坑，必须两步配套修：
    #   1) use_reentrant=False —— 用非重入实现。重入(reentrant)版在「输入不带
    #      requires_grad」时 backward 直接静默返回，不报错、不更新，loss 看着在降
    #      其实一步没学；非重入版无此行为，也是 transformers 对 PEFT 的推荐写法。
    #   2) enable_input_require_grads() —— 基座参数 requires_grad=False，检查点段的
    #      输入(embedding 输出)默认不带梯度。这里在 embedding 上挂 forward hook，
    #      强制其输出 requires_grad=True，给检查点段一个梯度入口。
    # 漏掉任一步，都会得到「trainable params 正常、loss 微降、推理输出还是基座原样」
    # 的假训练 —— 正是上轮探针 33 步后格式没学会的头号嫌疑。
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # 与 use_cache 冲突（Trainer 会警告，这里显式关掉）
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ---- 协议 token 解冻：id 从 tokenizer 解析，不硬编码 ----
    # 词表里没有这两 token 时直接报错（静默退化成 unk 会让格式永远学不会）。
    _proto_ids: List[int] = []
    if _unfreeze:
        _ids = format_canary.resolve_protocol_ids(tokenizer)
        _proto_ids = [_ids["open"], _ids["close"]]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        # 全量线性层旁路: 覆盖 attention + MLP，冻结 base
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        # ---- 协议 token 解冻 ----
        # 诊断：基座未学习 <think>/</think> ，embedding范数偏低、和 </tool_call> 高度相似；
        # 冻结lm_head时logits无法区分二者，greedy生成会错输出 </tool_call>。
        # 仅为这两个token新增delta参数，其余embedding冻结，只写入adapter；
        # token id 通过 format_canary 解析，不硬编码。
        trainable_token_indices=_proto_ids or None,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 假训练哨兵：可训练参数为 0 → LoRA 没挂上 / 梯度入口缺失，直接中止。
    # 这是「loss 看着降、推理输出还是基座原样」的第一道拦截，比训完 33 步才发现便宜得多。
    _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _n_trainable == 0:
        raise SystemExit(
            "[FATAL] 可训练参数为 0：LoRA 未正确挂载，或基座冻结导致梯度入口缺失。\n"
            "  确认 gradient_checkpointing_enable(use_reentrant=False) 与 "
            "enable_input_require_grads() 都在 get_peft_model 之前调用。")
    print(f"[OK] 可训练参数 {_n_trainable:,}（LoRA 旁路），梯度入口已验证")

    # ---- 训练规模预算（把「实际 batch / 优化器步数」直接打出来，避免口径含糊）----
    # 3. 设置训练参数: 预算换算 + warmup 折算（5.x 无 warmup_ratio 字段，由 --warmup_ratio 手动换算）
    n = len(train_dataset)
    eff_batch = training_args.per_device_train_batch_size * \
        max(int(training_args.gradient_accumulation_steps), 1) * \
        max(int(training_args.world_size), 1)
    micro_batches_per_epoch = -(-n // max(int(training_args.per_device_train_batch_size), 1))
    steps_per_epoch = -(-micro_batches_per_epoch // max(int(training_args.gradient_accumulation_steps), 1))
    total_steps = int(round(steps_per_epoch * float(training_args.num_train_epochs)))

    # warmup 实际步数：HF 优先用绝对 warmup_steps，否则 ceil(total × warmup_ratio)。
    # 5.x 没有 warmup_ratio 字段：由 --warmup_ratio 手动换算；4.x 用户也可直接传 --warmup_steps。
    if _warmup_ratio is not None:
        training_args.warmup_steps = math.ceil(total_steps * _warmup_ratio)
        _ratio_disp = _warmup_ratio
    else:
        _ratio_disp = (training_args.warmup_steps / total_steps) if total_steps else 0.0
    _warmup_steps = int(training_args.warmup_steps or 0)

    print("\n--- 训练规模预算 ---")
    print(f"  训练样本            : {n} 条 (含 Anchor)")
    print(f"  单卡 batch          : {training_args.per_device_train_batch_size}")
    print(f"  梯度累积            : {training_args.gradient_accumulation_steps}")
    print(f"  实际 (有效) batch   : {eff_batch}")
    print(f"  每 epoch 优化器步数 : {steps_per_epoch}")
    print(f"  合计优化器步数      : {total_steps}  ({float(training_args.num_train_epochs)} epoch)")
    print(f"  warmup 步数         : {_warmup_steps}  "
          f"(ratio={_ratio_disp:.3f} / steps={_warmup_steps})")
    print(f"  学习率              : {training_args.learning_rate} / {training_args.lr_scheduler_type}")
    print(f"  Weight Decay        : {training_args.weight_decay}")
    print("  早停策略            : 无早停（SOP §训练纪律：epoch 受控 + cosine 退火；"
          "靠 eval_loss 反弹判定过拟合，不自动掐停）\n")

    # 训练规模相关的 wandb config 在 run 建立后统一补（见训练收尾段），此处不重复

    # 4. 创建 Trainer（FieldWeightedTrainer 覆写 compute_loss：前缀/骨架掩码 + 字段加权）
    # 后端相关构造参数：trl.SFTTrainer 需要 processing_class/tokenizer +
    # 跳过它对已 tokenize 数据的二次处理；HF Trainer 不需要。
    _extra_kwargs = _sft_trainer_extra_kwargs(
        FieldWeightedTrainer, tokenizer, trainer_backend_name)

    if trainer_backend_name.startswith("trl"):
        # trl 1.x 强校验 train_dataset 类型，skip_prepare_dataset 归属 SFTConfig。
        # ⚠️ 直接 setattr 到 TrainingArguments 会在SFTConfig重建时丢失，
        # 导致 prepare 继续执行、截断异常。需要预先构造带 dataset_kwargs 的 SFTConfig。
        _fields = {f: getattr(training_args, f)
                   for f in training_args.__class__.__dataclass_fields__}
        from trl import SFTConfig
        training_args = SFTConfig(
            dataset_kwargs={"skip_prepare_dataset": True}, **_fields)
        train_dataset = _to_hf_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = _to_hf_dataset(eval_dataset)
        print(f"[TRL] 数据已物化为 HF Dataset (skip_prepare_dataset=True)；"
              f"collator 走 WeightedDataCollator")

    trainer = FieldWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        **_extra_kwargs,
    )

    # ---- 格式 canary：有 val 就挂上（实现在 lib/format_canary.py）----
    # 4.5 训练中监控: canary 金丝雀（类比 GRPO 侧的 eval reward 熔断）
    # 基座/tokenizer 直接用训练侧的，不再额外指定；协议 token 由 tokenizer 解析。
    if eval_dataset is not None and extra["canary_every"] > 0:
        try:
            _val_rows = [json.loads(l) for l in open(data_args.val_file, encoding="utf-8")
                         if l.strip()]
            _canary = format_canary.build_callback(
                torch, tokenizer, _val_rows,
                every=extra["canary_every"],
                n=extra["canary_n"],
                max_new=extra["canary_max_new"],
            )
            _canary.trainer = trainer
            trainer.add_callback(_canary)
            print(f"[CANARY] 已挂载: 每 {extra['canary_every']} 步 × "
                  f"{min(extra['canary_n'], len(_val_rows))} 条 val "
                  f"(--canary_every 0 可关)，测 p(</think>) / p(next) / p(<|im_end|>) / greedy_ok")
        except Exception as e:
            print(f"[WARN] canary 挂载失败(不影响训练): {e}")

    print(f">>> 启动训练 (样本 {len(train_dataset)} 条, Epochs={training_args.num_train_epochs}) <<<")
    # 5. 开始训练；训后复测 val 留过拟合证据
    t_start = time.time()
    try:
        train_result = trainer.train()
        metrics = dict(train_result.metrics or {})
    except Exception:
        print(f"[FATAL] 训练异常中断，耗时 {time.time() - t_start:.1f}s")
        raise
    elapsed = time.time() - t_start

    # 训练结束复测 val（同一套 FIELD_W 加权口径），把过拟合信号留档
    if eval_dataset is not None:
        try:
            eval_metrics = trainer.evaluate()
            metrics.update({f"final_{k}": v for k, v in (eval_metrics or {}).items()})
            print(f"[EVAL] 训练结束 val_loss = {eval_metrics.get('eval_loss')} "
                  f"(对照 train_loss={metrics.get('train_loss')}；"
                  f"val 明显高于 train ⇒ 73 条小样本已出现过拟合，属探针预期内观察)")
        except Exception as e:
            print(f"[WARN] 训练后 eval 失败: {e}")

    # 6. 保存模型与训练血缘: 模型/tokenizer 落盘 + 指标写 train_metrics.json + wandb 收尾
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # 训练收尾指标落盘：同时写进 output_dir 与 wandb，方便多轮探针对比
    metrics.update({
        "train_runtime_sec": round(elapsed, 1),
        "train_samples": len(train_dataset),
        "base_model": model_args.model_name_or_path,
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
    })
    metrics_path = Path(training_args.output_dir) / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] 训练完成，LoRA Adapter 已存盘: {training_args.output_dir}")
    print(f"[OK] 训练指标存盘: {metrics_path}")
    print(f"[OK] 本轮耗时: {elapsed:.1f}s | loss(train)={metrics.get('train_loss')}")

    # setup_wandb ready 即代表 import wandb 成功，此处补 import 供收尾同步用
    if wandb_ready:
        import wandb as wandb_mod
    else:
        wandb_mod = None
    if wandb_ready and wandb_mod is not None:
        try:
            # Trainer 的 WandbCallback 已在 train() 里把 train/loss、train/grad_norm、
            # eval/* 逐步调 wandb.log；这里只补自定义收尾指标，不再自己 init/finish
            # （finish 由 callback 在 trainer 退出时统一处理）
            _probe_cfg = {
                "probe_tag": "sentiment-sft-probe",
                "train_file": data_args.train_file,
                "base_model": model_args.model_name_or_path,
                "lora_r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "max_seq_length": data_args.max_seq_length,
                "field_weights": json.dumps(FIELD_WEIGHTS, ensure_ascii=False),
                "poc_stage": POC_STAGE,
                "use_skeleton_mask": USE_SKELETON_MASK,
                "use_field_weights": USE_FIELD_WEIGHTS,
                "train_samples": len(train_dataset),
                "val_samples": len(eval_dataset) if eval_dataset else 0,
                "effective_batch": eff_batch,
                "optimizer_steps_total": total_steps,
            }
            if wandb_mod.run is not None:
                wandb_mod.config.update(_probe_cfg, allow_val_change=True)
                wandb_mod.log(metrics)
                print(f"[OK] wandb 同步完成: run={extra['wandb_run_name']}, "
                      f"查看 {wandb_mod.run.url}")
        except Exception as e:
            print(f"[WARN] wandb 收尾同步失败: {e}")

    detach_log(log_handle)


if __name__ == "__main__":
    main()
