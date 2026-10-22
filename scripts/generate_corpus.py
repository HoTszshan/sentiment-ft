#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情负面精判 · LLM 构造语料生成与蒸馏标注流水线

适用场景: **无法接入公司真实日志时**，用 LLM 构造语料 + 异族 LLM 打标，
          产出可跑通训练的构造区数据集。

严格遵守 SOP:
  - §1.3.2 构造区数据一律 provenance="synthetic"，严禁标 real
  - §3.2  打标模型与 judge 必须异族（生成模型与打标模型亦应异族）
  - §7.2  meta 必填 label_model / source / anno_prompt_version
  - §3.1  四查一票否决（JSON合法 / evidence子串 / CoT自洽 / judge support）
  - §2.3  MinHash 去重（3-gram Jaccard ≥ 0.8 视为重复）

用法:
  # 1. 连通性自检
  python scripts/generate_corpus.py --selftest

  # 2. 小批试跑（先跑 5 条看质量）
  python scripts/generate_corpus.py --n 5 --out output/corpus/smoke.jsonl

  # 3. 正式生成（生成用 Qwen-Max，打标用 DeepSeek——异族）
  python scripts/generate_corpus.py --n 100 \
      --gen_provider dashscope --gen_model qwen-max \
      --anno_provider deepseek --anno_model deepseek-chat \
      --out output/corpus/synth_100.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from llm_client import LLMClient, parse_json_loose  # noqa: E402
from evidence_support import judge_evidence_support  # noqa: E402
from rubric_check import rubric_check  # noqa: E402
from corpus_schema import (  # noqa: E402
    INDUSTRY_SCHEMAS, SUBJECTS, PLATFORM_VOICES, PERSONAS, REGISTERS,
    LENGTH_HINTS, HARD_CASES, HC_SENT_ALLOWED, HC_REL_ALLOWED,
    VALID_HC, VALID_SENT, VALID_REL, VALID_RISK,
    load_prompts, schema_to_inline,
)


# ============================================================
# 特征规划：确保多样性
# ============================================================

def plan_specs(n: int, seed: int = 42, neg_ratio: float = 0.55,
               filter_hard_cases: Optional[List[str]] = None,
               filter_sentiments: Optional[List[str]] = None,
               filter_relevances: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    规划 n 条生成规格。支持定向指定 hard_case / sentiment / relevance。
    """
    rng = random.Random(seed)
    combos = [(p, s, r, l)
              for p in PLATFORM_VOICES for s in PERSONAS
              for r in REGISTERS for l in LENGTH_HINTS]
    rng.shuffle(combos)
    if n > len(combos):
        raise ValueError(f"n={n} 超出可用随机化组合数 {len(combos)}")

    inds = list(INDUSTRY_SCHEMAS.keys())
    target_hc_pool = filter_hard_cases or HARD_CASES

    specs = []
    for i in range(n):
        p, s, r, l = combos[i]
        hc = target_hc_pool[i % len(target_hc_pool)]
        ind = inds[i % len(inds)]

        # 极性：若有定向指定则从定向池与合法池的交集中选，否则按默认
        allowed = HC_SENT_ALLOWED[hc]
        cand_sentiments = [x for x in (filter_sentiments or ["negative", "positive", "neutral"]) if x in allowed]
        if not cand_sentiments:
            cand_sentiments = sorted(list(allowed))
        sent = rng.choice(cand_sentiments)

        # 语篇地位：若有定向指定则从交集中选，否则默认 core 为主
        rel_allowed = sorted(HC_REL_ALLOWED[hc])
        cand_rels = [x for x in (filter_relevances or []) if x in rel_allowed]
        if cand_rels:
            relevance = rng.choice(cand_rels)
        else:
            relevance = ("core" if rng.random() < 0.7 or len(rel_allowed) == 1
                         else rng.choice([x for x in rel_allowed if x != "core"]))

        specs.append({
            "idx": i,
            "target_subject": rng.choice(SUBJECTS[ind]),
            "industry": ind,
            "hard_case_type": hc,
            "sentiment": sent,
            "relevance": relevance,
            "length_hint": l,
            "platform": p, "persona": s, "register": r,
        })
    return specs


# ============================================================
# 四查质检
# ============================================================


def check_four(text: str, target: str, out_raw: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    四查中的①②③（查④ judge support 需另一次 API 调用，见 judge_support()）。
    返回 (是否通过, 解析结果, 失败原因)
    """
    # 查① JSON 合法 + 结构
    if "<think>" not in out_raw or "</think>" not in out_raw:
        return False, {}, "查① 缺少思考标签"
    try:
        cot = out_raw[out_raw.index("<think>") + len("<think>"):
                      out_raw.index("</think>")]
        after = out_raw[out_raw.index("</think>") + len("</think>"):]
        if not after.lstrip().startswith("{"):
            return False, {}, "查① 尾赘：end_of_thought 后首字符非 {"
        pred = parse_json_loose(after)
    except Exception as e:
        return False, {}, f"查① JSON 解析失败: {e}"

    req = ["subject", "relevance", "sentiment", "hard_case_type", "risk_categories", "evidence"]
    if list(pred.keys()) != req:
        return False, pred, f"查① 字段不符: {list(pred.keys())}"
    if pred["relevance"] not in VALID_REL:
        return False, pred, f"查① relevance 非法枚举: {pred['relevance']}"
    if pred["sentiment"] not in VALID_SENT:
        return False, pred, f"查① sentiment 非法枚举: {pred['sentiment']}"
    if pred["hard_case_type"] not in VALID_HC:
        return False, pred, f"查① hard_case_type 非法枚举: {pred['hard_case_type']}"
    if not isinstance(pred["risk_categories"], list):
        return False, pred, "查① risk_categories 非列表"
    bad = [r for r in pred["risk_categories"] if r not in VALID_RISK]
    if bad:
        return False, pred, f"查① risk 非法枚举: {bad}"

    # 查② evidence 严格子串
    ev = pred["evidence"]
    if ev not in text:
        return False, pred, f"查② evidence 非原文子串: {ev[:30]!r}"
    if len(ev) < 4:
        return False, pred, f"查② evidence 过短: {ev!r}"
    if len(ev) > 80:
        return False, pred, f"查② evidence 过长({len(ev)}字>80): {ev[:30]!r}…"
    # 单句截取：句末标点不得出现在 evidence 中间（尾部标点允许）
    body = ev.rstrip("。！？；…")
    if any(p in body for p in "。！？"):
        return False, pred, f"查② evidence 跨句拼接: {ev[:30]!r}"

    # 查③ CoT↔JSON 自洽 + 非空转
    if pred["hard_case_type"] not in cot:
        return False, pred, f"查③ CoT 未提及 hard_case={pred['hard_case_type']}"

    sent_zh = {"negative": "负面", "positive": "正面", "neutral": "中性"}.get(pred["sentiment"], "")
    if pred["sentiment"] not in cot and (not sent_zh or sent_zh not in cot):
        return False, pred, f"查③ CoT 未提及 sentiment={pred['sentiment']}"
    # 空转检测：CoT 必须引用原文的任一连续片段（而非仅复述结论）。
    # 只要 CoT 中存在原文的任一 >=8 字连续片段即判为"有引用"。
    norm_cot = re.sub(r"\s+", "", cot)
    cited = False
    for i in range(0, max(0, len(text) - 8)):
        chunk = text[i:i + 8]
        if re.sub(r"\s+", "", chunk) in norm_cot:
            cited = True
            break
    if not cited:
        return False, pred, "查③ CoT 空转（未引用原文）"

    return True, pred, ""


def judge_support(client: LLMClient, text: str, target: str, pred: Dict[str, Any]) -> Tuple[bool, str]:
    """
    查④：独立的 judge 判断 evidence 能否推出 sentiment+risk。
    调用方必须保证 client 与被测样本的打标模型【异族】（SOP §3.2 铁律）。

    **判定逻辑已下沉到 `lib/evidence_support.py`**（与评估手册 Rubric 第③项
    共调的唯一实现）—— 本函数只负责「把 client 适配成 callable」，
    不再自己拼 prompt、自己解析结论。
    """
    def _call(prompt: str) -> str:
        return client.chat([{"role": "user", "content": prompt}],
                            temperature=0, max_tokens=300)

    return judge_evidence_support(_call, text, target, pred)


# ============================================================
# 去重（3-gram Jaccard）
# ============================================================

def trigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================
# 主流程
# ============================================================

def build_one(gen: LLMClient, anno: LLMClient, spec: Dict[str, Any],
              gen_sys: str, anno_sys: str, judge: Optional[LLMClient],
              keep_cot: bool = True) -> Tuple[Optional[Dict], str]:
    """生成一条并打标，返回 (样本 or None, 失败原因)"""
    ind = spec["industry"]

    # ---- 步骤1: 生成文本 ----
    user = (
        f"【目标主体】{spec['target_subject']}\n"
        f"【所属行业】{ind}\n"
        f"【目标难点类型】{spec['hard_case_type']}\n"
        f"【目标情感极性】{spec['sentiment']}\n"
        f"【目标语篇地位】{spec['relevance']}\n"
        f"【长度区间】{spec['length_hint']}\n"
        f"【本次随机化要求】平台口吻={spec['platform']}；作者人设={spec['persona']}；"
        f"语域={spec['register']}\n"
        f"【行业风险二级词表（供参考，不必全部覆盖）】\n{schema_to_inline(ind)}"
    )
    try:
        g = gen.chat_json(
            [{"role": "system", "content": gen_sys},
             {"role": "user", "content": user}],
            temperature=0.95, max_tokens=2400,
        )
    except Exception as e:
        return None, f"生成失败: {e}"

    text = (g.get("text") or "").strip()
    if len(text) < 20:
        return None, f"生成文本过短({len(text)}字)"

    # ---- 步骤2: 异族模型打标 ----
    anno_user = (
        f"【目标主体 target_subject】{spec['target_subject']}\n"
        f"【所属行业 industry】{ind}\n"
        f"【行业风险二级词表 custom_schema】\n{schema_to_inline(ind)}\n\n"
        f"【待标注原文】\n{text}"
    )
    try:
        out_raw = anno.chat(
            [{"role": "system", "content": anno_sys},
             {"role": "user", "content": anno_user}],
            temperature=0, max_tokens=3000,
        )
    except Exception as e:
        return None, f"打标失败: {e}"

    # ---- 步骤3: 四查①②③ ----
    ok, pred, why = check_four(text, spec["target_subject"], out_raw)
    if not ok:
        return None, why

    # ---- 步骤3.5: 目标命中校验（生成端要的 vs 打标端判的）----
    # 不命中说明生成端没把目标特征写显性，或该文本本身歧义大。
    # 两种情况都不该进训练集——放行只会污染配比、引入噪声标签。
    if pred["hard_case_type"] != spec["hard_case_type"]:
        return None, (f"目标未命中: 想写 {spec['hard_case_type']}，"
                      f"打标判成 {pred['hard_case_type']}")
    if pred["sentiment"] != spec["sentiment"]:
        return None, (f"目标未命中: 想写 {spec['sentiment']}，"
                      f"打标判成 {pred['sentiment']}")
    if pred["relevance"] != spec["relevance"]:
        return None, (f"目标未命中: 想写 relevance={spec['relevance']}，"
                      f"打标判成 {pred['relevance']}")

    # ---- 步骤4: 查④ judge support（异族）----
    if judge is not None:
        sup_ok, reason = judge_support(judge, text, spec["target_subject"], pred)
        if not sup_ok:
            return None, f"查④ support=no: {reason}"

    # ---- 步骤4.5: 离线 Rubric 推理链复核（评估手册 TODO 5.4）----
    # Rubric 判三项（CoT 是否点名主体 / 推理与标签是否矛盾 / 证据是否撑得起结论）；
    # 第③项「证据支撑」与查④ **同一份实现**（lib/evidence_support.py），不重复判两次。
    # 裁判模型与打标模型必须异族（与查④ 同一 client，沿用异族铁律）。
    if judge is not None:
        _cot = out_raw[out_raw.index("<think>") + len("<think>"):
                       out_raw.index("</think>")].strip()
        rb = rubric_check(cot=_cot, pred=pred, input_text=text,
                          judge=lambda p: judge.chat(
                              [{"role": "user", "content": p}],
                              temperature=0, max_tokens=300))
        if not rb["pass"]:
            return None, f"Rubric 复核未过: {rb['reason']}"

    # ---- 组装交付样本（输入侧 = 线上推理模板，铁律）----
    instr = (f"你是舆情负面精判模型。目标主体：【{spec['target_subject']}】；"
             f"行业：【{ind}】；行业风险口径：\n{schema_to_inline(ind)}")

    if keep_cot:
        cot = out_raw[out_raw.index("<think>") + len("<think>"):
                      out_raw.index("</think>")].strip()
        output = (f"<think>\n{cot}\n</think>"
                  + json.dumps(pred, ensure_ascii=False))
    else:
        output = json.dumps(pred, ensure_ascii=False)

    return {
        "instruction": instr,
        "input": text,
        "output": output,
        "meta": {
            # SOP §7.2 必填血缘
            "hard_case_type": pred["hard_case_type"],
            "sentiment": pred["sentiment"],
            "relevance": pred["relevance"],
            "event_id": f"evt_synth_{spec['idx']:04d}",
            "pair_id": None,
            # 构造区强制标记 —— 严禁改成 real
            "provenance": "synthetic",
            "source": "llm_generated",
            "industry": ind,
            "label_model": f"{anno.provider}/{anno.model}",
            "gen_model": f"{gen.provider}/{gen.model}",
            "judge_model": f"{judge.provider}/{judge.model}" if judge else None,
            "anno_prompt_version": "anno_prompt_v1.0",
            "gen_prompt_version": "gen_prompt_v1.0",
            # 生成时的随机化维度，供多样性审计
            "gen_spec": {"platform": spec["platform"], "persona": spec["persona"],
                         "register": spec["register"], "length_hint": spec["length_hint"],
                         # 目标标签一并落盘：这是唯一可信的目标真值。
                         # 审计若用 plan_specs 复算，sentiment 的贪心配比依赖总 n，
                         # n 对不上就会错位对齐（假阳性告警）。
                         "hard_case_type": spec["hard_case_type"],
                         "sentiment": spec["sentiment"],
                         "relevance": spec["relevance"]},
        },
    }, ""


def main():
    ap = argparse.ArgumentParser(description="LLM 构造语料生成 + 异族蒸馏标注")
    ap.add_argument("--n", type=int, default=10, help="生成条数")
    ap.add_argument("--out", default="output/corpus/synth.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_provider", default="dashscope")
    ap.add_argument("--gen_model", default="qwen-max")
    ap.add_argument("--anno_provider", default="deepseek",
                    help="必须与 gen_provider 异族 (SOP §3.2)")
    ap.add_argument("--anno_model", default="deepseek-chat")
    ap.add_argument("--judge_provider", default=None,
                    help="留空则跳过查④；指定则必须与 anno 异族")
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--no_judge", action="store_true", help="显式跳过查④（省成本）")
    ap.add_argument("--max_dedup_jaccard", type=float, default=0.8)
    ap.add_argument("--max_retry", type=int, default=3,
                    help="单条规格四查失败后的重试次数（每次重跑生成+打标）")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="并发线程数（推荐 4~8 并发）")
    ap.add_argument("--append", action="store_true",
                    help="追加模式：保留已有文件内容并在此基础上增量追加")
    ap.add_argument("--cross_dedup", nargs="+", default=None,
                    help="跨文件去重参考（可多个）。隔离/评测批生成时传入训练集，"
                         "避免与训练集产生近邻重复污染")
    ap.add_argument("--filter_hard_cases", nargs="+", default=None,
                    help="定向生成指定的难点类型，例如: --filter_hard_cases fact_implicit")
    ap.add_argument("--filter_sentiments", nargs="+", default=None,
                    help="定向生成指定的情感极性，例如: --filter_sentiments neutral negative")
    ap.add_argument("--filter_relevances", nargs="+", default=None,
                    help="定向生成指定的语篇地位，例如: --filter_relevances passing")
    ap.add_argument("--selftest", action="store_true", help="仅测试连通性")
    args = ap.parse_args()

    gen_sys, anno_sys = load_prompts()
    print(f"提示词已加载: gen={len(gen_sys)} 字符, anno={len(anno_sys)} 字符")

    if args.selftest:
        for prov, mdl in [(args.gen_provider, args.gen_model),
                          (args.anno_provider, args.anno_model)]:
            print(f"\n--- 自检 {prov}/{mdl} ---")
            try:
                c = LLMClient(provider=prov, model=mdl)
                out = c.chat([{"role": "user", "content": "只回复两个字：连通"}],
                             temperature=0, max_tokens=20)
                print(f"  OK 回复={out!r}  {c.usage.summary()}")
            except Exception as e:
                print(f"  FAIL {e}")
        return

    # 异族校验（SOP §3.2 铁律）
    if args.gen_provider == args.anno_provider and args.gen_model == args.anno_model:
        print("[FATAL] 生成模型与打标模型不能同族（SOP §3.2）。请换 anno 模型。")
        sys.exit(2)

    judge = None
    if not args.no_judge and args.judge_provider:
        if args.judge_provider == args.anno_provider and args.judge_model == args.anno_model:
            print("[FATAL] judge 与打标模型同族将导致 support 虚高（SOP §3.2）。")
            sys.exit(2)
        judge = LLMClient(provider=args.judge_provider, model=args.judge_model)
        print(f"judge: {judge.provider}/{judge.model}")
    else:
        print("查④ judge support: 已跳过（--no_judge 或未指定 judge_provider）")

    gen = LLMClient(provider=args.gen_provider, model=args.gen_model)
    anno = LLMClient(provider=args.anno_provider, model=args.anno_model)
    print(f"生成模型: {gen.provider}/{gen.model}")
    print(f"打标模型: {anno.provider}/{anno.model}")

    specs = plan_specs(args.n, seed=args.seed,
                       filter_hard_cases=args.filter_hard_cases,
                       filter_sentiments=args.filter_sentiments,
                       filter_relevances=args.filter_relevances)
    print(f"已规划 {len(specs)} 条生成规格（随机化组合互不重复）\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    file_lock = threading.Lock()
    dedup_lock = threading.Lock()
    seen_tris = []

    # 跨文件去重：将训练集（或其他参考文件）的 trigram 指纹并入去重池，
    # 用于「探针/评测隔离批」生成，杜绝与训练集的文本近邻污染
    if args.cross_dedup:
        for ref in args.cross_dedup:
            ref_path = Path(ref)
            if not ref_path.exists():
                print(f"⚠️ --cross_dedup 参考文件不存在，跳过: {ref_path}")
                continue
            n_ref = 0
            with open(ref_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            row = json.loads(line)
                            if "input" in row:
                                seen_tris.append(trigrams(row["input"]))
                                n_ref += 1
                        except Exception:
                            pass
            print(f"跨文件去重: 已载入参考文件 {ref_path} 的 {n_ref} 条指纹")
        print()

    # 追加模式：读取已有文件内容提取 trigrams 确保全局去重
    if args.append and out_path.exists():
        n_self = len(seen_tris)
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        if "input" in row:
                            seen_tris.append(trigrams(row["input"]))
                    except Exception:
                        pass
        print(f"追加模式: 已加载现有文件中的 {len(seen_tris) - n_self} 条样本特征用于去重"
              f"（去重池当前共 {len(seen_tris)} 条）\n")

    out_file = open(out_path, mode, encoding="utf-8")
    accepted, rejected = [], []
    t0 = time.time()
    total_specs = len(specs)
    completed_counter = 0
    counter_lock = threading.Lock()

    def process_one(spec: Dict[str, Any]):
        nonlocal completed_counter
        last_why = ""
        s = None
        for attempt in range(1, args.max_retry + 1):
            s, why = build_one(gen, anno, spec, gen_sys, anno_sys, judge)
            if s is not None:
                tris = trigrams(s["input"])
                with dedup_lock:
                    dup = next((j for j, t in enumerate(seen_tris)
                                if jaccard(tris, t) >= args.max_dedup_jaccard), None)
                    if dup is None:
                        seen_tris.append(tris)
                        break
                    s, why = None, f"与已有样本重复 (Jaccard≥{args.max_dedup_jaccard})"
            last_why = why

        with counter_lock:
            completed_counter += 1
            curr = completed_counter

        if s is None:
            with counter_lock:
                rejected.append({"idx": spec["idx"], "reason": last_why,
                                 "attempts": args.max_retry})
            print(f"  [{curr}/{total_specs}] ✗ {last_why}  (重试 {args.max_retry} 次均失败)", flush=True)
            return None
        else:
            with file_lock:
                out_file.write(json.dumps(s, ensure_ascii=False) + "\n")
                out_file.flush()
            with counter_lock:
                accepted.append(s)
            print(f"  [{curr}/{total_specs}] ✓ {spec['industry']}/{spec['hard_case_type']} "
                  f"rel={spec['relevance']} len={len(s['input'])} 字", flush=True)
            return s

    concurrency = max(1, args.concurrency)
    print(f"启动生成流水线: 并发线程数={concurrency}, 计划规格={total_specs}\n")

    if concurrency == 1:
        for spec in specs:
            process_one(spec)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one, spec) for spec in specs]
            concurrent.futures.wait(futures)

    out_file.close()

    # 多样性与配比审计
    print(f"\n{'=' * 66}")
    print(f"产出: {out_path}  ({len(accepted)} 条通过 / {len(rejected)} 条被拒)")
    print(f"耗时: {time.time() - t0:.1f}s")
    print(f"生成用量: {gen.usage.summary()}")
    print(f"打标用量: {anno.usage.summary()}")
    if judge:
        print(f"judge 用量: {judge.usage.summary()}")

    if accepted:
        tpl = Counter(re.sub(r"[一-龥A-Za-z0-9]+", "X", s["input"])[:60]
                      for s in accepted)
        print(f"\n多样性: 归一化后 {len(tpl)} 个不同文本骨架 / {len(accepted)} 条 "
              f"(最高重复 {tpl.most_common(1)[0][1]} 次)")
        print("难点分布:", dict(Counter(s["meta"]["hard_case_type"] for s in accepted)))
        print("极性分布:", dict(Counter(s["meta"]["sentiment"] for s in accepted)))
        print("行业分布:", dict(Counter(s["meta"]["industry"] for s in accepted)))

    if rejected:
        print(f"\n被拒原因统计:")
        for reason, cnt in Counter(r["reason"].split(":")[0] for r in rejected).most_common():
            print(f"  {cnt:>3}  {reason}")

    print("=" * 66)
    print("注意: 所有样本 provenance=synthetic（构造区），严禁标 real，不混真实区统计。")


if __name__ == "__main__":
    main()
