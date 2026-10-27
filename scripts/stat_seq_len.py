#!/usr/bin/env python3
"""
scripts/stat_seq_len.py
训练数据序列长度统计：按 domain / anchor 分账，输出 prompt / output / total 的
p50 / p95 / p99 / max 分位数，定位 OOM 长样本来源。

用法（AutoDL，需 transformers 已装 + 基座路径有 tokenizer）:
  python3 scripts/stat_seq_len.py \
      --train_file /root/autodl-tmp/datasets/mixed/sft_train.jsonl \
      --tokenizer /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
      --top_n 5

可选产物 (都不给 = 只打印 stdout, 零副作用):
  --out <path.json>   机器可读统计落盘 (分桶分位 / Top-N / R_length 落档计数),
                      回填手册或留存证据用
  --log <path.log>    整份 stdout 副本追加写日志 (可复跑不覆盖)

无 tokenizer 环境（任何机器）退化为字符级近似（--no_tokenizer 自动触发）:
  python3 scripts/stat_seq_len.py --train_file data/train_synth_81.jsonl

口径说明:
  - prompt = <|im_start|>system...assistant\n 模板 + instruction + input（前缀掩码区）
  - output = assistant 输出（CoT + JSON）+ <|im_end|>
  - total  = prompt + output，与训练 max_seq_length 同口径（含模板，不含 padding）
  - **cot  = output 里 <think>…</think> 之间的推理段**（不含标签、不含 JSON 骨架），
    与 reward_fns 的 R_length 同口径（R_length 只量这一段）。带 tokenizer 跑时
    额外打印「R_length 落档」——直接看现有阈值 40/65/120 在这批数据上是否还有效。
  - 按 meta.provenance == public_anchor 分 anchor / domain 两桶，直接回答
    「长 token 是领域样本还是锚点带来的」
"""

import argparse
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

# 外壳与训练/推理同源；长度口径必须与实际喂进模型的序列一致，否则 p99 定的 max_len 是假的
import prompt_template as pt  # noqa: E402

# CoT 段的切分与 R_length 阈值同源：统计口径必须与奖励口径一致，否则「按统计结果定阈值」
# 定出来的是另一个任务的长度
from schema_contract import THOUGHT_BEGIN, THOUGHT_END  # noqa: E402
from reward_fns import COT_DEAD, COT_IDEAL_HI, COT_SHORT  # noqa: E402

_COT_RE = re.compile(re.escape(THOUGHT_BEGIN) + r"(.*?)" + re.escape(THOUGHT_END),
                     re.DOTALL)


def split_cot(output: str):
    """拆出 (cot 段, json 段)。无 <think> 标签时 cot 记空串（与 parse_output 同语义）。"""
    m = _COT_RE.search(output or "")
    if not m:
        return "", output or ""
    return m.group(1).strip(), output[m.end():].strip()


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _bucket(row):
    meta = row.get("meta") or {}
    if meta.get("provenance") == "public_anchor" or \
            meta.get("hard_case_type") == "general_anchor":
        return "anchor"
    return "domain"


def _stats(totals, prompts, outputs, cots):
    """分桶 → 机器可读统计 (stdout 报告与 --out 落盘共用同一套数)。"""
    s = {}
    for name, v in (("prompt", sorted(prompts)), ("output", sorted(outputs)),
                    ("total", sorted(totals))):
        s[name] = {"p50": _pct(v, 50), "p95": _pct(v, 95),
                   "p99": _pct(v, 99), "max": v[-1] if v else 0}
    if cots:
        c = sorted(cots)
        s["cot"] = {"p50": _pct(c, 50), "p95": _pct(c, 95),
                    "p99": _pct(c, 99), "max": c[-1]}
    return s


def _report(tag, totals, prompts, outputs, unit, top_rows, top_n, cots=None, say=print):
    t = sorted(totals)
    p = sorted(prompts)
    o = sorted(outputs)
    say(f"\n--- {tag} ({len(t)} 条, 单位={unit}) ---")
    say(f"  {'':8s}{'p50':>8s}{'p95':>8s}{'p99':>8s}{'max':>8s}")
    for name, v in (("prompt", p), ("output", o), ("total", t)):
        say(f"  {name:8s}{_pct(v,50):>8d}{_pct(v,95):>8d}{_pct(v,99):>8d}{v[-1]:>8d}")
    if cots:
        c = sorted(cots)
        say(f"  {'cot':8s}{_pct(c,50):>8d}{_pct(c,95):>8d}{_pct(c,99):>8d}{c[-1]:>8d}"
            f"   <- R_length 只量这一段")
    if top_rows:
        top_sorted = sorted(top_rows, key=lambda x: -x[0])
        say(f"  Top{top_n} 最长样本:")
        for tok_n, idx, bucket, meta in top_sorted[:top_n]:
            hc = meta.get("hard_case_type", "?")
            src = meta.get("source", meta.get("provenance", "?"))
            say(f"    #{idx} {bucket:6s} {tok_n:>6d} {unit}  "
                f"hard_case={hc}  source={src}")


def _report_cot_buckets(cots, unit, say=print):
    """
    CoT 落在 R_length 哪一档 —— 直接回答「40/65/120 这组阈值还适用吗」。

    ⚠️ 阈值是在 **token 空间**定义的（reward_fns 用 tokenizer 计数）。字符级近似
    模式下这些数字没有可比性 —— 此时只报分位、不做落档判定，免得把「370 字符」
    拿去和「120 tokens」比大小得出错误结论。

    判定式与 reward_fns.r_length 逐条一致：
      n < COT_SHORT               → 过短罚  (n/40 − 1, −1.0~0)
      COT_SHORT ≤ n ≤ COT_IDEAL_HI → 满分 +1.0
      COT_IDEAL_HI < n < COT_DEAD  → 线性衰减 (1 − (n−65)/55, 1.0~0)
      n ≥ COT_DEAD                 → 归 0
    若「过短罚」与「满分」两档都为 0 条（全部挤在衰减/归 0），说明这组阈值对当前数据失效。
    """
    if not cots:
        return
    if unit != "token":
        c = sorted(cots)
        say(f"\n--- CoT 分位（单位=char；阈值是 token 口径，此处不做落档判定）---")
        say(f"  p50={_pct(c,50)}  p95={_pct(c,95)}  p99={_pct(c,99)}  max={c[-1]} 字符")
        say("  带 --tokenizer 重跑才会输出 R_length 落档判定。")
        return

    n_all = len(cots)
    short = sum(1 for n in cots if n < COT_SHORT)
    ideal = sum(1 for n in cots if COT_SHORT <= n <= COT_IDEAL_HI)
    decay = sum(1 for n in cots if COT_IDEAL_HI < n < COT_DEAD)
    dead = sum(1 for n in cots if n >= COT_DEAD)

    say(f"\n--- R_length 落档 (阈值 {COT_SHORT}/{COT_IDEAL_HI}/{COT_DEAD} {unit}) ---")
    for name, cnt, note in (
        (f"过短罚 <{COT_SHORT}", short, "n/40 − 1，越短越负"),
        (f"满分 {COT_SHORT}~{COT_IDEAL_HI}", ideal, "+1.0"),
        (f"衰减 {COT_IDEAL_HI}~{COT_DEAD}", decay, "线性 1 → 0"),
        (f"归零 ≥{COT_DEAD}", dead, "0"),
    ):
        bar = "█" * int(round(cnt / max(1, n_all) * 30))
        say(f"  {name:14s} {cnt:>4} ({cnt / n_all:>5.1%})  {bar}  {note}")

    if ideal == 0:
        say(f"  ⚠️ 理想带一条都没有：这组阈值与本批数据不匹配。")
        say(f"     当前 CoT 中位数 {_pct(sorted(cots), 50)} {unit}，"
            f"建议把「理想带」对准 p50 附近重设三档。")
        if dead == n_all:
            say("     R_length 本批数据上恒为 0 —— 该维暂时没有塑形效果（不影响判别正确性）。")


def main():
    ap = argparse.ArgumentParser(description="训练数据序列长度统计（domain/anchor 分账）")
    ap.add_argument("--train_file", required=True, help="训练集 jsonl（建议混合后的 sft_train.jsonl）")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer/基座路径；不给或加载失败则退化为字符级近似")
    ap.add_argument("--top_n", type=int, default=5, help="打印最长 Top-N 样本索引")
    ap.add_argument("--out", default=None,
                    help="统计结果落盘路径 (json)。不给则只打印 stdout，不落盘")
    ap.add_argument("--log", default=None,
                    help="追加写日志文件 (整份 stdout 副本 tee 进去)。不给则只打印 stdout")
    args = ap.parse_args()

    # --log: tee 模式, stdout 原样打印 + 整份副本追加进日志 (可复跑不覆盖, 留存时间线)
    log_f = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "a", encoding="utf-8")

    def say(msg=""):
        print(msg)
        if log_f:
            log_f.write(msg + "\n")

    rows = [json.loads(l) for l in open(args.train_file, encoding="utf-8") if l.strip()]
    say(f"数据: {args.train_file}  共 {len(rows)} 条")

    encode = None
    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            encode = lambda s: len(tok.encode(s, add_special_tokens=False))
            unit = "token"
        except Exception as e:
            tok = None
            say(f"⚠️ tokenizer 加载失败 ({e})，退化为字符级近似 (1 汉字≈1.4 token)")
    if encode is None:
        encode = lambda s: len(s or "")
        unit = "char"

    buckets = {}
    for i, r in enumerate(rows):
        b = _bucket(r)
        # 与训练/推理同源：chat 模板可用就用 chat（与 Dataset 同一渲染路径），否则手写
        prompt = pt.render_prompt(tok if args.tokenizer else None,
                                  r.get("instruction", ""),
                                  r.get("input", ""))
        output = (r.get("output", "") or "") + pt.ASSISTANT_SUFFIX
        cot_text, _json_text = split_cot(r.get("output", "") or "")
        d = buckets.setdefault(b, {"p": [], "o": [], "t": [], "c": [], "top": []})
        np_, no, nt = encode(prompt), encode(output), encode(prompt + output)
        d["p"].append(np_)
        d["o"].append(no)
        d["t"].append(nt)
        d["c"].append(encode(cot_text))
        d["top"].append((nt, i, b, r.get("meta") or {}))

    # total 降序拿 Top-N（按 token 实测或字符近似均可）
    top_all = sorted((t for d in buckets.values() for t in d["top"]),
                     key=lambda x: -x[0])

    for b in ("domain", "anchor"):
        if b in buckets:
            d = buckets[b]
            _report(f"{b}", d["t"], d["p"], d["o"], unit, d["top"], args.top_n, d["c"],
                    say=say)

    # CoT 长度分位 + 现有 R_length 阈值是否还适用（只对有 CoT 标签的样本判定）
    all_cots = [c for d in buckets.values() for c in d["c"] if c > 0]
    if all_cots:
        _report_cot_buckets(all_cots, unit, say=say)
    else:
        say("\n⚠️ 全部样本都没有 <think> 标签，跳过 CoT 长度统计。")

    if "domain" in buckets and "anchor" in buckets:
        dd, da = buckets["domain"], buckets["anchor"]
        say(f"\n--- 归因速读 ---")
        say(f"  domain max = {max(dd['t'])} {unit}, anchor max = {max(da['t'])} {unit}")
        if max(da["t"]) > max(dd["t"]):
            say("  → 锚点样本更长: OOM 长尾主要来自 anchor 分账")
        else:
            say("  → 领域样本更长: OOM 长尾主要来自 domain 分账")

    # 全局 Top-N（不分账）
    _report("全量 Top-N（不分 domain/anchor）",
            [t for d in buckets.values() for t in d["t"]],
            [x for d in buckets.values() for x in d["p"]],
            [x for d in buckets.values() for x in d["o"]],
            unit, top_all, args.top_n,
            [c for d in buckets.values() for c in d["c"]], say=say)

    if unit == "char":
        say("\n⚠️ 以上为字符级近似。真 token 口径请在 AutoDL 上带 --tokenizer 重跑。"
            "\n   CoT 分位也依赖真分词 —— 定 R_length 阈值必须用 token 口径。")

    # --out: 机器可读统计落盘 (stdout 是人读版, 这是回填/留存用的结构版)
    if args.out:
        report = {
            "train_file": args.train_file,
            "n_rows": len(rows),
            "tokenizer": args.tokenizer,
            "unit": unit,
            "top_n": args.top_n,
            "buckets": {b: _stats(d["t"], d["p"], d["o"], d["c"])
                        for b, d in buckets.items()},
            "top": [{"idx": i, "bucket": b, "total": n, "hard_case": m.get("hard_case_type"),
                     "source": m.get("source", m.get("provenance"))}
                    for n, i, b, m in top_all[:args.top_n]],
        }
        if all_cots and unit == "token":
            n_all = len(all_cots)
            report["r_length_buckets"] = {
                f"short(<{COT_SHORT})": sum(1 for n in all_cots if n < COT_SHORT),
                f"ideal({COT_SHORT}~{COT_IDEAL_HI})":
                    sum(1 for n in all_cots if COT_SHORT <= n <= COT_IDEAL_HI),
                f"decay({COT_IDEAL_HI}~{COT_DEAD})":
                    sum(1 for n in all_cots if COT_IDEAL_HI < n < COT_DEAD),
                f"dead(>={COT_DEAD})": sum(1 for n in all_cots if n >= COT_DEAD),
                "n": n_all,
            }
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"\n统计已落盘: {args.out}")
    if log_f:
        if args.out:
            log_f.write(f"统计已落盘: {args.out}\n")
        log_f.close()
        print(f"日志已追加: {args.log}")


if __name__ == "__main__":
    main()
