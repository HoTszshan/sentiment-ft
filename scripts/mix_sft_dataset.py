#!/usr/bin/env python3
"""
scripts/mix_sft_dataset.py
将舆情领域合成数据 (Domain SFT) 与公开通用数据 (Anchor) 按 9:1 严格配比混合。
自动完成：格式对齐、洗牌打散 (Shuffle)、训练集与验证集切分 (Train/Val Split)。

两账分离铁律 (DataPipeline_SOP §3.3)：
  - 领域数据走分层切分 (--stratified)，保证 val 里正/负/中性与各难点桶都有代表；
  - Anchor 通用数据**全部进 train，不进 val** —— 防遗忘样本不参与舆情早停判据，
    也不与领域数据混账算 loss。

⚠️ val 仅作「训练期早停/调 FIELD_W」的仪表盘，**不是成绩单**；
   简历合法数字必须来自 500 条纯人标金标测试集（plan/SFT负面精判_技术评审.md 锁死）。
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _strat_key(item):
    """分层键：难点类型 + 情感极性。Anchor 数据无舆情语义，单独归为 general_anchor 桶。"""
    meta = item.get("meta") or {}
    return (meta.get("hard_case_type", "unknown"), meta.get("sentiment", "unknown"))


def stratified_split(domain_data, val_ratio, rng):
    """
    按 (hard_case_type, sentiment) 分层切 val。
    每桶至少保证 1 条进 val（桶 ≥2 时），保证小桶不被整体漏掉；
    总量按 val_ratio 逼近。
    """
    buckets = defaultdict(list)
    for item in domain_data:
        buckets[_strat_key(item)].append(item)

    for k in buckets:
        rng.shuffle(buckets[k])

    n_domain = len(domain_data)
    n_val_target = max(1, round(n_domain * val_ratio))

    val, train = [], []
    # 第一遍：每个 >=2 条的桶先贡献 1 条，保证覆盖面
    for k in sorted(buckets, key=str):
        if len(buckets[k]) >= 2:
            val.append(buckets[k].pop())
    # 第二遍：若还不够，从最大的桶继续按比例补
    while len(val) < n_val_target:
        biggest = max(buckets, key=lambda k: len(buckets[k]))
        if not buckets[biggest]:
            break
        val.append(buckets[biggest].pop())
    # 若超额，从 val 里按桶还剩多的还回去，优先保证 train 体量
    for k in sorted(buckets, key=str):
        while len(val) > n_val_target and any(_strat_key(v) == k for v in val):
            for i, v in enumerate(val):
                if _strat_key(v) == k:
                    buckets[k].append(val.pop(i))
                    break
            else:
                break

    for k in buckets:
        train.extend(buckets[k])

    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def _prov(item):
    return (item.get("meta") or {}).get("provenance", "unknown")


def _is_anchor(item):
    """公开 Anchor 无舆情语义 (hard_case_type=general_anchor / provenance=public_anchor)。"""
    meta = item.get("meta") or {}
    return meta.get("provenance") == "public_anchor" or meta.get("hard_case_type") == "general_anchor"


def print_split_stats(tag: str, rows):
    """分账统计：血缘 / 难点 / 情感 / 语篇 / 行业，逐维度打印 Train 与 Val 各自的分布。"""
    n = len(rows)
    print(f"\n--- {tag} 分布统计 ({n} 条) ---")
    if n == 0:
        print("  (空)")
        return

    def counter(fn):
        c = defaultdict(int)
        for it in rows:
            c[fn(it)] += 1
        return dict(sorted(c.items(), key=lambda x: (-x[1], str(x[0]))))

    def fmt(c):
        return "  ".join(f"{k}:{v}({v/n*100:.0f}%)" for k, v in c.items())

    # 血缘
    print(f"  血缘   : {fmt(counter(_prov))}")
    # 锚定行统一标为「公开通用」，避免和舆情字段混读
    def hc_key(d):
        if _is_anchor(d):
            return "公开通用(非舆情)"
        return (d.get("meta") or {}).get("hard_case_type", "?")

    def anchor_or(key):
        return lambda d: "—" if _is_anchor(d) else (d.get("meta") or {}).get(key, "?")

    print(f"  难点   : {fmt(counter(hc_key))}")
    print(f"  情感   : {fmt(counter(anchor_or('sentiment')))}")
    print(f"  语篇   : {fmt(counter(anchor_or('relevance')))}")
    print(f"  行业   : {fmt(counter(anchor_or('industry')))}")


def mix_datasets(domain_file: str, anchor_file: str, out_dir: str,
                 anchor_ratio: float = 0.1, val_ratio: float = 0.1,
                 seed: int = 42, stratified: bool = True,
                 anchor_max_chars: int = 1200):
    rng = random.Random(seed)
    domain_path = Path(domain_file)
    anchor_path = Path(anchor_file)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not domain_path.exists():
        raise FileNotFoundError(f"找不到领域训练数据: {domain_path}")

    domain_data = [json.loads(l) for l in open(domain_path, encoding="utf-8") if l.strip()]
    n_domain = len(domain_data)

    # Anchor 数量：按 9:1 配比 (anchor 占总量 10%)
    n_anchor = int(n_domain * anchor_ratio / (1.0 - anchor_ratio))

    anchor_pool = []
    if anchor_path.exists():
        raw = [json.loads(l) for l in open(anchor_path, encoding="utf-8") if l.strip()]
        # ---- 长度过滤（混合前）: 锚点是通用指令文本, 无 CoT+JSON 协议长度约束,
        # COIG-CQIA 里混有长文 (实测单条 prompt 段可达 1828 token, 为 domain max 719 的 2.6 倍),
        # 是 OOM 长尾的主要来源。按字符数过滤 (1 汉字≈1.4 token, 1200 字符 ≈ 1680 token 上限;
        # 默认 1200 字符下 95%+ 正常锚点保留, 超长维基类长文被剔除)。
        n_raw = len(raw)
        filtered = [r for r in raw
                    if len(r.get("instruction", "") or "") + len(r.get("input", "") or "")
                    + len(r.get("output", "") or "") <= anchor_max_chars]
        n_dropped = n_raw - len(filtered)
        raw = filtered
        if n_dropped:
            print(f"⚠️ Anchor 长度过滤: 剔除 {n_dropped}/{n_raw} 条 (拼接超 {anchor_max_chars} 字符, "
                  f"防 OOM 长尾; 可用 --anchor_max_chars 调整)")
        if not raw:
            raise SystemExit(f"[FATAL] 锚点数据经长度过滤后为空 (max_chars={anchor_max_chars}), "
                             f"请检查 anchor 文件或放宽 --anchor_max_chars")
        rng.shuffle(raw)
        anchor_pool = raw[:n_anchor]
        # 不足则循环补齐
        i = 0
        while len(anchor_pool) < n_anchor and raw:
            anchor_pool.append(raw[i % len(raw)])
            i += 1
    else:
        print(f"⚠️ 未找到 {anchor_path}（先跑 scripts/download_assets.py --download_anchor）；"
              f"本次全量使用领域数据，0 Anchor。")

    total = n_domain + len(anchor_pool)
    print(f"\n=======================================================")
    print(f"【数据集混合方案 (SOP §1.0 严格 9:1 黄金配比)】")
    print(f"  - 领域数据 (Domain) : {n_domain} 条 ({n_domain/total*100:.1f}%)")
    print(f"  - 通用防遗忘 (Anchor): {len(anchor_pool)} 条 ({len(anchor_pool)/total*100:.1f}%)")
    print(f"  - 合计总库          : {total} 条")
    print(f"=======================================================")

    if stratified:
        train_domain, val_domain = stratified_split(domain_data, val_ratio, rng)
        mode = "分层切分 (hard_case_type × sentiment)"
    else:
        shuffled = list(domain_data)
        rng.shuffle(shuffled)
        n_val = max(1, round(n_domain * val_ratio))
        val_domain, train_domain = shuffled[:n_val], shuffled[n_val:]
        mode = "随机洗牌切分"

    # Anchor 全部进 train (不进 val)
    train_set = train_domain + anchor_pool
    val_set = val_domain
    rng.shuffle(train_set)
    rng.shuffle(val_set)

    train_file = out_path / "sft_train.jsonl"
    val_file = out_path / "sft_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✓ 混合切分完成（{mode}）：")
    print(f"  - 训练集 (Train): {len(train_set)} 条 -> {train_file}")
    print(f"  - 验证集 (Val)  : {len(val_set)} 条 -> {val_file}")

    print_split_stats("Train", train_set)
    print_split_stats("Val", val_set)

    # 分账核对：Anchor 必须只在 Train，Val 里出现即为 bug
    n_anchor_train = sum(1 for i in train_set if _prov(i) == "public_anchor")
    n_anchor_val = sum(1 for i in val_set if _prov(i) == "public_anchor")
    flag = "OK" if n_anchor_val == 0 else "!! 异常：Val 出现了 Anchor！"
    print(f"\n--- 分账核对 (Anchor 只能进 Train) ---")
    print(f"  Anchor 在 Train: {n_anchor_train} 条")
    print(f"  Anchor 在 Val  : {n_anchor_val} 条   [{flag}]")
    if n_anchor_train:
        ratio = n_anchor_train / max(len(train_set), 1) * 100
        print(f"  Anchor 占 Train: {ratio:.1f}%   (SOP 目标 ~10%，即 9:1)")

    print(f"\n⚠️ 用途提示: val 仅供训练期早停 / FIELD_W 调参，**不是成绩单**。"
          f"简历合法数字须来自 500 条人标金标测试集。")


def main():
    p = argparse.ArgumentParser(description="领域数据与公开 Anchor 数据混合脚本")
    p.add_argument("--domain_file", default="data/train_synth_aligned.jsonl")
    p.add_argument("--anchor_file", default="data/coig_cqia_anchor.jsonl")
    p.add_argument("--out_dir", default="data/mixed")
    p.add_argument("--anchor_ratio", type=float, default=0.1, help="Anchor 占比 (默认 10%，即 9:1)")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Val 占比 (默认 10%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchor_max_chars", type=int, default=1200,
                   help="锚点单条 (instruction+input+output) 最大字符数, 超长剔除防 OOM (默认 1200≈1680 token)")
    p.add_argument("--random_split", action="store_true",
                   help="关掉分层，退化为随机洗牌切分")
    args = p.parse_args()

    mix_datasets(args.domain_file, args.anchor_file, args.out_dir,
                 anchor_ratio=args.anchor_ratio, val_ratio=args.val_ratio,
                 seed=args.seed, stratified=not args.random_split,
                 anchor_max_chars=args.anchor_max_chars)


if __name__ == "__main__":
    main()
