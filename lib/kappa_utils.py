#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一致性统计纯函数 · 单一判定源

原内嵌于 build_eval_set.py（已归档至 scripts/archive/），因 human_recheck_kappa.py
等活代码仍需复用 κ 计算，下沉到 lib（谁都不许再写一份 κ）。
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

JUDGE_FIELDS = ["sentiment", "relevance", "hard_case_type"]


def risk_key(r: Any) -> Tuple[str, ...]:
    return tuple(sorted(r if isinstance(r, (list, tuple)) else []))


def pair_agreement(pairs: List[Tuple[Any, Any]]) -> Tuple[int, int, float]:
    """两两一致的条数/总数/一致率。"""
    if not pairs:
        return 0, 0, 0.0
    ok = sum(1 for x, y in pairs if x == y)
    return ok, len(pairs), ok / len(pairs)


def cohen_kappa(pairs: List[Tuple[Any, Any]]) -> Optional[float]:
    """Cohen's Kappa。样本过少或 p_e=1 时返回 None。"""
    n = len(pairs)
    if n < 5:
        return None
    po, _, _ = pair_agreement(pairs)
    po /= n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    if abs(1 - pe) < 1e-9:
        return None
    return round((po - pe) / (1 - pe), 4)
