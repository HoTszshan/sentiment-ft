#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工复标 κ 验收 · 纯逻辑单测 (零外部依赖, 不需要 GPU / 网络)

覆盖: _field_pairs 对齐与跳过 / build_report 逐字段 κ 与一致率 /
      样本不足记账 (不假装通过) / verdict 三档门槛与跳过字段点名

口径来源: notes/舆情负面精判_评估手册_SFT_GRPO.md 模块一①「人工复标」
          (κ ≥ 0.74 放行 / [0.6,0.74) BOUNDARY / <0.6 FAIL)

运行:
    python3 -m unittest unit.test_human_recheck_kappa -v
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import human_recheck_kappa as hrk   # noqa: E402  (CLI 模块, 仅取纯函数)


def row(sent=None, rel=None, hc=None, risk_h=None, risk_m=None, **over):
    """构造一条复标记录 (human 为参照侧)。传 None 表示该侧缺该字段。"""
    human = {}
    model = {}
    if sent is not None:
        human["sentiment"], model["sentiment"] = sent
    if rel is not None:
        human["relevance"], model["relevance"] = rel
    if hc is not None:
        human["hard_case_type"], model["hard_case_type"] = hc
    if risk_h is not None:
        human["risk_categories"] = risk_h
    if risk_m is not None:
        model["risk_categories"] = risk_m
    r = {"id": over.pop("rid", "x"), "human": human, "model": model}
    r.update(over)
    return r


_PARAM_OF_FIELD = {"sentiment": "sent", "relevance": "rel",
                   "hard_case_type": "hc"}


def balanced_rows(n_pairs, mismatches, field="sentiment"):
    """
    构造 n_pairs 条、其中 mismatches 条不一致的样本，且 human/model 边际分布
    均为 n/2 : n/2 —— 此时 p_e = 0.5，κ = (po - 0.5) / 0.5 = 2*po - 1。
    """
    rows = []
    for i in range(n_pairs):
        h = "negative" if i % 2 == 0 else "positive"
        if i < mismatches:                       # 前 mismatches 条判错
            m = "positive" if h == "negative" else "negative"
        else:
            m = h
        kw = {_PARAM_OF_FIELD[field]: (h, m)}
        rows.append(row(**kw, rel=("core", "core"),
                        hc=("explicit", "explicit"),
                        risk_h=["product"], risk_m=["product"]))
    return rows


class TestFieldPairs(unittest.TestCase):
    def test_missing_side_skipped(self):
        """human 侧缺字段 → 该条不参与对齐 (不计错, 也不虚增分母)。"""
        rows = [row(rel=("core", "core")), row(model={"sentiment": "negative"})]
        pairs = hrk._field_pairs(rows, "relevance")
        self.assertEqual(pairs, [("core", "core")])

    def test_risk_uses_sorted_tuple(self):
        """risk 是多标签，必须展开成集合比对 (顺序无关)。"""
        rows = [row(risk_h=["marketing", "product"], risk_m=["product", "marketing"])]
        pairs = hrk._field_pairs(rows, "risk")
        self.assertEqual(pairs, [(("marketing", "product"), ("marketing", "product"))])

    def test_empty_rows(self):
        self.assertEqual(hrk._field_pairs([], "sentiment"), [])


class TestBuildReport(unittest.TestCase):
    def test_perfect_agreement_kappa_one(self):
        """po=1, pe=0.5 → κ=(1-0.5)/0.5=1.0"""
        rep = hrk.build_report(balanced_rows(10, 0))
        self.assertEqual(rep["sentiment"]["kappa"], 1.0)
        self.assertEqual(rep["sentiment"]["agree"], 10)
        self.assertEqual(rep["sentiment"]["rate"], 1.0)
        self.assertEqual(rep["n_input"], 10)

    def test_hand_computed_boundary_kappa(self):
        """10 条错 2 条: po=0.8, pe=0.5 → κ=0.6 (手算值, 非对拍同一实现)。"""
        rep = hrk.build_report(balanced_rows(10, 2))
        self.assertEqual(rep["sentiment"]["kappa"], 0.6)
        self.assertEqual(rep["sentiment"]["agree"], 8)

    def test_all_disagree_negative_kappa(self):
        """po=0 → κ=-1.0，必须能被 verdict 判 FAIL。"""
        rep = hrk.build_report(balanced_rows(10, 10))
        self.assertEqual(rep["sentiment"]["kappa"], -1.0)

    def test_insufficient_sample_records_not_fakes(self):
        """n < MIN_KAPPA_N → κ=None 且写明 skipped_reason，不许假装通过。"""
        rows = balanced_rows(3, 0)
        rep = hrk.build_report(rows)
        e = rep["sentiment"]
        self.assertIsNone(e["kappa"])
        self.assertEqual(e["n"], 3)
        self.assertIn("skipped_reason", e)
        self.assertIn(str(hrk.MIN_KAPPA_N), e["skipped_reason"])
        self.assertEqual(e["rate"], 1.0)      # 一致率仍可报 (n>0)

    def test_zero_sample_rate_is_none(self):
        rep = hrk.build_report([])
        self.assertIsNone(rep["sentiment"]["rate"])
        self.assertIsNone(rep["sentiment"]["kappa"])

    def test_all_four_fields_present(self):
        rep = hrk.build_report(balanced_rows(6, 0))
        for f in ("sentiment", "relevance", "hard_case_type", "risk"):
            self.assertIn(f, rep)


class TestVerdict(unittest.TestCase):
    def test_pass_at_threshold(self):
        self.assertTrue(hrk.verdict(hrk.build_report(balanced_rows(10, 0)))
                        .startswith("PASS"))

    def test_boundary_inclusive_lower_edge(self):
        """κ=0.60 恰在 BOUNDARY 下沿 (区间 [0.6, 0.74) 左闭)。"""
        v = hrk.verdict(hrk.build_report(balanced_rows(10, 2)))
        self.assertTrue(v.startswith("BOUNDARY"), v)

    def test_fail_below_boundary(self):
        """10 条错 4 条: po=0.6 → κ=0.2 < 0.6 → FAIL 退回重打标。"""
        v = hrk.verdict(hrk.build_report(balanced_rows(10, 4)))
        self.assertTrue(v.startswith("FAIL"), v)
        self.assertIn("退回重打标", v)

    def test_verdict_takes_worst_field(self):
        """三档取「最低 κ 字段」定档 —— sentiment 满分但 hard_case 崩 → 整体不 PASS。"""
        rows = balanced_rows(10, 0)
        for i, r in enumerate(rows[:4]):                     # 打崩 hard_case
            r["model"]["hard_case_type"] = "irony"
            r["human"]["hard_case_type"] = "explicit"
        v = hrk.verdict(hrk.build_report(rows))
        self.assertIn("hard_case_type", v)
        self.assertFalse(v.startswith("PASS"), v)

    def test_skipped_field_is_named(self):
        """被跳过的字段必须在判定串里点名，不许静默通过后假装全字段达标。"""
        rows = balanced_rows(10, 0)
        for r in rows[:7]:                                   # risk 侧只留 3 条有效
            r["human"].pop("risk_categories", None)
        v = hrk.verdict(hrk.build_report(rows))
        self.assertIn("跳过", v)
        self.assertIn("risk", v)

    def test_nothing_judgeable_is_fail(self):
        """全字段样本不足 → FAIL (不是 PASS、不是 N/A)。"""
        v = hrk.verdict(hrk.build_report(balanced_rows(2, 0)))
        self.assertTrue(v.startswith("FAIL"), v)
        self.assertIn("无任何字段可判", v)


class TestThresholds(unittest.TestCase):
    def test_thresholds_match_manual(self):
        """门槛常量必须与评估手册模块一① 一致 (0.74 放行 / 0.60 下沿)。"""
        self.assertEqual(hrk.THRESH_HUMAN_KAPPA, 0.74)
        self.assertEqual(hrk.THRESH_BOUNDARY_KAPPA, 0.60)
        self.assertLess(hrk.THRESH_BOUNDARY_KAPPA, hrk.THRESH_HUMAN_KAPPA)

    def test_min_kappa_n_matches_cohen_kappa_floor(self):
        """MIN_KAPPA_N 必须与 kappa_utils.cohen_kappa 的 n<5 同源，否则会出现
        「report 说可判、κ 返回 None」的自相矛盾。
        注意: p_e=1（全同值对）时 cohen_kappa 仍返回 None，这是 κ 公式本身的
        退化情形，与样本量无关 —— 因此用有分歧的对来验证 n=5 可判。"""
        from kappa_utils import cohen_kappa
        self.assertEqual(hrk.MIN_KAPPA_N, 5)
        self.assertIsNone(cohen_kappa([("a", "a")] * 4))       # 样本不足
        pairs = [("negative", "negative")] * 4 + [("negative", "positive")]
        self.assertIsNotNone(cohen_kappa(pairs))               # n=5 且有分歧 → 可判


if __name__ == "__main__":
    unittest.main()
