#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务终验单元测试 · business_verify（TODO 5.6）

覆盖:
  - 漏报核查四象限（净增漏报硬线 / 存量漏报另计 / 新捞回 / 双命中）
  - 双人盲标不一致 → BLOCKED（不仲裁、不假装 PASS）
  - 工时核查同源三约束（缺版本 / 时间窗不配对 / 入文量不同 → 该客户剔除并记账）
  - 工时降幅计算（含 human_reviewed_old=0 的除零保护）
  - 事后回流（escalate 确认漏报 / revoke 单列不混入）

运行:
    python3 -m unittest unit.test_business_verify -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from business_verify import (  # noqa: E402
    feedback_check,
    missed_alarm_check,
    workload_check,
)


def A(sid, neg=True, alarm=True, la=True, lb=True):
    r = {"sample_id": sid, "confirmed_negative": neg, "alarmed": alarm}
    if la is not None:
        r["labeler_a"] = la
    if lb is not None:
        r["labeler_b"] = lb
    return r


class TestMissedAlarm(unittest.TestCase):
    def test_net_new_missed_must_be_zero(self):
        """旧版告警、新版未告警、确认负面 → 净增漏报 → FAIL。"""
        new = [A("m1"), A("m2", alarm=False)]   # m2: old 告警 new 未告警
        old = [A("m1"), A("m2")]
        r = missed_alarm_check(new, old)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["net_new_missed"]["count"], 1)
        self.assertIn("m2", r["net_new_missed"]["ids"])

    def test_legacy_missed_not_attributed(self):
        """两版都没告警 → 存量漏报另计，不触发 FAIL。"""
        new = [A("x1", alarm=False)]
        old = [A("x1", alarm=False)]
        r = missed_alarm_check(new, old)
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["net_new_missed"]["count"], 0)
        self.assertEqual(r["legacy_missed_old_model"]["count"], 1)

    def test_rescued_counted(self):
        """新版告警、旧版未告警 → 新捞回记账。"""
        new = [A("r1")]
        old = [A("r1", alarm=False)]
        r = missed_alarm_check(new, old)
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["rescued_by_new"]["count"], 1)

    def test_labeler_conflict_blocks(self):
        """双人盲标不一致 → BLOCKED，不仲裁不判定。"""
        new = [A("a1", la=True, lb=False)]
        old = [A("a1")]
        r = missed_alarm_check(new, old)
        self.assertEqual(r["status"], "BLOCKED")

    def test_abs_miss_rate_denominator(self):
        """绝对漏报率分母 = 确认负面的抽样样本。"""
        new = [A("m1"), A("m2", alarm=False), A("m3", alarm=False)]
        old = [A("m1"), A("m2"), A("m3", alarm=False)]
        r = missed_alarm_check(new, old)
        # 确认负面 3 条；新版漏 m2（净增）+ m3（存量）= 2 → 66.67%
        self.assertAlmostEqual(r["absolute_miss_rate_new_pct"], 66.67, places=1)

    def test_non_negative_excluded(self):
        """confirmed_negative=false 的样本不进分母。"""
        new = [A("m1", neg=False, alarm=False)]
        old = [A("m1", neg=False, alarm=False)]
        r = missed_alarm_check(new, old)
        self.assertEqual(r["n_confirmed_negative"], 0)
        self.assertIsNone(r["absolute_miss_rate_new_pct"])


class TestWorkload(unittest.TestCase):
    def test_valid_pair_reduces(self):
        wl = [
            {"customer_id": "c1", "period": "2025-03", "version": "new",
             "articles_ingested": 30000, "human_reviewed": 45},
            {"customer_id": "c1", "period": "2025-03", "version": "old",
             "articles_ingested": 30000, "human_reviewed": 225},
        ]
        r = workload_check(wl)
        self.assertEqual(r["status"], "OK")
        self.assertAlmostEqual(r["totals"]["overall_reduction_pct"], 80.0)
        self.assertAlmostEqual(r["per_customer"][0]["reduction_pct"], 80.0)

    def test_missing_version_dropped(self):
        wl = [{"customer_id": "c1", "period": "2025-03", "version": "new",
               "articles_ingested": 100, "human_reviewed": 5}]
        r = workload_check(wl)
        self.assertEqual(r["n_customers_dropped"], 1)
        self.assertEqual(r["status"], "NO_VALID_DATA")

    def test_ingest_mismatch_dropped(self):
        """入文量不同源 → 该客户整体剔除（分子分母必须同源）。"""
        wl = [
            {"customer_id": "c1", "period": "2025-03", "version": "new",
             "articles_ingested": 30000, "human_reviewed": 45},
            {"customer_id": "c1", "period": "2025-03", "version": "old",
             "articles_ingested": 25000, "human_reviewed": 200},
        ]
        r = workload_check(wl)
        self.assertEqual(r["n_customers_valid"], 0)
        self.assertIn("入文量不同源", r["dropped"][0]["reason"])

    def test_period_mismatch_dropped(self):
        wl = [
            {"customer_id": "c1", "period": "2025-03", "version": "new",
             "articles_ingested": 100, "human_reviewed": 5},
            {"customer_id": "c1", "period": "2025-04", "version": "old",
             "articles_ingested": 100, "human_reviewed": 9},
        ]
        r = workload_check(wl)
        self.assertEqual(r["n_customers_valid"], 0)

    def test_zero_old_reviewed_no_crash(self):
        """old 版复核 0 条 → 降幅除零保护，不崩。"""
        wl = [
            {"customer_id": "c1", "period": "2025-03", "version": "new",
             "articles_ingested": 100, "human_reviewed": 3},
            {"customer_id": "c1", "period": "2025-03", "version": "old",
             "articles_ingested": 100, "human_reviewed": 0},
        ]
        r = workload_check(wl)
        self.assertIsNone(r["per_customer"][0]["reduction_pct"])


class TestFeedback(unittest.TestCase):
    def test_escalate_confirmed_counted(self):
        fb = [{"sample_id": "x1", "action": "escalate", "confirmed_negative": True},
              {"sample_id": "x2", "action": "escalate", "confirmed_negative": False},
              {"sample_id": "x3", "action": "revoke", "confirmed_negative": False}]
        r = feedback_check(fb)
        self.assertEqual(r["confirmed_missed_after_release"]["count"], 1)
        self.assertEqual(r["revoked_false_alarms"]["count"], 1)

    def test_empty_feedback(self):
        self.assertEqual(feedback_check([])["status"], "NO_DATA")


if __name__ == "__main__":
    unittest.main()
