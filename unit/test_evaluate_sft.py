#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测链路单元测试 · evaluate_sft / eval_four_tier / human_recheck_kappa（TODO 5.17）

覆盖范围（对应 TODO 5.11~5.16 各项边界）:
  - schema_contract.parse_output   闭合率 / JSON / 闭集契约（5.11 / 5.18）
  - schema_contract.resolve_gold   金标字段来源统一 + 冲突清单（5.15）
  - eval_metrics 非法标签剔除       （5.12）
  - eval_metrics 分母统一           （5.13）
  - eval_metrics 负向召回 × 误报率成对（5.9）
  - release_gate 四门独立判定       （5.5: 门 1/门 3 缺输入 → N/A 不假装 PASS）
  - confusion_slices 小样本保护     （5.3）

运行:
    python3 -m unittest unit.test_evaluate_sft -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from eval_metrics import (  # noqa: E402
    MIN_BUCKET_N, confusion_slices, illegal_label_counts, macro_f1,
    pa_fp_pair, subset_accuracy,
)
from release_gate import evaluate_gates, overall_verdict  # noqa: E402
from schema_contract import (  # noqa: E402
    parse_output, resolve_gold, validate_schema,
)


def _good_pred(sent="negative", hc="explicit", rel="core",
               risk=("product",), ev="才一天就亏了3000多"):
    return {"subject": "某品牌", "relevance": rel, "sentiment": sent,
            "hard_case_type": hc, "risk_categories": list(risk), "evidence": ev}


OUT_OK = ('<think>Step1 主体某品牌；Step2 判 negative'
          '</think>{"subject":"某品牌","relevance":"core",'
          '"sentiment":"negative","hard_case_type":"explicit",'
          '"risk_categories":["product"],"evidence":"才一天就亏了3000多"}')


# ============================================================
# 5.11 / 5.18 · 契约层判定（训练与评测同一份判定式）
# ============================================================

class TestParseOutput(unittest.TestCase):
    def test_full_valid_output(self):
        cot, pred, flags = parse_output(OUT_OK)
        self.assertTrue(flags["valid_thought_tags"])
        self.assertTrue(flags["valid_json"])
        self.assertIsNone(flags["schema_error"])
        self.assertEqual(pred["sentiment"], "negative")

    def test_missing_thought_tags_flagged(self):
        """不带思考标签 → 闭合率必须判 False（5.18: 闭合率进验收）。"""
        _, _, flags = parse_output('{"sentiment":"negative","relevance":"core",'
                                   '"hard_case_type":"explicit",'
                                   '"risk_categories":[],"evidence":"abcd",'
                                   '"subject":"x"}')
        self.assertFalse(flags["valid_thought_tags"])
        self.assertFalse(flags["valid_json"])   # 缺标签 = 协议失败，不做裸 JSON 降级

    def test_schema_violation_is_invalid_json(self):
        """闭集越界 → 训练侧判 -2.5，评测侧必须同判（5.11）。"""
        _, _, flags = parse_output(
            '<think>x</think>'
            '{"sentiment":"超级负面","relevance":"core","hard_case_type":"explicit",'
            '"risk_categories":[],"evidence":"abcd","subject":"x"}')
        self.assertTrue(flags["valid_json"])          # json.loads 能过
        self.assertIsNotNone(flags["schema_error"])   # 但契约违规
        self.assertTrue(validate_schema.__doc__ is not None)

    def test_nested_object_rejected(self):
        _, _, flags = parse_output(
            '<think>x</think>'
            '{"sentiment":"negative","relevance":"core","hard_case_type":"explicit",'
            '"risk_categories":{"a":1},"evidence":"abcd","subject":"x"}')
        self.assertIsNotNone(flags["schema_error"])


# ============================================================
# 5.15 · 金标字段来源统一 + 冲突清单
# ============================================================

class TestResolveGold(unittest.TestCase):
    def test_output_json_is_authoritative(self):
        """output 与 meta 冲突 → 以 output 为准，且冲突入清单（不静默）。"""
        out = ('<think>x</think>'
               + _good_pred().__repr__().replace("'", '"'))
        row = {"output": ('{"subject":"某品牌","relevance":"core",'
                          '"sentiment":"negative","hard_case_type":"explicit",'
                          '"risk_categories":["product"],"evidence":"abcd"}'),
               "meta": {"sentiment": "positive", "industry": "finance"}}
        gold, conflicts = resolve_gold(row)
        self.assertEqual(gold["sentiment"], "negative")
        self.assertTrue(any(c.startswith("sentiment") for c in conflicts))

    def test_meta_used_as_fallback_only(self):
        row = {"output": "",
               "meta": {"sentiment": "positive", "hard_case_type": "irony",
                        "risk_categories": ["service"]}}
        gold, conflicts = resolve_gold(row)
        self.assertEqual(gold["sentiment"], "positive")
        self.assertEqual(conflicts, [])


# ============================================================
# 5.12 · 非法标签单列并从集合指标剔除
# ============================================================

class TestIllegalLabels(unittest.TestCase):
    def test_illegal_sentiment_excluded_from_macro(self):
        golds = ["negative", "negative"]
        preds = ["超级负面", "negative"]     # 第 1 条闭集外
        stat = macro_f1(golds, preds, ["negative", "positive", "neutral"],
                        {"negative", "positive", "neutral"})
        self.assertEqual(stat["n_skipped_illegal"], 1)
        self.assertEqual(stat["n_scored"], 1)

    def test_illegal_risk_excluded_from_subset(self):
        golds = [["product"], ["service"]]
        preds = [["不存在的风险"], ["service"]]
        stat = subset_accuracy(golds, preds, {"product", "service"})
        self.assertEqual(stat["n_skipped_illegal"], 1)
        self.assertAlmostEqual(stat["subset_accuracy"], 100.0)

    def test_parse_failed_never_scores_empty_gold(self):
        """P0 回归: 解析失败（pred=None）白拿空表分 —— 崩 JSON 不得骗过 Subset。"""
        stat = subset_accuracy([[]], [None])
        self.assertEqual(stat["n_parse_failed"], 1)
        self.assertEqual(stat["n_scored"], 0)

    def test_illegal_counts_report_values(self):
        counts = illegal_label_counts([{"sentiment": "超级负面"},
                                       {"sentiment": "超级负面"},
                                       {"sentiment": "negative"}],
                                      "sentiment", {"negative", "positive", "neutral"})
        self.assertEqual(counts, {"超级负面": 2})


# ============================================================
# 5.13 · 分母统一（总样本数）
# ============================================================

class TestDenominator(unittest.TestCase):
    def test_missing_evidence_counts_as_failure(self):
        """缺 evidence 即不通过（分母=总样本数），另报字段覆盖率。"""
        golds = [["product"], []]
        preds = [["product"], [None]]  # 第二条缺 evidence 由上层计 ev_present
        stat = subset_accuracy(golds, preds)
        # 只验证 subset 口径本身；分母统一在 evaluate_sft.build_protocol_block
        self.assertEqual(stat["n_scored"], 2)

    def test_macro_skips_zero_support(self):
        stat = macro_f1(["negative"] * 3, ["negative"] * 3,
                        ["negative", "positive", "neutral"])
        # positive / neutral support=0 不入宏均，宏均=100
        self.assertAlmostEqual(stat["macro_f1"], 100.0)


# ============================================================
# 5.9 · 负向召回与误报率成对
# ============================================================

class TestPairReporting(unittest.TestCase):
    def test_pair_always_returns_both(self):
        pa = pa_fp_pair(["negative", "positive"],
                        ["negative", "negative"])
        self.assertIn("negative_recall", pa)
        self.assertIn("false_alarm_rate", pa)
        self.assertAlmostEqual(pa["negative_recall"], 100.0)
        # 误报率 = 1 − negative_precision = 1 − 1/2 = 50%
        # （口径: 推给人看的预警里有多少是假的，分母是「判为负面的条数」）
        self.assertAlmostEqual(pa["false_alarm_rate"], 50.0)

    def test_no_negative_gold_recall_undefined(self):
        pa = pa_fp_pair(["positive", "neutral"], ["positive", "neutral"])
        self.assertFalse(pa["recall_defined"])

    def test_no_predicted_negative_far_vacuous(self):
        pa = pa_fp_pair(["negative"], ["positive"])
        self.assertFalse(pa["false_alarm_defined"])
        self.assertAlmostEqual(pa["negative_recall"], 0.0)


# ============================================================
# 5.5 · 四门独立判定: 缺输入 → N/A，不假装 PASS
# ============================================================

class TestGates(unittest.TestCase):
    def _gold_pred(self, n=10):
        golds = [_good_pred() for _ in range(n)]
        preds = [dict(p, valid_json=True) for p in golds]
        return golds, preds

    def test_missing_inputs_report_na_not_pass(self):
        golds, preds = self._gold_pred()
        gates = evaluate_gates(golds, preds)
        self.assertEqual(gates["gate1"], "N/A")   # 无旧错题集
        self.assertEqual(gates["gate3"], "N/A")   # 无 L3 联动
        self.assertIn(gates["gate2"], ("PASS", "FAIL"))  # 门 2 必须实际判定，不许 N/A

    def test_gate4_fails_on_json_break(self):
        golds, preds = self._gold_pred(4)
        preds[0]["valid_json"] = False            # 1/4 崩 → 75% < 99.8%
        gates = evaluate_gates(golds, preds)
        self.assertEqual(gates["gate4"], "FAIL")

    def test_gate1_counts_unfixed_old_errors(self):
        golds, preds = self._gold_pred(4)
        preds[1]["sentiment"] = "positive"        # 把第 2 条判错
        gates = evaluate_gates(golds, preds,
                               old_error_ids=[f"id-{i}" for i in range(4)])
        # 4 条旧错题里 1 条未修复 → 75% < 95%
        self.assertEqual(gates["gate1"], "FAIL")
        self.assertTrue(any(a["gate"] == "gate1" for a in gates["attribution"]))

    def test_gate2_zero_tolerance_on_safety_field(self):
        """金标 10 条全是 negative，模型判错 1 条 → 召回 90%；
        在役基线 100% 时 0 容差必须 FAIL。"""
        golds = [{"sentiment": "negative", "hard_case_type": "irony",
                  "risk_categories": ["product"]} for _ in range(10)]
        preds = [dict(_good_pred(sent="negative", hc="irony"), valid_json=True)
                 for _ in range(10)]
        preds[0]["sentiment"] = "positive"     # 漏报 1 条 → 召回 90%
        gates = evaluate_gates(golds, preds,
                               baseline={"negative_recall": 100.0})
        self.assertEqual(gates["gate2"], "FAIL")
        self.assertTrue(any(a["gate"] == "gate2" for a in gates["attribution"]))

    def test_gate2_none_baseline_value_skips_not_crash(self):
        """P0 回归: baseline 里的 None 值按缺基线跳过记账，不许 float(None) 崩门禁。"""
        golds, preds = self._gold_pred()
        gates = evaluate_gates(golds, preds,
                               baseline={"negative_recall": None,
                                         "macro_f1": None,
                                         "subset_accuracy": None})
        self.assertEqual(gates["gate2"], "PASS")
        self.assertGreater(len(gates["details"]["gate2"]["skipped"]), 0)

    def test_overall_verdict_fails_on_na_free_fail(self):
        self.assertEqual(overall_verdict({"gate1": "PASS", "gate2": "FAIL",
                                          "gate3": "N/A", "gate4": "PASS"}), "FAIL")


# ============================================================
# 5.3 · 四维切片 + 小样本保护
# ============================================================

class TestSlices(unittest.TestCase):
    def test_small_bucket_marked_insufficient(self):
        dims = {"industry": ["finance"] * (MIN_BUCKET_N - 1) + ["tech"] * 8}
        preds = [_good_pred() for _ in range(MIN_BUCKET_N - 1 + 8)]
        gold = {"sentiment": ["negative"] * (MIN_BUCKET_N - 1 + 8),
                "hard_case_type": ["explicit"] * (MIN_BUCKET_N - 1 + 8),
                "risk_categories": [["product"]] * (MIN_BUCKET_N - 1 + 8)}
        out = confusion_slices(dims, preds, ["industry"], gold=gold)
        self.assertTrue(out["industry"]["finance"]["insufficient"])
        self.assertFalse(out["industry"]["tech"]["insufficient"])
        self.assertAlmostEqual(out["industry"]["tech"]["risk_exact"], 100.0)

    def test_missing_dim_value_goes_to_unknown(self):
        dims = {"industry": [None]}
        out = confusion_slices(dims, [_good_pred()], ["industry"])
        self.assertIn("unknown", out["industry"])


# ============================================================
# 模式 A · Anchor 分账（锚点不走舆情四查，只查行级结构）
# ============================================================

class TestFormatDriftGate4(unittest.TestCase):
    """格式漂移回归（探针 run_20261028_102858）：模型输出漂移成
    <tool_call> 包 JSON（缺 <think>），宽松提取能捞到字段算指标，
    但 valid_json 协议口径不得放宽——JSON 合法率 0%、gate4 FAIL。
    """

    def _write_jsonl(self, rows):
        import json as _json
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def test_drifted_text_counts_json_invalid(self):
        """文本行强解析失败 → json_valid 记 0%，即使宽松提取捞到了字段。"""
        from evaluate_sft import parse_model_output
        drifted = ('<tool_call> {"subject":"资生堂红腰子","relevance":"core",'
                   '"sentiment":"positive","hard_case_type":"fact_implicit",'
                   '"risk_categories":["product"],"evidence":"让人过敏"}')
        res = parse_model_output(drifted)
        self.assertFalse(res["valid_json"])   # 协议口径：缺 <think> = 不合法
        self.assertIsNotNone(res["pred"])     # 宽松提取仍供字段对比

    def test_gate4_fails_on_drifted_output(self):
        """漂移输出过 gate4 必须 FAIL（宽松提取不许洗白格式门槛）。"""
        gold = _good_pred(sent="negative", hc="explicit")
        gold["id"] = "#1"
        pred = dict(gold)
        pred["valid_json"] = False   # 按强解析口径记 False
        pred["valid_tags"] = False
        gates = evaluate_gates([gold], [pred],
                               thresholds={"json_valid": 99.8})
        self.assertEqual(gates["gate4"], "FAIL")


class TestModeAAnchorPartition(unittest.TestCase):
    """模式 A 单文件质检：Anchor 样本分账跳过协议四查（run_single_file_check）。"""

    def _write_jsonl(self, rows):
        import json as _json
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def _domain_row(self, i=0):
        return {"instruction": f"判读第{i}条舆情", "input": "原文提到才一天就亏了3000多",
                "output": OUT_OK,
                "meta": {"provenance": "synthetic", "sentiment": "negative",
                         "hard_case_type": "explicit", "industry": "finance_insurance"}}

    def _anchor_row(self, i=0):
        return {"instruction": f"解释CAP定理第{i}题", "input": "",
                "output": "CAP 定理指出一致性、可用性与分区容错三者不可兼得。",
                "meta": {"source": "COIG-CQIA", "provenance": "public_anchor",
                         "hard_case_type": "general_anchor"}}

    def test_anchor_skipped_domain_four_check_passes(self):
        """混合集(领域+锚点)：锚点分账跳过四查，领域分账正常 PASS。"""
        from evaluate_sft import run_single_file_check
        path = self._write_jsonl([self._domain_row(i) for i in range(3)] +
                                 [self._anchor_row(i) for i in range(5)])
        ok = run_single_file_check(path)
        self.assertTrue(ok)

    def test_anchor_structural_bad_fails(self):
        """锚点行缺 output → 行级结构 FAIL（这是锚点唯一会被判的项）。"""
        from evaluate_sft import run_single_file_check
        bad = self._anchor_row()
        bad["output"] = ""
        path = self._write_jsonl([self._domain_row(), bad])
        ok = run_single_file_check(path)
        self.assertFalse(ok)

    def test_domain_rows_only_no_anchor_block(self):
        """纯领域数据：n_anchor=0，四查照常执行。"""
        from evaluate_sft import run_single_file_check
        path = self._write_jsonl([self._domain_row(i) for i in range(4)])
        ok = run_single_file_check(path)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
