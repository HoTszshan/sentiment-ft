#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 训练链路 · 离线单测 (不联网、不加载模型、不 import torch/trl)

覆盖三件:
  1. grpo_common    数据契约展平/还原、k 分桶、训练预算、整除校验、ZAM 等价性
  2. grpo_probe_core 四维打分入组、采样器注入式 rollout、completion 形态兼容
  3. train_grpo     自定义 flag 拦截、reward 列值对齐 (TRL 三种布局)、版本兼容装配

运行:
    python3 -m unittest unit.test_grpo_probe -v
    或在 MVP/tests 目录: python3 -m unittest unit.test_grpo_probe -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from grpo_common import (  # noqa: E402
    GrpoHyper, bucket_label, build_dataset_rows, flatten_gold, group_totals,
    k_of, rebuild_gold, score_one, summarize_buckets, assert_zam_equivalence,
    budget_note, validate_generation_shape,
    should_fuse,
)
from grpo_probe_core import (  # noqa: E402
    extract_completion_text, oracle_sampler, run_probe, score_group,
)
import train_grpo as tg  # noqa: E402


# ---------------------------------------------------------------
# 样本
# ---------------------------------------------------------------

INPUT_TEXT = "今天去广发证券体验了一把所谓的专业理财，呵呵，真的是“稳赚不赔”啊！才一天就亏了3000多。"

PROMPT = ("<|im_start|>system\n你是舆情负面精判模型。目标主体：【广发证券】<|im_end|>\n"
          "<|im_start|>user\n" + INPUT_TEXT + "<|im_end|>\n<|im_start|>assistant\n")

GOLD = {
    "sentiment": "negative",
    "relevance": "core",
    "hard_case_type": "irony",
    "risk_categories": ["product", "marketing"],
    "evidence": "才一天就亏了3000多",
}


def _mk(sentiment="negative", hard="irony", risk=None, evidence=None, rel="core",
        cot=None) -> str:
    import json as _j
    body = {"subject": "广发证券", "relevance": rel, "sentiment": sentiment,
            "hard_case_type": hard,
            "risk_categories": GOLD["risk_categories"] if risk is None else risk,
            "evidence": GOLD["evidence"] if evidence is None else evidence}
    cot = cot or ("Step1 主体居叙事中心故core。Step2 字面语义与真实取向相反判irony。"
                  "Step3 风险归product与marketing，负面定negative。")
    return f"<think>{cot}</think>" + _j.dumps(body, ensure_ascii=False)


GOOD = _mk()
BROKEN = "<think>cot</think>{broken"


# ============================================================
# 1. grpo_common · 数据契约
# ============================================================

class TestGoldContract(unittest.TestCase):
    def test_round_trip(self):
        flat = flatten_gold(GOLD)
        self.assertEqual(flat["gold_sentiment"], "negative")
        self.assertEqual(flat["gold_risk_json"], '["product", "marketing"]')
        back = rebuild_gold(flat)
        self.assertEqual(back, GOLD)

    def test_empty_risk_round_trip(self):
        flat = flatten_gold(dict(GOLD, risk_categories=[]))
        self.assertEqual(rebuild_gold(flat)["risk_categories"], [])

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            flatten_gold({"sentiment": "negative"})

    def test_not_dict_raises(self):
        with self.assertRaises(ValueError):
            flatten_gold("不是 dict")

    def test_build_dataset_rows_columns(self):
        rows = build_dataset_rows([{"prompt": PROMPT, "gold": GOLD},
                                   {"prompt": PROMPT, "gold": GOLD}])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_text"], INPUT_TEXT)   # 从 prompt 反解，不重复落原文
        for c in ("gold_sentiment", "gold_relevance", "gold_hard_case_type",
                  "gold_risk_json", "gold_evidence"):
            self.assertIn(c, rows[0])

    def test_build_dataset_rows_missing_prompt_raises(self):
        with self.assertRaises(ValueError):
            build_dataset_rows([{"gold": GOLD}])


# ============================================================
# 2. grpo_common · k 分桶与预算
# ============================================================

class TestBucketing(unittest.TestCase):
    def test_group_totals_chunking(self):
        self.assertEqual(group_totals([1, 2, 3, 4], 2), [[1, 2], [3, 4]])
        self.assertEqual(group_totals([], 4), [])

    def test_group_totals_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            group_totals([1, 2, 3], 4)
        with self.assertRaises(ValueError):
            group_totals([1, 2, 3, 4], 0)

    def test_k_counts_accuracy_pass_only(self):
        self.assertEqual(k_of([{"accuracy_pass": True}, {"accuracy_pass": False},
                               {"accuracy_pass": True}, {}]), 2)
        # 门禁短路的 detail 里没有 accuracy_pass → 记 0
        self.assertEqual(k_of([{"short_circuit": "format"}]), 0)

    def test_bucket_labels(self):
        self.assertEqual(bucket_label(0), "drop_k0")
        self.assertEqual(bucket_label(1), "keep")
        self.assertEqual(bucket_label(3), "keep")
        self.assertEqual(bucket_label(4), "drop_k_max")
        # 留池的只有 k∈{1,2,3}，G 变大时中间档一律丢（手册 §四 原文口径）
        self.assertEqual(bucket_label(5, g=6), "drop")
        self.assertEqual(bucket_label(6, g=6), "drop_k_max")

    def test_summarize_buckets(self):
        rows = [{"k": 0, "hard_case_type": "irony", "sentiment": "negative"},
                {"k": 1, "hard_case_type": "irony", "sentiment": "negative"},
                {"k": 2, "hard_case_type": "metaphor", "sentiment": "positive"},
                {"k": 4, "hard_case_type": "irony", "sentiment": "negative"}]
        s = summarize_buckets(rows, g=4)
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["keep"], 2)
        self.assertEqual(s["drop_k0"], 1)
        self.assertEqual(s["drop_k_max"], 1)
        self.assertAlmostEqual(s["keep_ratio"], 0.5)
        self.assertEqual(s["keep_by_hard_case"], {"irony": 1, "metaphor": 1})
        self.assertEqual(s["keep_by_sentiment"], {"positive": 1, "negative": 1})

    def test_summarize_empty(self):
        s = summarize_buckets([], g=4)
        self.assertEqual(s["keep"], 0)
        self.assertEqual(s["keep_ratio"], 0.0)


class TestBudget(unittest.TestCase):
    def test_default_matches_manual(self):
        h = GrpoHyper()
        self.assertEqual(h.prompts_per_step, 8)        # 4 × 2
        self.assertEqual(h.trajectories_per_step, 32)  # 8 题 × G=4 ✅ 手册 1.3
        self.assertEqual(h.steps(51), 7)               # ceil(51/8)

    def test_budget_keys(self):
        b = GrpoHyper().budget(51)
        for k in ("n_prompts", "prompts_per_step", "num_generations",
                  "trajectories_per_step", "steps_per_epoch", "total_steps",
                  "warmup_steps", "lr", "temperature", "beta_kl"):
            self.assertIn(k, b)
        self.assertEqual(b["trajectories_per_step"], 32)
        self.assertGreaterEqual(b["warmup_steps"], 0)

    def test_zero_prompts(self):
        self.assertEqual(GrpoHyper().steps(0), 0)

    def test_completion_budget_note_mentions_json(self):
        self.assertIn("JSON", budget_note(GrpoHyper()))

    def test_generation_shape_valid_and_invalid(self):
        self.assertIsNone(validate_generation_shape(51, 4, 4, 1))
        self.assertIsNotNone(validate_generation_shape(51, 4, 2, 1))   # 2 不能被 4 整除
        self.assertIsNotNone(validate_generation_shape(51, 0, 4, 1))
        self.assertIsNone(validate_generation_shape(51, 4, 4, 2))      # 8 % 4 == 0

    def test_zam_equivalence(self):
        self.assertTrue(assert_zam_equivalence())


# ============================================================
# 3. grpo_probe_core · 打分与 rollout
# ============================================================

class TestCompletionShape(unittest.TestCase):
    def test_str(self):
        self.assertEqual(extract_completion_text("abc"), "abc")

    def test_dict_content(self):
        self.assertEqual(extract_completion_text({"content": "abc"}), "abc")

    def test_list_of_dicts(self):
        self.assertEqual(extract_completion_text([{"content": "abc"}]), "abc")

    def test_empty(self):
        self.assertEqual(extract_completion_text([]), "")
        self.assertEqual(extract_completion_text(None), "")


class TestScoreGroup(unittest.TestCase):
    def test_perfect_group_k_is_four(self):
        res = score_group(PROMPT, [GOOD] * 4, GOLD)
        self.assertEqual(res["k"], 4)
        self.assertTrue(all(t == 2.0 for t in res["totals"]))

    def test_corrupt_group_short_circuits(self):
        res = score_group(PROMPT, [BROKEN] * 4, GOLD)
        self.assertEqual(res["k"], 0)
        self.assertTrue(all(t <= -2.5 for t in res["totals"]))

    def test_mixed_group_k_counts_passes(self):
        wrong = _mk(sentiment="positive")            # 危机漏报
        res = score_group(PROMPT, [GOOD, wrong, GOOD, BROKEN], GOLD)
        self.assertEqual(res["k"], 2)
        self.assertEqual(res["bucket"], "keep")

    def test_grounding_uses_prompt_recovered_input(self):
        # evidence 不在原文里 → 门禁短路，且 input_text 是从 prompt 反解的
        bad = _mk(evidence="这句原文里根本没有的证据")
        res = score_group(PROMPT, [bad], GOLD)
        self.assertEqual(res["totals"][0], -2.5)
        self.assertEqual(res["input_text"], INPUT_TEXT)

    def test_flags_exported(self):
        res = score_group(PROMPT, [GOOD], GOLD)
        f = res["flags"][0]
        for k in ("accuracy_pass", "sentiment_ok", "risk_ok", "hard_case_ok",
                  "relevance_ok", "miss_severe", "false_alarm", "short_circuit"):
            self.assertIn(k, f)
        self.assertTrue(f["accuracy_pass"])


class TestRunProbe(unittest.TestCase):
    def _records(self):
        # 三条同题同 gold（gold 变了 GOOD 就不成立了，k 由采样结果决定）
        return [{"prompt": PROMPT, "gold": dict(GOLD)} for _ in range(3)]

    def test_injected_sampler_all_good(self):
        rows = run_probe(self._records(), lambda p, g: [GOOD] * g, g=4)
        self.assertEqual([r["k"] for r in rows], [4, 4, 4])
        self.assertEqual([r["bucket"] for r in rows], ["drop_k_max"] * 3)

    def test_injected_sampler_all_corrupt(self):
        rows = run_probe(self._records(), lambda p, g: [BROKEN] * g, g=4)
        self.assertEqual([r["k"] for r in rows], [0, 0, 0])
        self.assertEqual([r["bucket"] for r in rows], ["drop_k0"] * 3)

    def test_k_bucket_keeps_only_1_to_3(self):
        # 第 1 题 1 对、第 2 题 2 对、第 3 题 3 对 → 全部 keep
        seqs = [
            [GOOD, _mk(sentiment="positive"), _mk(hard="explicit"), _mk(rel="passing")],
            [GOOD, GOOD, _mk(sentiment="positive"), _mk(hard="explicit")],
            [GOOD, GOOD, GOOD, _mk(sentiment="positive")],
        ]
        it = iter(seqs)
        rows = run_probe(self._records(), lambda p, g: next(it), g=4)
        self.assertEqual([r["k"] for r in rows], [1, 2, 3])
        self.assertEqual([r["bucket"] for r in rows], ["keep"] * 3)

    def test_sampler_wrong_count_raises(self):
        with self.assertRaises(ValueError):
            run_probe(self._records()[:1], lambda p, g: [GOOD] * (g + 1), g=4)

    def test_row_meta_fields(self):
        rows = run_probe([{"prompt": PROMPT, "gold": dict(GOLD, industry="x"),
                           "meta": {"industry": "finance_insurance"}}],
                         lambda p, g: [GOOD] * g, g=4)
        r = rows[0]
        self.assertEqual(r["hard_case_type"], "irony")
        self.assertEqual(r["sentiment"], "negative")
        self.assertEqual(r["industry"], "finance_insurance")
        self.assertEqual(len(r["totals"]), 4)

    def test_oracle_sampler_scores_full_marks(self):
        s = oracle_sampler(GOLD)
        comps = s(PROMPT, 4)
        self.assertEqual(len(comps), 4)
        res = score_group(PROMPT, comps, GOLD)
        self.assertEqual(res["k"], 4, "oracle 反构的标准答案必须满分")


# ============================================================
# 4. train_grpo · flag 拦截 / reward 对齐 / 版本兼容
# ============================================================

class TestFlagInterception(unittest.TestCase):
    def test_value_flags(self):
        opts, rest = tg.intercept_custom_flags(
            ["--data", "a.jsonl", "--log_file", "x.log", "--dry_run"])
        self.assertEqual(opts["data"], "a.jsonl")
        self.assertEqual(opts["log_file"], "x.log")
        self.assertEqual(rest, ["--dry_run"])

    def test_bool_flag(self):
        opts, rest = tg.intercept_custom_flags(["--no_wandb", "--limit", "5"])
        self.assertTrue(opts["no_wandb"])
        self.assertEqual(rest, ["--limit", "5"])
        self.assertFalse(tg.intercept_custom_flags([])[0]["no_wandb"])

    def test_missing_value_raises(self):
        with self.assertRaises(SystemExit):
            tg.intercept_custom_flags(["--data"])

    def test_wandb_run_name_auto_timestamp(self):
        opts = tg.intercept_custom_flags([])[0]
        self.assertTrue(opts["wandb_run_name"].startswith("grpo_"))


class TestRewardFnColumnLayouts(unittest.TestCase):
    """TRL 各版本对 reward 列的对齐方式不一致，三种布局必须算出同一个 gold。"""

    def test_pick_layouts(self):
        g = 4
        # (a) 逐 completion 对齐
        self.assertEqual(tg._pick(["a", "b", "c", "d"], 2, g), "c")
        # (b) 逐 prompt 对齐 (列长 = n//G → 取第 i//G 题)
        self.assertEqual(tg._pick(["p0", "p1", "p2"], 5, g), "p1")   # 5//4 = 1
        self.assertEqual(tg._pick(["p0", "p1"], 6, g), "p1")         # 6//4 = 1
        # (c) 标量
        self.assertEqual(tg._pick("scalar", 3, g), "scalar")
        # (d) 索引越界兜底
        self.assertEqual(tg._pick(["only"], 9, g), "only")

    def test_gold_risk_parsing(self):
        self.assertEqual(tg._gold_risk('["product"]'), ["product"])
        self.assertEqual(tg._gold_risk("[]"), [])
        self.assertEqual(tg._gold_risk(""), [])
        self.assertEqual(tg._gold_risk(["product"]), ["product"])
        self.assertEqual(tg._gold_risk(None), [])

    def test_reward_fn_per_prompt_layout(self):
        fn = tg.make_reward_fn(g=4)
        comps = [GOOD, _mk(sentiment="positive"), GOOD, BROKEN]
        out = fn(completions=comps,
                 input_text=[INPUT_TEXT] * 4,
                 gold_sentiment=["negative"] * 4,
                 gold_relevance=["core"] * 4,
                 gold_hard_case_type=["irony"] * 4,
                 gold_risk_json=['["product", "marketing"]'] * 4,
                 gold_evidence=[GOLD["evidence"]] * 4)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(-2.5 <= v <= 2.0 for v in out))
        self.assertEqual(sum(1 for v in out if v > 0), 2)

    def test_reward_fn_completions_as_dicts(self):
        fn = tg.make_reward_fn(g=1)
        out = fn(completions=[{"content": GOOD}],
                 input_text=[INPUT_TEXT],
                 gold_sentiment=["negative"], gold_relevance=["core"],
                 gold_hard_case_type=["irony"],
                 gold_risk_json=['["product", "marketing"]'],
                 gold_evidence=[GOLD["evidence"]])
        self.assertEqual(out, [2.0])


class TestCompatHelpers(unittest.TestCase):
    def test_check_versions_no_strict(self):
        info = tg.check_versions(strict=False)
        self.assertIn("trl", info)
        self.assertIn("transformers", info)

    def test_is_main_process(self):
        self.assertTrue(tg.is_main_process())

    def test_decide_wandb_off(self):
        args = type("A", (), {"report_to": "wandb"})()
        self.assertFalse(tg.decide_wandb(args, {"no_wandb": True}))
        self.assertFalse(tg.decide_wandb(type("A", (), {"report_to": "none"})(), {}))

    def test_build_trainer_renames_processor_arg(self):
        class _Old:                      # 旧版：只认 tokenizer=，多给就 TypeError
            def __init__(self, tokenizer=None):
                self.tokenizer = tokenizer

        class _New:
            def __init__(self, processing_class=None, **kw):
                self.processing_class = processing_class

        t = tg.build_trainer(_Old, processing_class="tok")
        self.assertEqual(t.tokenizer, "tok")
        t2 = tg.build_trainer(_New, processing_class="tok")
        self.assertEqual(t2.processing_class, "tok")

    def test_build_trainer_propagates_unrelated_typeerror(self):
        class _Bad:
            def __init__(self, **kw):
                raise TypeError("其他字段不认")

        with self.assertRaises(TypeError):
            tg.build_trainer(_Bad, processing_class="tok")

    def test_dry_run_inspect_runs(self):
        """干跑体检必须能在没有 torch 的机器上跑通 (关键路径自检)。"""
        import tempfile
        from grpo_common import attach_log, detach_log
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as f:
            f.write(json.dumps({"prompt": PROMPT, "gold": GOLD},
                               ensure_ascii=False) + "\n")
            path = f.name
        try:
            h = GrpoHyper()
            tg.dry_run_inspect(path, h)          # 不应抛异常
            self.assertGreater(len(tg.load_records(path)), 0)
        finally:
            os.unlink(path)


class TestScoreOneIsSingleSource(unittest.TestCase):
    def test_score_one_matches_compute_reward(self):
        from reward_fns import compute_reward
        a = score_one(GOOD, INPUT_TEXT, GOLD)
        b = compute_reward(GOOD, INPUT_TEXT, GOLD)
        self.assertEqual(a, b)
        self.assertEqual(a["total"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestShouldFuse(unittest.TestCase):
    """对齐税熔断判定（GRPO 手册模块 3 闸② / TODO 5.19）。"""

    def test_fuse_when_drop_exceeds_threshold(self):
        self.assertTrue(should_fuse(1.5, 1.5 * 0.96))          # 跌 4%

    def test_no_fuse_when_drop_within_threshold(self):
        self.assertFalse(should_fuse(1.5, 1.5 * 0.99))         # 跌 1%

    def test_none_baseline_means_no_fuse(self):
        self.assertFalse(should_fuse(None, 1.0))               # 首轮 eval，无基线

    def test_none_current_means_no_fuse(self):
        self.assertFalse(should_fuse(1.0, None))               # 缺数据不是退化

    def test_nonpositive_baseline_uses_abs_diff(self):
        self.assertTrue(should_fuse(-2.0, -2.1, 0.02))         # 绝对差 0.1 > 0.02
        self.assertFalse(should_fuse(-2.0, -1.9, 0.02))

    def test_exactly_threshold_does_not_fuse(self):
        self.assertFalse(should_fuse(1.0, 0.98))               # 恰 2%：严格大于才熔断
        self.assertTrue(should_fuse(1.0, 0.979))               # 2.1% 熔断
