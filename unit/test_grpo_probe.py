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
    GrpoHyper, bucket_label, group_totals,
    k_of, score_one, summarize_buckets, assert_zam_equivalence,
    budget_note, validate_generation_shape,
    should_fuse,
)
from grpo_dataset import (  # noqa: E402
    build_dataset_rows, flatten_gold, rebuild_gold,
)
from grpo_probe_core import (  # noqa: E402
    extract_completion_text, oracle_sampler, run_probe, score_group,
)
import train_grpo as tg  # noqa: E402
import grpo_eval_fuse as gef  # noqa: E402


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
    # 2026-10-27 R_length 阈值重设为 120/200/320 (750 条实测 p50=201)。
    # 自然展开的三跳推理 ≈190 汉字 ≈138 tokens，落在理想带 → GOOD 组 total = 2.0。
    cot = cot or (
        "Step1 主体识别：通读全文，叙事焦点始终落在广发证券身上，主体相关度判 core。"
        "Step2 语义分析：字面用「稳赚不赔」的褒义表述，但紧跟「才一天就亏了3000多」的事实反转，"
        "典型反讽修辞，难例类型判 irony。Step3 风险归因：亏损直接指向理财产品表现归 product，"
        "「稳赚不赔」的宣传口径涉嫌误导归 marketing。综合判 negative。"
    )
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
        # trl 0.14 语义: per_device=prompt 数, 无整除约束 (任意 ≥1 合法);
        # 整除报错只在 trl>=1.x 有意义 (validate_generation_shape 已按版本语义放宽)
        self.assertIsNone(validate_generation_shape(51, 4, 4, 1))
        self.assertIsNone(validate_generation_shape(51, 4, 2, 1))      # 0.14: 合法
        self.assertIsNone(validate_generation_shape(51, 4, 1, 1))      # 0.14: 合法 (冒烟实测)
        self.assertIsNone(validate_generation_shape(51, 4, 4, 2))      # 8 % 4 == 0
        self.assertIsNotNone(validate_generation_shape(51, 0, 4, 1))   # G 必须 ≥1
        self.assertIsNotNone(validate_generation_shape(51, 4, 0, 1))   # per_device ≥1

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
        # 2026-10-27 阶梯式 accuracy 后 k 口径变化: 情感对即 accuracy_pass。
        # 构造非 pass 轨迹必须真的打掉 sentiment (rel/hard/risk 错现在都算 pass)。
        seqs = [
            [GOOD, _mk(sentiment="positive"), _mk(sentiment="neutral"), _mk(sentiment="positive")],
            [GOOD, GOOD, _mk(sentiment="positive"), _mk(sentiment="neutral")],
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
# 4. train_grpo · 简化 CLI / reward 按 completion 对齐
# ============================================================

class TestTrainGrpoEntry(unittest.TestCase):
    def test_prompt_length_is_raised_without_truncation(self):
        self.assertEqual(tg.resolve_prompt_limit(1000, 500), 1058)

    def test_prompt_over_hard_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "4096"):
            tg.resolve_prompt_limit(4097, 2048)

    def test_prompt_within_configured_limit_is_unchanged(self):
        self.assertEqual(tg.resolve_prompt_limit(1000, 2048), 2048)

    def test_parser_accepts_launcher_flags(self):
        args = tg.build_parser().parse_args([
            "--data", "a.jsonl", "--log_file", "x.log", "--dry_run", "--no_wandb"])
        self.assertEqual(args.data, "a.jsonl")
        self.assertEqual(args.log_file, "x.log")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.no_wandb)

    def test_eval_options_are_explicitly_supported(self):
        args = tg.build_parser().parse_args([
            "--eval_file", "eval.jsonl", "--eval_steps", "2",
            "--fuse_threshold", "0.02", "--fuse_hard"])
        self.assertEqual(args.eval_file, "eval.jsonl")
        self.assertEqual(args.eval_steps, 2)
        self.assertEqual(args.fuse_threshold, 0.02)
        self.assertTrue(args.fuse_hard)

    def test_per_completion_reward(self):
        completions = [GOOD, _mk(sentiment="positive"), GOOD, BROKEN]
        out = tg.make_reward_fn(g=4)(
            completions=completions,
            input_text=[INPUT_TEXT] * 4,
            gold_sentiment=["negative"] * 4,
            gold_relevance=["core"] * 4,
            gold_hard_case_type=["irony"] * 4,
            gold_risk_json=['["product", "marketing"]'] * 4,
            gold_evidence=[GOLD["evidence"]] * 4)
        self.assertEqual(len(out), 4)
        self.assertEqual(sum(v > 0 for v in out), 2)

    def test_mismatched_gold_column_raises(self):
        with self.assertRaisesRegex(ValueError, "gold_sentiment"):
            tg.make_reward_fn(g=4)(
                completions=[GOOD] * 8,
                gold_sentiment=["negative", "positive"],
                gold_relevance=["core"] * 8,
                gold_hard_case_type=["irony"] * 8,
                gold_risk_json=['["product", "marketing"]'] * 8,
                gold_evidence=[GOLD["evidence"]] * 8,
                input_text=[INPUT_TEXT] * 8)

    def test_reward_accepts_completion_dict(self):
        out = tg.make_reward_fn(g=1)(
            completions=[{"content": GOOD}], input_text=[INPUT_TEXT],
            gold_sentiment=["negative"], gold_relevance=["core"],
            gold_hard_case_type=["irony"],
            gold_risk_json=['["product", "marketing"]'],
            gold_evidence=[GOLD["evidence"]])
        self.assertEqual(out, [2.0])

    def test_wandb_off(self):
        args = tg.build_parser().parse_args(["--no_wandb"])
        self.assertFalse(tg.wandb_enabled(args))

    def test_dry_run_reports_optional_eval_dataset(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as stream:
            stream.write(json.dumps({"prompt": PROMPT, "gold": GOLD},
                                    ensure_ascii=False) + "\n")
            path = stream.name
        try:
            with patch.object(sys, "argv", ["train_grpo.py", "--data", path,
                                             "--eval_file", path, "--dry_run", "--no_wandb"]):
                from contextlib import redirect_stdout
                from io import StringIO
                output = StringIO()
                with redirect_stdout(output):
                    tg.main()
            self.assertIn("eval 基线集", output.getvalue())
            self.assertIn("熔断 callback 将在真实训练时注册", output.getvalue())
        finally:
            os.unlink(path)

    def test_dry_run_uses_no_training_dependencies(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as stream:
            stream.write(json.dumps({"prompt": PROMPT, "gold": GOLD},
                                    ensure_ascii=False) + "\n")
            path = stream.name
        try:
            with patch.object(sys, "argv", ["train_grpo.py", "--data", path,
                                             "--dry_run", "--no_wandb"]):
                tg.main()
        finally:
            os.unlink(path)

class TestScoreOneIsSingleSource(unittest.TestCase):
    def test_score_one_matches_compute_reward(self):
        from reward_fns import compute_reward
        a = score_one(GOOD, INPUT_TEXT, GOLD)
        b = compute_reward(GOOD, INPUT_TEXT, GOLD)
        self.assertEqual(a, b)
        self.assertEqual(a["total"], 2.0)


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

    def test_nonpositive_baseline_no_silent_absdiff_branch(self):
        # [P0 20261030] 旧实现在 baseline ≤ 0 时退化为「绝对差 > 0.02」，负分区的
        # 0.02 绝对变化是噪声级，静默熔断。改为统一走相对跌幅判据（baseline=0 不动作）。
        self.assertTrue(should_fuse(-2.0, -2.2, 0.02))          # 相对跌幅 10%
        self.assertTrue(should_fuse(-2.0, -2.1, 0.02))          # 相对跌幅 5% > 2%
        self.assertFalse(should_fuse(-2.0, -2.02, 0.02))        # 相对跌幅 1%：噪声不熔断
        self.assertFalse(should_fuse(0.0, -0.5, 0.02))          # 基线恰为 0：测度零，不动作

    def test_exactly_threshold_does_not_fuse(self):
        self.assertFalse(should_fuse(1.0, 0.98))               # 恰 2%：严格大于才熔断
        self.assertTrue(should_fuse(1.0, 0.979))               # 2.1% 熔断


# ============================================================
# 5. 回归：真实训练路径的启动期崩溃（dry_run 看不出来的那类）
# ============================================================

class _FakeTrainerState:
    """模拟 HF 递进回调的 TrainerState：**不可下标**。

    熔断 callback 的闭包状态若与回调形参同名，第一次 `state["baseline"]`
    就会抛 TypeError —— 本类专门把这种遮蔽钉死。
    """

    def __getitem__(self, key):
        raise TypeError(f"TrainerState 不可下标（疑似闭包状态被形参遮蔽）: {key}")

    def __init__(self, global_step=1):
        self.global_step = global_step


class _FakeArgs:
    """on_step_end 触发只看 args.eval_steps（Trainer 内置 eval 已关）。"""

    def __init__(self, eval_steps=1):
        self.eval_steps = eval_steps


class TestFuseCallbackBuildAndFuse(unittest.TestCase):
    """独立 eval 熔断模块的可选 callback 单测，不属于 train_grpo 主链路。"""

    def setUp(self):
        import types
        self._orig_modules = {k: sys.modules.get(k) for k in ("transformers",)}
        fake = types.ModuleType("transformers")

        class _CB:                       # 只提供本用例用到的基类
            pass

        fake.TrainerCallback = _CB
        sys.modules["transformers"] = fake
        self._orig_eval = gef._eval_reward_mean
        self._ctrl = type("C", (), {"should_training_stop": False})()

    def tearDown(self):
        gef._eval_reward_mean = self._orig_eval
        for k, v in self._orig_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def _build(self, fuse_hard: bool, seq):
        it = iter(seq)
        calls = []

        def eval_mean(tr, rf, g, eval_rows=None):
            calls.append(True)
            value = next(it)
            if isinstance(value, Exception):
                raise value
            return value

        gef._eval_reward_mean = eval_mean
        cb = gef.build_fuse_callback(fuse_threshold=0.02, fuse_hard=fuse_hard,
                                    reward_fn=object(), g=4)
        trainer = object()
        cb._attach(trainer)
        cb.initialize_baseline(trainer)
        self.assertEqual(len(calls), 1, "训练前 baseline 应只生成一次")
        return cb

    def _fire(self, cb, eval_steps=1, global_step=1):
        """on_step_end 周期触发一次（callback 改为自带 eval 节流的接线方式）。"""
        return cb.on_step_end(_FakeArgs(eval_steps), _FakeTrainerState(global_step),
                              self._ctrl)

    def test_callback_instantiates_and_sets_pretrain_baseline(self):
        cb = self._build(fuse_hard=True, seq=[1.0, 1.0])
        self.assertTrue(callable(cb.on_step_end))
        self._fire(cb)
        self.assertFalse(self._ctrl.should_training_stop)

    def test_first_training_eval_compares_against_pretrain_baseline(self):
        cb = self._build(fuse_hard=True, seq=[1.0, 0.90])
        self._fire(cb)                                          # 跌 10% > 2%
        self.assertTrue(self._ctrl.should_training_stop)

    def test_soft_fuse_only_warns(self):
        cb = self._build(fuse_hard=False, seq=[1.0, 0.90])
        self._fire(cb)
        self.assertFalse(self._ctrl.should_training_stop)

    def test_drop_within_threshold_keeps_training(self):
        cb = self._build(fuse_hard=True, seq=[1.0, 0.995])      # 跌 0.5%
        self._fire(cb)
        self.assertFalse(self._ctrl.should_training_stop)

    def test_no_trainer_attached_is_noop(self):
        gef._eval_reward_mean = lambda tr, rf, g, eval_rows=None: 0.0
        cb = gef.build_fuse_callback(fuse_threshold=0.02, fuse_hard=True,
                                    reward_fn=object(), g=4)
        self._fire(cb)                                          # 未 _attach
        self.assertFalse(self._ctrl.should_training_stop)

    def test_eval_error_does_not_crash_training(self):
        cb = self._build(fuse_hard=True, seq=[1.0, RuntimeError("generate 炸了")])
        self._fire(cb)                                          # 后续 eval 异常只 WARN
        self.assertFalse(self._ctrl.should_training_stop)

    def test_eval_steps_throttle_skips_non_multiple_steps(self):
        cb = self._build(fuse_hard=True, seq=[1.0, 0.90])
        self._fire(cb, eval_steps=4, global_step=2)             # 非倍数步：不触发
        self.assertFalse(self._ctrl.should_training_stop)



if __name__ == "__main__":
    unittest.main(verbosity=2)
