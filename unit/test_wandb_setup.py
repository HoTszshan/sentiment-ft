#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lib/wandb_setup.setup_wandb 单测（零 wandb 依赖：探测行为用假 sys.modules 覆盖）"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from wandb_setup import setup_wandb  # noqa: E402


class TestSetupWandb(unittest.TestCase):
    def test_no_wandb_flag_disables(self):
        # 显式禁用: 不 import wandb、不动 env
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(setup_wandb("p", "r", no_wandb=True))

    def test_user_report_to_none_disables(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(setup_wandb("p", "r", user_report_to="none"))

    def test_empty_report_to_forms_disable(self):
        # GRPO 原口径: "" / None 也算禁用形态; SFT 的 list 形态 [""] 同理
        # (None=未传 → 不算禁用，交给未安装降级兜底)
        for rt in ("", [], [""]):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(setup_wandb("p", "r", user_report_to=rt), f"rt={rt!r}")

    def test_wandb_missing_degrades(self):
        # 未安装 wandb → WARN + False
        with mock.patch.dict(sys.modules, {"wandb": None}), \
             mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("builtins.print") as p:
            self.assertFalse(setup_wandb("p", "r"))
            self.assertTrue(any("未安装 wandb" in str(c.args) for c in p.call_args_list))

    def test_ready_injects_env(self):
        # ready: 三件套注入 + 不覆盖用户已有 env（注入假 wandb 模块，不依赖真安装）
        fake = mock.MagicMock()
        env = {"WANDB_NAME": "user-run"}
        with mock.patch.dict(sys.modules, {"wandb": fake}), \
             mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(setup_wandb("proj", "auto-run"))
            self.assertEqual(os.environ["WANDB_PROJECT"], "proj")
            self.assertEqual(os.environ["WANDB_NAME"], "user-run")      # setdefault 不覆盖
            self.assertEqual(os.environ["WANDB_LOG_MODEL"], "false")

    def test_wandb_none_in_modules_counts_as_missing(self):
        # sys.modules["wandb"]=None 时 import 抛 ImportError → 降级
        with mock.patch.dict(sys.modules, {"wandb": None}), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(setup_wandb("p", "r"))


if __name__ == "__main__":
    unittest.main()
