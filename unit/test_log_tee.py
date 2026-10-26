#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT / GRPO 共用日志双写 · 离线单测。"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from log_tee import Tee, attach_log, detach_log  # noqa: E402


class TestLogTee(unittest.TestCase):
    def test_attach_and_detach(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "train.log")
            terminal_out, terminal_err = io.StringIO(), io.StringIO()
            with patch.object(sys, "stdout", terminal_out), patch.object(sys, "stderr", terminal_err):
                handle = attach_log(path)
                try:
                    print("输出\r进度", end="")
                    print("报错", file=sys.stderr)
                    self.assertIsInstance(sys.stdout, Tee)
                    self.assertEqual(sys.stdout.encoding, terminal_out.encoding)
                    self.assertFalse(sys.stdout.isatty())
                    self.assertEqual(sys.stdout.write(""), 0)
                finally:
                    detach_log(handle)
                self.assertIs(sys.stdout, terminal_out)
                self.assertIs(sys.stderr, terminal_err)
                self.assertTrue(handle["fh"].closed)
                detach_log(handle)  # 重复关闭安全
            self.assertIn("输出\r进度", terminal_out.getvalue())
            self.assertIn("报错", terminal_err.getvalue())
            with open(path, encoding="utf-8") as fh:
                log = fh.read()
            self.assertIn("训练任务启动", log)
            self.assertIn("输出\n进度", log)
            self.assertIn("报错", log)

    def test_append_and_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.log")
            self.assertIsNone(attach_log(None))
            detach_log(None)
            terminal_out = io.StringIO()
            with patch.object(sys, "stdout", terminal_out):
                for text in ("第一轮", "第二轮"):
                    handle = attach_log(path)
                    try:
                        print(text)
                    finally:
                        detach_log(handle)
            with open(path, encoding="utf-8") as fh:
                log = fh.read()
            self.assertEqual(log.count("训练任务启动"), 2)
            self.assertIn("第一轮", log)
            self.assertIn("第二轮", log)

    def test_terminal_file_descriptor_pass_through(self):
        with tempfile.TemporaryFile(mode="w+") as terminal, io.StringIO() as logfile:
            tee = Tee(terminal, logfile, terminal)
            self.assertEqual(tee.fileno(), terminal.fileno())
            self.assertEqual(tee.encoding, terminal.encoding)


if __name__ == "__main__":
    unittest.main()
