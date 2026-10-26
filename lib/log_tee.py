#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志双写 · 纯 Python (零 torch/trl 依赖，可独立单测)

职责:
  stdout/stderr 同时写终端与日志文件 (tee)。
  覆盖三类输出源: print、transformers.logging (走 StreamHandler→sys.stdout)、tqdm。

  SFT (train_sft_lora.py) 与 GRPO (train_grpo.py) 共用 —— 原先两侧各持一份
  _Tee/Tee 实现，SFT 版是超集 (\r→\n、isatty/fileno/encoding 透传)，
  统一以 SFT 版为基准下沉到这里。

用法:
  handle = attach_log("logs/train.log")   # None = 未启用
  ...
  detach_log(handle)                      # 恢复 stdout/stderr 并关文件
"""

import io
import os
import sys
import time
from typing import Any, Optional


class Tee:
    """把 stdout/stderr 同时写向原终端与日志文件。"""

    def __init__(self, stream, file_handle, orig):
        self._stream = stream
        self._file = file_handle
        self._orig = orig

    def write(self, data):
        try:
            self._orig.write(data)
        except Exception:
            pass
        try:
            self._file.write(data.replace("\r", "\n"))  # tqdm 的 \r 换成换行，避免文件里糊成一行
            self._file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    # tqdm / print(end="") 需要的属性透传
    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._orig.fileno()


def attach_log(path: Optional[str]) -> Optional[Any]:
    """
    启用日志双写，返回句柄供收尾 detach_log()。
    以追加模式打开，保证同一文件可承载多次探针迭代。
    path 为空 → 返回 None (未启用，调用方跳过)。
    """
    if not path:
        return None
    log_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = io.open(log_path, "a", encoding="utf-8", buffering=1)
    fh.write("\n" + "=" * 74 + "\n")
    fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 训练任务启动\n")
    fh.write("=" * 74 + "\n")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, fh, old_out)
    sys.stderr = Tee(sys.stderr, fh, old_err)
    return {"fh": fh, "log_path": log_path, "streams": (old_out, old_err)}


def detach_log(handle) -> None:
    """恢复 stdout/stderr 并关闭日志文件；None / 已 detach 均为 no-op (可安全重复调用)。"""
    if not handle or handle.get("_detached"):
        return
    handle["_detached"] = True
    sys.stdout, sys.stderr = handle["streams"]
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        handle["fh"].close()
    except Exception:
        pass
