#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简 LLM 客户端 (纯标准库 urllib，零第三方依赖)

支持任何 OpenAI 兼容端点:
  - 阿里百炼 DashScope : https://dashscope.aliyuncs.com/compatible-mode/v1
  - DeepSeek          : https://api.deepseek.com/v1
  - 智谱 GLM          : https://open.bigmodel.cn/api/paas/v4
  - 月之暗面 Kimi      : https://api.moonshot.cn/v1

设计要点:
  1. 指数退避重试 (应对限流 429 / 5xx 抖动)
  2. JSON 模式支持 (response_format=json_object)
  3. 记录 token 用量与耗时，供成本核算 (SOP §1.3.1 要求成本预算)
  4. 绝不打印 API Key
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 各家默认端点
DEFAULT_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "openai": "https://api.openai.com/v1",
}

# 各 provider 常用模型名
KNOWN_MODELS = {
    "dashscope": ["qwen-max", "qwen-plus", "qwen-turbo", "qwen3-max"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
}


@dataclass
class LLMUsage:
    """累计用量记账 (跨所有调用)。线程安全：并发生成时多个 worker 共享同一实例。"""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    failures: int = 0
    elapsed_sec: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, init=False)

    def add(self, usage: Dict[str, Any], elapsed: float):
        with self._lock:
            self.calls += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
            self.total_tokens += usage.get("total_tokens", 0) or 0
            self.elapsed_sec += elapsed

    def add_failure(self):
        with self._lock:
            self.failures += 1

    def summary(self) -> str:
        with self._lock:
            return (f"调用 {self.calls} 次 (失败 {self.failures}) | "
                    f"prompt {self.prompt_tokens} + completion {self.completion_tokens} "
                    f"= {self.total_tokens} tokens | 耗时 {self.elapsed_sec:.1f}s")


class LLMClient:
    """
    一个 provider 一个实例。线程安全（无共享可变状态，除 usage 计数）。
    """

    def __init__(
        self,
        provider: str = "dashscope",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 180,
        max_retries: int = 4,
        verbose: bool = True,
    ):
        self.provider = provider
        self.base_url = (base_url or DEFAULT_BASE_URLS.get(provider, "")).rstrip("/")
        if not self.base_url:
            raise ValueError(f"未知 provider '{provider}'，请显式指定 base_url")

        self.model = model or (KNOWN_MODELS.get(provider, ["qwen-max"])[0])

        # Key 解析顺序: 显式传入 > 环境变量
        env_map = {
            "dashscope": ["DASHSCOPE_API_KEY", "QWEN_API_KEY", "ALIYUN_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "zhipu": ["ZHIPU_API_KEY", "GLM_API_KEY"],
            "moonshot": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
        }
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = None
            for var in env_map.get(provider, []):
                if os.environ.get(var):
                    self.api_key = os.environ[var]
                    break
        if not self.api_key:
            raise ValueError(
                f"未找到 {provider} 的 API Key。请设置环境变量 "
                f"{' 或 '.join(env_map.get(provider, ['<PROVIDER>_API_KEY']))}"
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.usage = LLMUsage()

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 2048,
        json_mode: bool = False,
        top_p: float = 0.9,
    ) -> str:
        """
        单轮对话，返回助手文本。失败抛 RuntimeError。
        json_mode=True 时要求模型输出合法 JSON 对象。
        """
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        last_err = None
        for attempt in range(self.max_retries):
            start = time.time()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - start
                self.usage.add(data.get("usage", {}), elapsed)

                # 截断检测：finish_reason=length 说明 max_tokens 不够，
                # 模型输出被硬切，下游解析必然缺尾巴（如 <|end_of_thought|>）
                choice = data["choices"][0]
                fr = choice.get("finish_reason")
                if fr == "length":
                    u = data.get("usage", {})
                    raise RuntimeError(
                        f"[{self.provider}/{self.model}] 输出被截断 "
                        f"(finish_reason=length, completion="
                        f"{u.get('completion_tokens')} >= max_tokens={max_tokens})，"
                        f"请调大 max_tokens")
                return choice["message"]["content"]

            except urllib.error.HTTPError as e:
                body_text = ""
                try:
                    body_text = e.read().decode("utf-8")[:400]
                except Exception:
                    pass
                last_err = f"HTTP {e.code}: {body_text}"
                # 4xx（除 429）通常不可重试
                if 400 <= e.code < 500 and e.code != 429:
                    self.usage.add_failure()
                    raise RuntimeError(f"[{self.provider}/{self.model}] {last_err}")
            except RuntimeError:
                # 截断属于确定性失败，直接抛出，不再重试
                self.usage.add_failure()
                raise
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

            self.usage.add_failure()
            if attempt < self.max_retries - 1:
                sleep_s = min(2 ** attempt, 20)
                if self.verbose:
                    print(f"    [重试 {attempt + 1}/{self.max_retries}] {last_err} → 等待 {sleep_s}s")
                time.sleep(sleep_s)

        raise RuntimeError(f"[{self.provider}/{self.model}] 重试耗尽: {last_err}")

    # ------------------------------------------------------------------
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> Any:
        """要求 JSON 输出并解析（容忍 ```json 围栏）。"""
        raw = self.chat(messages, temperature=temperature,
                        max_tokens=max_tokens, json_mode=True)
        return parse_json_loose(raw)


def parse_json_loose(text: str) -> Any:
    """宽松 JSON 解析：剥离 markdown 围栏、截取最外层花括号。"""
    if text is None:
        raise ValueError("空响应")
    s = text.strip()

    if s.startswith("```"):
        s = s.split("```")[1] if len(s.split("```")) > 1 else s
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 截取最外层 {...}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError(f"无法解析 JSON: {text[:200]}")


if __name__ == "__main__":
    # 自检：连通性 + 模型可用性
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else "dashscope"
    model = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"测试 {provider}" + (f" / {model}" if model else ""))
    try:
        c = LLMClient(provider=provider, model=model)
        print(f"  端点: {c.base_url}")
        print(f"  模型: {c.model}")
        out = c.chat([{"role": "user", "content": "只回复两个字：连通"}],
                     temperature=0, max_tokens=20)
        print(f"  回复: {out!r}")
        print(f"  {c.usage.summary()}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise SystemExit(1)
