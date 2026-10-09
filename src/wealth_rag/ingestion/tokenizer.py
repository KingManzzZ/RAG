"""Token 计数与硬切分。

优先使用 tiktoken（cl100k_base）——与 RAGFlow 的计数方式对齐（common/token_utils.py）；
在 tiktoken 不可用（如无法下载词表）时退化为启发式估算，保证管线可运行。
"""

from __future__ import annotations

import math
import os
from pathlib import Path


class TokenCounter:
    def __init__(self, cache_dir: Path | None = None):
        self._enc = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir))
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:  # 无网络 / 词表不可用时退化
            self._enc = None

    @property
    def exact(self) -> bool:
        return self._enc is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        # 启发式：CJK 字符约 1 token/字，其余约 1 token/4 字符
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        others = len(text) - cjk
        return cjk + max(1, math.ceil(others / 4))

    def hard_split(self, text: str, max_tokens: int) -> list[str]:
        """将超长文本按 token 上限硬切分（③ 超限兜底）。"""
        if self.count(text) <= max_tokens:
            return [text]
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            return [
                self._enc.decode(ids[i : i + max_tokens])
                for i in range(0, len(ids), max_tokens)
            ]
        # 启发式：按字符近似（CJK 约 1.5 字符/token）
        step = max(1, int(max_tokens * 1.5))
        return [text[i : i + step] for i in range(0, len(text), step)]

    def tail(self, text: str, n_tokens: int) -> str:
        """取末尾 n_tokens 个 token 的文本（块间重叠用）。"""
        if n_tokens <= 0 or not text:
            return ""
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            return self._enc.decode(ids[-n_tokens:])
        step = max(1, int(n_tokens * 1.5))
        return text[-step:]