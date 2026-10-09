"""稀疏检索：jieba 中文分词 + BM25。

两种模式（config.sparse.method）：

- ``qdrant_sparse``：把词频构造为稀疏向量写入 Qdrant（服务端 IDF 修正，近似 BM25），
  与稠密路一起用 Qdrant 原生 prefetch + RRF 融合；
- ``jieba_bm25``：rank_bm25 本地索引 + 应用层 RRF 融合（兜底方案，不依赖 Qdrant 稀疏能力）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
import numpy as np

_TOKEN_KEEP = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词；过滤纯标点/空白，英文统一小写。"""
    tokens = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok or not _TOKEN_KEEP.search(tok):
            continue
        tokens.append(tok.lower())
    return tokens


class SparseEncoder:
    """qdrant_sparse 模式：term → index 词表 + 词频权重。词表持久化于 db_dir/sparse_vocab.json。"""

    def __init__(self, vocab_path: Path):
        self.vocab_path = vocab_path
        self.vocab: dict[str, int] = {}
        if vocab_path.exists():
            self.vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_path.write_text(
            json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8"
        )

    def reset(self) -> None:
        self.vocab = {}
        if self.vocab_path.exists():
            self.vocab_path.unlink()

    def encode(self, text: str, update_vocab: bool) -> tuple[list[int], list[float]]:
        """→ (indices, values)：词频权重；update_vocab=True 时登记新词（摄入阶段）。"""
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        indices: list[int] = []
        values: list[float] = []
        for tok, freq in counts.items():
            if tok not in self.vocab:
                if not update_vocab:
                    continue
                self.vocab[tok] = len(self.vocab)
            indices.append(self.vocab[tok])
            values.append(float(freq))
        return indices, values


class ChineseBM25:
    """jieba_bm25 模式：本地 BM25 索引（rank_bm25），支持增量重建。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._bm25 = None

    def build(self, ids: list[str], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = list(ids)
        corpus = [tokenize(t) for t in texts]
        corpus = [c if c else ["_"] for c in corpus]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.ids:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]