"""检索编排：稠密 + 稀疏双路召回 → RRF 融合 → 重排 → 证据块。

- ``qdrant_sparse`` 模式：双路召回与 RRF 由 Qdrant Query API 原生完成（单次请求）；
- ``jieba_bm25`` 模式：稠密走 Qdrant、稀疏走本地 BM25 索引，应用层 RRF 融合（兜底）。
"""

from __future__ import annotations

from ..config import Settings
from ..embedding.encoder import Qwen3Embedder
from ..models import SearchResult
from ..sparse.bm25 import ChineseBM25, SparseEncoder
from ..store.qdrant_store import QdrantStore
from .rerank import Qwen3Reranker


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = QdrantStore(settings)
        self.embedder = Qwen3Embedder(settings)
        self.sparse_encoder = SparseEncoder(settings.paths.db_dir / "sparse_vocab.json")
        self.bm25 = ChineseBM25(k1=settings.sparse.k1, b=settings.sparse.b)
        self._bm25_loaded = False
        self.reranker = Qwen3Reranker(settings) if settings.rerank.enabled else None

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        doc_id: str | None = None,
    ) -> list[SearchResult]:
        r = self.settings.retrieval
        final_k = top_k or r.final_top
        if use_rerank is None:
            use_rerank = self.settings.rerank.enabled

        qvec = self.embedder.encode_query(query)

        if self.settings.sparse.method == "qdrant_sparse":
            indices, values = self.sparse_encoder.encode(query, update_vocab=False)
            hits = self.store.hybrid_search(
                qvec,
                (indices, values),
                dense_limit=r.dense_top,
                sparse_limit=r.sparse_top,
                limit=r.fused_top,
                doc_id=doc_id,
            )
            results = [self._to_result(hit) for hit in hits]
        else:
            results = self._hybrid_local(qvec, query, doc_id)

        if use_rerank and self.reranker is not None and results:
            # 重排输入使用含面包屑的 text（标题路径提供上下文）
            scores = self.reranker.score(query, [x.text for x in results])
            for res, s in zip(results, scores):
                res.rerank_score = s
            results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        return results[:final_k]

    # ------------------------------------------------------------------
    def _hybrid_local(self, qvec, query: str, doc_id: str | None) -> list[SearchResult]:
        """jieba_bm25：本地 BM25 + 应用层 RRF。"""
        r = self.settings.retrieval
        dense_hits = self.store.dense_search(qvec, r.dense_top, doc_id=doc_id)

        self._ensure_bm25()
        bm25_hits = self.bm25.search(query, r.sparse_top)
        if doc_id:  # 本地 BM25 无过滤能力：检索后按 doc 过滤
            bm25_hits = [h for h in bm25_hits if self._belongs(h[0], doc_id)]
        # 补齐缺失 payload
        missing = [pid for pid, _ in bm25_hits if pid not in {h[0] for h in dense_hits}]
        fetched = self.store.get_payloads(missing) if missing else {}

        fused: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        scores: dict[str, float] = {}
        for rank, (pid, payload, score) in enumerate(dense_hits):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (r.rrf_k + rank + 1)
            payloads[pid] = payload
            scores[pid] = score
        for rank, (pid, score) in enumerate(bm25_hits):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (r.rrf_k + rank + 1)
            payloads.setdefault(pid, fetched.get(pid, {}))
            scores.setdefault(pid, score)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[: r.fused_top]
        return [
            self._to_result((pid, payloads[pid], fused_score))
            for pid, fused_score in ordered
        ]

    def _ensure_bm25(self) -> None:
        if self._bm25_loaded:
            return
        ids, texts = self.store.scroll_texts()
        self.bm25.build(ids, texts)
        self._bm25_loaded = True

    def _belongs(self, point_id: str, doc_id: str) -> bool:
        payload = self.store.get_payloads([point_id]).get(point_id, {})
        return payload.get("doc_id") == doc_id

    # ------------------------------------------------------------------
    @staticmethod
    def _to_result(hit: tuple[str, dict, float]) -> SearchResult:
        pid, payload, score = hit
        return SearchResult(
            chunk_id=pid,
            score=float(score),
            rerank_score=None,
            text=payload.get("text", ""),
            raw_text=payload.get("raw_text", ""),
            doc_id=payload.get("doc_id", ""),
            doc_title=payload.get("doc_title", ""),
            source_path=payload.get("source_path", ""),
            page=payload.get("page"),
            heading_path=payload.get("heading_path") or [],
            chunk_type=payload.get("chunk_type", "text"),
            parent_id=payload.get("parent_id"),
            parent_text=payload.get("parent_text"),
        )