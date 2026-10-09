"""Qdrant 本地向量库封装（零部署本地模式；后续可无缝切换服务模式）。

- 命名向量：``dense``（Qwen3-Embedding，余弦）+ ``sparse``（jieba 词频，IDF 修正，可选）
- 检索：Query API；qdrant_sparse 模式下使用原生 ``prefetch`` + ``RRF`` 融合
- 过滤：payload 过滤（doc_id / 文档类型）
"""

from __future__ import annotations

import atexit

import numpy as np
from qdrant_client import QdrantClient, models

from ..config import Settings
from ..models import Chunk

COLLECTION = "chunks"


class QdrantStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(path=str(settings.paths.db_dir / "qdrant"))
        self.collection = COLLECTION
        self.sparse_enabled = settings.sparse.method == "qdrant_sparse"
        self._closed = False
        # 本地模式在解释器退出阶段的自动清理会报噪音异常，这里提前显式关闭
        atexit.register(self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.client.close()
        except Exception:
            pass

    # ==================================================================
    # 建库 / 重建
    # ==================================================================
    def ensure(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        vectors = {
            "dense": models.VectorParams(
                size=self.settings.embedding.dim, distance=models.Distance.COSINE
            )
        }
        sparse_vectors = None
        if self.sparse_enabled:
            sparse_vectors = {
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        self.client.create_collection(
            self.collection, vectors_config=vectors, sparse_vectors_config=sparse_vectors
        )

    def recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure()

    # ==================================================================
    # 写入
    # ==================================================================
    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    def upsert(
        self,
        chunks: list[Chunk],
        dense_vecs: np.ndarray,
        sparse_vecs: list[tuple[list[int], list[float]]] | None = None,
    ) -> None:
        points: list[models.PointStruct] = []
        for i, chunk in enumerate(chunks):
            vector: dict = {"dense": dense_vecs[i].tolist()}
            if self.sparse_enabled and sparse_vecs is not None:
                indices, values = sparse_vecs[i]
                if indices:
                    vector["sparse"] = models.SparseVector(indices=indices, values=values)
            points.append(
                models.PointStruct(id=chunk.id, vector=vector, payload=chunk.to_payload())
            )
        batch = 256
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points=points[i : i + batch])

    # ==================================================================
    # 检索
    # ==================================================================
    @staticmethod
    def _filter(doc_id: str | None) -> models.Filter | None:
        if not doc_id:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )

    def dense_search(
        self, vector: np.ndarray, limit: int, doc_id: str | None = None
    ) -> list[tuple[str, dict, float]]:
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            using="dense",
            limit=limit,
            query_filter=self._filter(doc_id),
            with_payload=True,
        )
        return [(p.id, p.payload or {}, p.score) for p in resp.points]

    def hybrid_search(
        self,
        dense_vector: np.ndarray,
        sparse_vector: tuple[list[int], list[float]],
        dense_limit: int,
        sparse_limit: int,
        limit: int,
        doc_id: str | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Qdrant 原生混合检索：dense + sparse 双 prefetch → RRF 融合。"""
        indices, values = sparse_vector
        if not indices:
            return self.dense_search(dense_vector, limit, doc_id)
        resp = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense_vector.tolist(), using="dense", limit=dense_limit),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=sparse_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            query_filter=self._filter(doc_id),
            with_payload=True,
        )
        return [(p.id, p.payload or {}, p.score) for p in resp.points]

    def get_payloads(self, ids: list[str]) -> dict[str, dict]:
        records = self.client.retrieve(self.collection, ids=ids, with_payload=True)
        return {r.id: (r.payload or {}) for r in records}

    # ==================================================================
    # 遍历 / 统计
    # ==================================================================
    def scroll_texts(self, batch: int = 512) -> tuple[list[str], list[str]]:
        """全量遍历 (ids, texts)，用于本地 BM25 索引重建。"""
        ids: list[str] = []
        texts: list[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=batch,
                offset=offset,
                with_payload=["text"],
                with_vectors=False,
            )
            for p in points:
                ids.append(p.id)
                texts.append((p.payload or {}).get("text", ""))
            if offset is None:
                break
        return ids, texts

    def stats(self) -> dict:
        count = self.client.count(self.collection, exact=True).count
        doc_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=512,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for p in points:
                doc_ids.add((p.payload or {}).get("doc_id", ""))
            if offset is None:
                break
        doc_ids.discard("")
        return {"chunks": count, "docs": len(doc_ids), "doc_ids": sorted(doc_ids)}