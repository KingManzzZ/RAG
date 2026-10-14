"""核心数据模型：Chunk（摄入单元）与 SearchResult（检索结果）。

Chunk 采用 mom/text 两级结构（对齐 RAGFlow TokenChunker 设计）：
- text 子块：参与嵌入与检索的最小单元；
- mom 父块：所属「节」的全文，存入子块元数据供后续上下文扩展，不单独入库。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str  # 参与嵌入的文本（含面包屑前缀）
    raw_text: str  # 原始文本（不含面包屑；表格保持 Markdown）
    doc_id: str
    doc_title: str
    source_path: str
    page: int | None
    heading_path: list[str] = field(default_factory=list)
    chunk_type: str = "text"  # text | table | table_continued
    parent_id: str | None = None  # mom 块引用
    parent_text: str | None = None  # mom 块全文（上下文扩展用）

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_path": self.source_path,
            "page": self.page,
            "heading_path": self.heading_path,
            "chunk_type": self.chunk_type,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
        }


@dataclass
class SearchResult:
    chunk_id: str
    score: float  # 最终排序得分（RRF，或 RRF+重排融合后的得分）
    rerank_score: float | None  # 重排原始分（未重排时为 None）
    text: str
    raw_text: str
    doc_id: str
    doc_title: str
    source_path: str
    page: int | None
    heading_path: list[str] = field(default_factory=list)
    chunk_type: str = "text"
    parent_id: str | None = None
    parent_text: str | None = None

    @property
    def citation(self) -> str:
        """可读的来源引用，供后续编排阶段直接拼接。"""
        loc = f"第 {self.page} 页" if self.page else ""
        path = " > ".join(self.heading_path)
        parts = [self.doc_title, loc, path]
        return " | ".join(p for p in parts if p)