"""摄入管线编排：发现文档 → 解析 → 切分 → 编码 → 入库。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..embedding.encoder import Qwen3Embedder
from ..sparse.bm25 import SparseEncoder
from ..store.qdrant_store import QdrantStore
from .chunker import MarkdownChunker
from .parser import DocumentParser, discover_files


@dataclass
class FileReport:
    path: str
    chunks: int
    error: str | None = None


@dataclass
class IngestReport:
    files: list[FileReport] = field(default_factory=list)
    total_chunks: int = 0
    rebuild: bool = False

    @property
    def ok_files(self) -> int:
        return sum(1 for f in self.files if f.error is None)


def ingest(
    settings: Settings,
    path: str | Path | None = None,
    rebuild: bool = False,
    on_file=None,
) -> IngestReport:
    """执行摄入。``on_file`` 为可选回调（用于 CLI 进度输出）。"""
    data_dir = Path(path) if path else settings.paths.data_dir
    if not data_dir.is_absolute():
        data_dir = (settings.root / data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files = discover_files(data_dir)
    report = IngestReport(rebuild=rebuild)
    if not files:
        return report

    store = QdrantStore(settings)
    if rebuild:
        store.recreate()
    else:
        store.ensure()

    sparse_encoder = SparseEncoder(settings.paths.db_dir / "sparse_vocab.json")
    if rebuild:
        sparse_encoder.reset()

    parser = DocumentParser(settings)
    chunker = MarkdownChunker(settings)
    embedder = Qwen3Embedder(settings)

    for f in files:
        rel = f.relative_to(data_dir).as_posix()
        try:
            parsed = parser.parse(f, data_dir)
            chunks = chunker.chunk(parsed.markdown, parsed.doc_id, parsed.source_path, parsed.title)
            if not rebuild:
                store.delete_doc(parsed.doc_id)  # 幂等覆盖（含已删除章节的清理）
            if chunks:
                dense = embedder.encode_documents([c.text for c in chunks])
                sparse_vecs = None
                if store.sparse_enabled:
                    sparse_vecs = [
                        sparse_encoder.encode(c.text, update_vocab=True) for c in chunks
                    ]
                store.upsert(chunks, dense, sparse_vecs)
            report.files.append(FileReport(path=rel, chunks=len(chunks)))
            report.total_chunks += len(chunks)
        except Exception as exc:  # 单文件失败不阻断整体摄入
            report.files.append(FileReport(path=rel, chunks=0, error=str(exc)))
        if on_file:
            on_file(report.files[-1])

    if store.sparse_enabled:
        sparse_encoder.save()
    return report