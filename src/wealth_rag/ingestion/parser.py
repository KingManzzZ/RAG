"""文档解析：统一产出「标准 Markdown + 结构元数据」。

- PDF / Word / PPT / Excel / HTML → Docling（MIT；统一多格式，保留标题层级与表格结构）
- Markdown / TXT → 直接读取

页码溯源：PDF 经 Docling 导出时插入 page_break_placeholder 标记，
由切分器按出现次序换算页码（1 起）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings

SUPPORTED_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".md",
    ".markdown",
    ".txt",
    ".html",
    ".htm",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    markdown: str
    source_path: str  # 相对 data_dir 的可读相对路径（正斜杠）


def make_doc_id(rel_path: str) -> str:
    """文档稳定 ID：相对路径的哈希（同一文件重复摄入时保持不变）。"""
    norm = rel_path.replace("\\", "/").lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]


class DocumentParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._converter = None  # Docling 按需加载（首次使用会下载版面分析模型）

    # ------------------------------------------------------------------
    def parse(self, path: Path, data_dir: Path) -> ParsedDoc:
        try:
            rel = path.resolve().relative_to(data_dir.resolve()).as_posix()
        except ValueError:
            rel = path.name
        doc_id = make_doc_id(rel)

        ext = path.suffix.lower()
        if ext in {".md", ".markdown", ".txt"}:
            markdown = path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf" and self.settings.parsing.pdf_engine in {"pymupdf", "auto"}:
            markdown = self._convert_pdf(path)
        else:
            markdown = self._convert_with_docling(path)

        title = self._extract_title(markdown) or path.stem
        return ParsedDoc(doc_id=doc_id, title=title, markdown=markdown, source_path=rel)

    # ------------------------------------------------------------------
    def _convert_with_docling(self, path: Path) -> str:
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        result = self._converter.convert(str(path))
        doc = result.document
        try:
            return doc.export_to_markdown(
                page_break_placeholder=self.settings.parsing.page_break_placeholder
            )
        except TypeError:  # 旧版 Docling 不支持占位符
            return doc.export_to_markdown()

    def _convert_pdf(self, path: Path) -> str:
        """PDF 解析：pymupdf 引擎直接走快速通道；auto 引擎先试 Docling、失败自动回退。"""
        if self.settings.parsing.pdf_engine == "pymupdf":
            return self._convert_with_pymupdf(path)
        try:
            return self._convert_with_docling(path)
        except Exception as exc:  # 模型不可用/网络异常 → 回退快速通道
            import logging

            logging.getLogger(__name__).warning(
                "Docling 解析 PDF 失败（%s），回退 PyMuPDF 快速通道: %s", type(exc).__name__, exc
            )
            return self._convert_with_pymupdf(path)

    def _convert_with_pymupdf(self, path: Path) -> str:
        """PyMuPDF 快速通道（备选）：逐页转 Markdown，插入页码标记。"""
        import pymupdf4llm

        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        marker = self.settings.parsing.page_break_placeholder
        parts = []
        for page in pages:
            text = page["text"] if isinstance(page, dict) else str(page)
            parts.append(f"{marker}\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_title(markdown: str) -> str:
        for line in markdown.split("\n"):
            m = _HEADING_RE.match(line.strip())
            if m:
                return m.group(2).strip()
            if line.strip():
                break
        return ""


def discover_files(data_dir: Path) -> list[Path]:
    """递归发现支持的知识文档（按路径排序，保证摄入顺序稳定）。"""
    files = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("~$")
    ]
    return sorted(files)