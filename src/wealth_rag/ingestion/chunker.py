"""结构感知切分器（核心模块）。

实现来源：源码级模仿 RAGFlow 并适配（详见 PLAN.md 4.2），六步策略：

① 结构切分   Markdown 标题树 → 「节」（对齐 rag/flow/chunker/title_chunker）
② 节内切分   split-then-merge，贪心合并至 max_tokens（对齐 rag/app/naive.py::naive_merge）
③ 超限兜底   单句超限 → 硬切分（对齐 TitleChunker hard split）
④ 表格处理   原子块；超限按行切分 + 表头重复 + 续表标记（对齐 rag/app/table.py）
⑤ 父子块组装 mom（节全文）/ text（子块）两级结构（对齐 TokenChunker 的 mom/text）
⑥ 元数据组装 面包屑注入（我方适配）+ 页码溯源 + 金融实体保护（我方新增）
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..config import Settings
from ..models import Chunk
from .tokenizer import TokenCounter

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MARKER_RE = re.compile(r"^<!--\s*(?P<tag>PAGEBREAK|page:\s*(?P<num>\d+))\s*-->$", re.IGNORECASE)

# 金融实体保护（⑥）：块尾「悬空实体」与下一段「实体延续」模式
_END_DANGLING = re.compile(r"(第\s*\d+|\d+(?:\.\d+)?)\s*$")
_NEXT_CONT = re.compile(r"^\s*(条|款|项|章|节|条第一款|％|%|年|月|日|元|万元|亿元|个基点|BP|bp)")

# 英文句子边界（③ 超限兜底用）：句末标点 + 空白 + 大写/数字起始。
# 要求句号后必须跟空白，因此「4.5%」「U.S. market」等不会被误切；中文句末标点不在此列，不受影响。
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])')


@dataclass
class _Block:
    kind: str  # para | table
    content: str
    page: int | None


@dataclass
class _Section:
    heading_path: list[str]
    blocks: list[_Block] = field(default_factory=list)


@dataclass
class _Piece:
    """切分产物（面包屑在组装阶段统一注入）。"""

    embed_text: str
    raw_text: str
    chunk_type: str
    page: int | None


def _uuid_from(text: str) -> str:
    """md5 → 标准 UUID 字符串（Qdrant 点 ID 约束），同一输入稳定不变。"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class MarkdownChunker:
    def __init__(self, settings: Settings):
        self.cfg = settings.chunking
        self.counter = TokenCounter(settings.paths.cache_dir)
        self._delim_set = set(self.cfg.delimiters)

    # ==================================================================
    # 对外入口
    # ==================================================================
    def chunk(self, markdown: str, doc_id: str, source_path: str, doc_title: str) -> list[Chunk]:
        sections = self._parse_sections(markdown)
        chunks: list[Chunk] = []
        for si, sec in enumerate(sections):
            pieces = self._section_pieces(sec)
            if not pieces:
                continue
            breadcrumb = " > ".join(sec.heading_path) if self.cfg.breadcrumb else ""
            prefix = f"{breadcrumb}\n" if breadcrumb else ""
            # ⑤ 父子块：本节产出多个子块时，节全文作为 mom 块存入每个子块元数据
            parent_id = f"{doc_id}-s{si}" if len(pieces) > 1 else None
            parent_text = (
                "\n\n".join(b.content for b in sec.blocks).strip() if parent_id else None
            )
            for piece in pieces:
                embed_text = (prefix + piece.embed_text).strip()
                if not embed_text:
                    continue
                idx = len(chunks)
                # 点 ID 需为 UUID 格式（Qdrant 约束），由 md5 稳定派生
                chunk_id = _uuid_from(f"{doc_id}|{idx}")
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=embed_text,
                        raw_text=piece.raw_text.strip(),
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_path=source_path,
                        page=piece.page,
                        heading_path=list(sec.heading_path),
                        chunk_type=piece.chunk_type,
                        parent_id=parent_id,
                        parent_text=parent_text,
                    )
                )
        return chunks

    # ==================================================================
    # ① 结构切分：Markdown → 标题树 → 节
    # ==================================================================
    def _parse_sections(self, md: str) -> list[_Section]:
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []
        current = _Section(heading_path=[])
        sections.append(current)

        lines = md.split("\n")
        n = len(lines)
        in_code = False
        page: int | None = None
        breaks = 0

        buf: list[str] = []
        buf_page: int | None = None

        def flush_para() -> None:
            nonlocal buf, buf_page
            text = "\n".join(buf).strip()
            if text:
                current.blocks.append(_Block("para", text, buf_page))
            buf = []
            buf_page = None

        i = 0
        while i < n:
            line = lines[i]
            stripped = line.strip()

            # 页码标记（PDF 由 Docling 插入）
            if not in_code:
                m = _MARKER_RE.match(stripped)
                if m:
                    if m.group("num"):
                        page = int(m.group("num"))
                    else:
                        breaks += 1
                        page = breaks + 1
                    i += 1
                    continue

            # 代码块整体视为普通段落内容
            if stripped.startswith("```"):
                in_code = not in_code
                if not buf:
                    buf_page = page
                buf.append(line)
                i += 1
                continue

            if not in_code:
                # 标题 → 开新节
                m = _HEADING_RE.match(line.strip())
                if m and len(m.group(1)) <= self.cfg.heading_max_level:
                    flush_para()
                    level = len(m.group(1))
                    title = m.group(2).strip()
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    current = _Section(heading_path=[t for _, t in stack])
                    sections.append(current)
                    i += 1
                    continue

                # 表格块
                if self._is_table_start(lines, i):
                    flush_para()
                    rows: list[str] = []
                    while i < n and lines[i].strip().startswith("|"):
                        rows.append(lines[i].strip())
                        i += 1
                    current.blocks.append(_Block("table", "\n".join(rows), page))
                    continue

                # 空行 → 段落边界
                if not stripped:
                    flush_para()
                    i += 1
                    continue

            if not buf:
                buf_page = page
            buf.append(line)
            i += 1

        flush_para()
        return [s for s in sections if s.blocks]

    @staticmethod
    def _is_table_start(lines: list[str], i: int) -> bool:
        line = lines[i].strip()
        if not line.startswith("|") or i + 1 >= len(lines):
            return False
        nxt = lines[i + 1].strip()
        if not nxt.startswith("|"):
            return False
        return "-" in nxt and set(nxt) <= set("|-: ")

    # ==================================================================
    # ②③④ 节内处理
    # ==================================================================
    def _section_pieces(self, sec: _Section) -> list[_Piece]:
        """把节内块序列转为切分产物：连续段落合并处理，表格独立处理。"""
        pieces: list[_Piece] = []
        run: list[_Block] = []

        def flush_run() -> None:
            nonlocal run
            if run:
                text = "\n\n".join(b.content for b in run)
                pieces.extend(self._text_pieces(text, run[0].page))
                run = []

        for block in sec.blocks:
            if block.kind == "para":
                run.append(block)
            else:
                flush_run()
                pieces.extend(self._table_pieces(block))
        flush_run()
        return pieces

    # ---------------- ② + ③ 文本：split-then-merge ----------------
    def _text_pieces(self, text: str, page: int | None) -> list[_Piece]:
        max_tokens = self.cfg.max_tokens
        if self.counter.count(text) <= max_tokens:
            return [_Piece(text, text, "text", page)]

        segments = self._split_segments(text)
        # ③ 超限兜底：单段仍超限 → 先按句子边界拆分（英文），单句仍超限再硬切分
        expanded: list[str] = []
        for seg in segments:
            if self.counter.count(seg) > max_tokens:
                expanded.extend(self._expand_oversized(seg, max_tokens))
            else:
                expanded.append(seg)

        groups = self._merge_segments(expanded)
        return [_Piece(g, g, "text", page) for g in groups]

    def _expand_oversized(self, seg: str, max_tokens: int) -> list[str]:
        """超限单元展开：优先按句子边界拆分（英文），单句仍超限时才硬切分。
        展开后的单元由 _merge_segments 按上限重新贪心合并，保证块边界落在句子上。"""
        sentences = self._split_sentences(seg)
        if len(sentences) <= 1:
            return self.counter.hard_split(seg, max_tokens)
        units: list[str] = []
        for sentence in sentences:
            if self.counter.count(sentence) > max_tokens:
                units.extend(self.counter.hard_split(sentence, max_tokens))
            else:
                units.append(sentence)
        return units

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按英文句子边界拆分，句间空白归属前句（与 _split_segments 同思路）。"""
        segments: list[str] = []
        last = 0
        for match in _SENT_BOUNDARY.finditer(text):
            segments.append(text[last : match.end()])
            last = match.end()
        if last < len(text):
            segments.append(text[last:])
        return [s for s in segments if s.strip()]

    def _split_segments(self, text: str) -> list[str]:
        """按分隔符切分，分隔符归属前一段（对齐 naive 的 delimiter 切分）。"""
        delim = "".join(re.escape(c) for c in sorted(self._delim_set))
        parts = re.split(f"([{delim}])", text)
        segments: list[str] = []
        buf = ""
        for part in parts:
            if not part:
                continue
            if len(part) == 1 and part in self._delim_set:
                buf += part
                segments.append(buf)
                buf = ""
            else:
                buf += part
        if buf.strip():
            segments.append(buf)
        return [s for s in segments if s.strip()]

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """贪心合并至 max_tokens（对齐 naive_merge）；合并边界做金融实体保护。"""
        max_tokens = self.cfg.max_tokens
        pull_limit = int(max_tokens * 1.25)
        groups: list[str] = []
        cur = ""
        cur_tokens = 0
        for seg in segments:
            t = self.counter.count(seg)
            if cur and cur_tokens + t > max_tokens:
                # ⑥ 实体保护：不把「第 24 | 条」「4.5 | %」拆到两块
                if self._should_pull(cur, seg) and cur_tokens + t <= pull_limit:
                    cur += seg
                    cur_tokens += t
                    continue
                groups.append(cur)
                cur, cur_tokens = seg, t
            else:
                cur += seg
                cur_tokens += t
        if cur.strip():
            groups.append(cur)
        return [g.strip() for g in groups if g.strip()]

    @staticmethod
    def _should_pull(cur: str, nxt: str) -> bool:
        return bool(_END_DANGLING.search(cur.rstrip())) and bool(_NEXT_CONT.match(nxt))

    # ---------------- ④ 表格处理 ----------------
    def _table_pieces(self, block: _Block) -> list[_Piece]:
        max_tokens = self.cfg.max_tokens
        lines = block.content.split("\n")
        rows = [self._split_row(ln) for ln in lines]
        # 去掉 Markdown 对齐行（|---|---|）
        header = rows[0] if rows else []
        data = [r for idx, r in enumerate(rows) if idx > 0 and not self._is_separator(r)]
        if not header:
            return []

        if self.counter.count(block.content) <= max_tokens:
            return [_Piece(self._textify_table(header, data), block.content, "table", block.page)]

        # 行级切分 + 表头重复 + 续表标记
        header_md = self._row_to_md(header)
        sep_md = "|" + "|".join("---" for _ in header) + "|"
        base_tokens = self.counter.count(header_md) + self.counter.count(sep_md)

        pieces: list[_Piece] = []
        cur_rows: list[list[str]] = []
        cur_tokens = 0

        def flush(first: bool) -> None:
            nonlocal cur_rows, cur_tokens
            if not cur_rows:
                return
            raw = "\n".join([header_md, sep_md] + [self._row_to_md(r) for r in cur_rows])
            ctype = "table" if first else "table_continued"
            pieces.append(_Piece(self._textify_table(header, cur_rows), raw, ctype, block.page))
            cur_rows, cur_tokens = [], 0

        for row in data:
            t = self.counter.count(self._row_to_md(row))
            if cur_rows and base_tokens + cur_tokens + t > max_tokens:
                flush(first=len(pieces) == 0)
            # 单行本身超限（极端）：硬拆该行
            if t > max_tokens:
                flush(first=len(pieces) == 0)
                for piece_text in self.counter.hard_split(self._row_to_md(row), max_tokens):
                    ctype = "table" if not pieces else "table_continued"
                    pieces.append(_Piece(piece_text, piece_text, ctype, block.page))
                continue
            cur_rows.append(row)
            cur_tokens += t
        flush(first=len(pieces) == 0)

        if not pieces:  # 仅有表头
            return [_Piece(self._textify_table(header, []), block.content, "table", block.page)]
        return pieces

    @staticmethod
    def _split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    @staticmethod
    def _is_separator(cells: list[str]) -> bool:
        joined = "".join(cells)
        return bool(joined) and set(joined) <= set("-: ") and "-" in joined

    @staticmethod
    def _row_to_md(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    @staticmethod
    def _textify_table(header: list[str], rows: list[list[str]]) -> str:
        """表格文本化（供嵌入与关键词匹配）：「表头: 值」逐行拼接。"""
        lines = []
        for row in rows:
            pairs = []
            for j, cell in enumerate(row):
                if not cell:
                    continue
                head = header[j] if j < len(header) and header[j] else f"列{j + 1}"
                pairs.append(f"{head}: {cell}")
            if pairs:
                lines.append("；".join(pairs))
        if not lines and header:
            lines.append("；".join(h for h in header if h))
        return "\n".join(lines)