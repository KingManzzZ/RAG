"""轻量检索评估：Recall@k / MRR / nDCG@k（文档级相关性判定）。

评估集格式（JSONL，每行一条）：

    {"query": "什么是再平衡策略？", "relevant": ["资产配置基础.md", "理财法规摘要.docx"]}

``relevant`` 条目与检索结果的 source_path（子串）/ doc_id / doc_title 任一匹配即视为命中。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_settings
from .retrieval.hybrid import Retriever


@dataclass
class QueryResult:
    query: str
    hit_rank: int | None
    rr: float
    ndcg: float
    relevant: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    top_k: int
    use_rerank: bool
    rows: list[QueryResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.hit_rank is not None) / len(self.rows)

    @property
    def mrr(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.rr for r in self.rows) / len(self.rows)

    @property
    def ndcg_at_k(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.ndcg for r in self.rows) / len(self.rows)


def load_queries(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"评估集不存在: {path}（参考 eval/queries.example.jsonl 的格式创建）"
        )
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        queries.append(json.loads(line))
    return queries


def is_relevant(result, relevant: list[str]) -> bool:
    for rel in relevant:
        if rel == result.doc_id or rel == result.doc_title:
            return True
        if rel and rel in result.source_path:
            return True
    return False


def evaluate(
    settings,
    queries_path: str | Path,
    top_k: int = 10,
    use_rerank: bool = True,
    retriever: Retriever | None = None,
) -> EvalReport:
    queries = load_queries(queries_path)
    retriever = retriever or Retriever(settings)
    report = EvalReport(top_k=top_k, use_rerank=use_rerank)

    for item in queries:
        results = retriever.search(item["query"], top_k=top_k, use_rerank=use_rerank)
        hit_rank: int | None = None
        for rank, res in enumerate(results, start=1):
            if is_relevant(res, item.get("relevant", [])):
                hit_rank = rank
                break
        rr = 0.0 if hit_rank is None else 1.0 / hit_rank
        ndcg = (
            0.0
            if hit_rank is None
            else 1.0 / math.log2(hit_rank + 1)  # 单一相关文档时 IDCG=1
        )
        report.rows.append(
            QueryResult(
                query=item["query"], hit_rank=hit_rank, rr=rr, ndcg=ndcg,
                relevant=item.get("relevant", []),
            )
        )
    return report


def render_report(report: EvalReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"检索评估（top-{report.top_k}，rerank={'开' if report.use_rerank else '关'}）")
    table.add_column("查询", max_width=42, overflow="ellipsis")
    table.add_column("命中排名", justify="right")
    table.add_column("RR", justify="right")
    table.add_column("nDCG", justify="right")
    for row in report.rows:
        table.add_row(
            row.query,
            "-" if row.hit_rank is None else str(row.hit_rank),
            f"{row.rr:.3f}",
            f"{row.ndcg:.3f}",
        )
    console.print(table)
    console.print(
        f"[bold]Recall@{report.top_k} = {report.recall_at_k:.3f}   "
        f"MRR = {report.mrr:.3f}   nDCG@{report.top_k} = {report.ndcg_at_k:.3f}   "
        f"（{len(report.rows)} 条查询）[/bold]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="检索评估：Recall@k / MRR / nDCG@k")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--queries", default="eval/queries.jsonl", help="评估集 JSONL 路径")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排再评估")
    args = parser.parse_args()

    settings = load_settings(args.config)
    report = evaluate(
        settings, args.queries, top_k=args.top_k, use_rerank=not args.no_rerank
    )
    render_report(report)


if __name__ == "__main__":
    main()