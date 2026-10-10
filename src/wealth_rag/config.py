"""配置加载：config.yaml → pydantic 模型（路径均相对配置文件的目录解析）。"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_NAME = "config.yaml"


class PathsConfig(BaseModel):
    data_dir: Path = Path("./data")
    db_dir: Path = Path("./db")
    models_dir: Path = Path("./models")
    cache_dir: Path = Path("./cache")


class EmbeddingConfig(BaseModel):
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    dim: int = 1024
    max_length: int = 8192
    batch_size: int = 8
    instruct: str = (
        "Given a financial question, retrieve relevant passages that answer the question"
    )


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 4096
    batch_size: int = 8
    instruct: str = "Given a financial query, retrieve passages that answer the query"


class SparseConfig(BaseModel):
    method: str = "qdrant_sparse"  # qdrant_sparse | jieba_bm25
    k1: float = 1.5
    b: float = 0.75


class RetrievalConfig(BaseModel):
    dense_top: int = 30
    sparse_top: int = 30
    fused_top: int = 20
    final_top: int = 8
    rrf_k: int = 60


class ChunkingConfig(BaseModel):
    max_tokens: int = 512
    delimiters: str = "\n!?。；！？"
    heading_max_level: int = 4
    overlap_tokens: int = 0
    breadcrumb: bool = True


class ParsingConfig(BaseModel):
    page_break_placeholder: str = "<!--PAGEBREAK-->"
    pdf_engine: str = "auto"  # auto（Docling 优先，失败回退 PyMuPDF）| docling | pymupdf


class Settings(BaseModel):
    """全局配置。"""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    device: str = "auto"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)

    #: 配置文件所在目录（运行期填充，不参与序列化）
    root: Path = Field(default=Path("."), exclude=True)

    def resolve(self) -> "Settings":
        """把相对路径解析为绝对路径，并创建必要目录。"""
        for name in ("data_dir", "db_dir", "models_dir", "cache_dir"):
            p: Path = getattr(self.paths, name)
            if not p.is_absolute():
                p = (self.root / p).resolve()
            setattr(self.paths, name, p)
        self.paths.db_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def resolve_device(self) -> str:
        """auto → 有 CUDA 用 cuda，否则 cpu。"""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def resolve_model_path(self, model: str) -> str:
        """优先使用 models_dir 下的本地模型目录，否则返回原始模型名（HF/ModelScope 拉取）。"""
        local = self.paths.models_dir / model.split("/")[-1]
        return str(local) if local.exists() else model


def load_settings(config_path: str | Path | None = None) -> Settings:
    """加载配置文件。默认读取项目根目录 config.yaml，可用环境变量 WEALTH_RAG_CONFIG 指定。"""
    if config_path is None:
        config_path = os.environ.get("WEALTH_RAG_CONFIG", DEFAULT_CONFIG_NAME)
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings(**raw)
    settings.root = path.parent
    return settings.resolve()