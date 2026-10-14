"""Qwen3-Embedding-0.6B 本地稠密编码器。

- 官方用法：transformers 直接加载 + last-token pooling + L2 归一化
- instruction-aware：查询侧加 "Instruct: {任务}\nQuery: {查询}" 前缀；文档侧不加前缀
- 设备自适应：CUDA 优先（FP16），CPU 兜底（FP32）
"""

from __future__ import annotations

import numpy as np

from ..config import Settings


def dtype_kwargs(dtype) -> dict:
    """transformers v5 起 `torch_dtype` 更名为 `dtype`，按版本兼容。"""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


class Qwen3Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.embedding
        self.device = settings.resolve_device()
        self.model_path = settings.resolve_model_path(self.cfg.model)
        self._tok = None
        self._model = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """按需加载模型（首次调用可能触发远程下载）。"""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
            self._model = AutoModel.from_pretrained(self.model_path, **dtype_kwargs(dtype))
        except Exception:
            # 部分模型需要信任远程代码
            self._tok = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, **dtype_kwargs(dtype)
            )
        self._model = self._model.to(self.device).eval()

    # ------------------------------------------------------------------
    def encode_documents(self, texts: list[str], batch_size: int | None = None, on_progress=None) -> np.ndarray:
        self.load()
        bs = batch_size or self.cfg.batch_size
        return self._encode(texts, bs, on_progress=on_progress)

    def encode_query(self, query: str) -> np.ndarray:
        self.load()
        prompt = f"Instruct: {self.cfg.instruct}\nQuery: {query}"
        return self._encode([prompt], self.cfg.batch_size)[0]

    # ------------------------------------------------------------------
    def _encode(self, texts: list[str], batch_size: int, on_progress=None) -> np.ndarray:
        import torch

        vecs: list[np.ndarray] = []
        total = len(texts)
        done = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self._model(**inputs).last_hidden_state[:, -1]
            out = torch.nn.functional.normalize(out.float()[:, : self.cfg.dim], p=2, dim=1)
            vecs.append(out.cpu().numpy())
            done += len(batch)
            if on_progress:
                on_progress(done, total)
        if not vecs:
            return np.zeros((0, self.cfg.dim), dtype=np.float32)
        return np.vstack(vecs)