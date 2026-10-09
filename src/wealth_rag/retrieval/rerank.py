"""Qwen3-Reranker-0.6B 本地重排器。

原理（关键）：decoder-only 判别式重排——不是文本生成，
而是取「yes / no」两个 token 在末位的 logits，经 softmax 得到相关性得分。
输入按官方 chat template 拼接 system + user（<Instruct>/<Query>/<Document>）。
"""

from __future__ import annotations

from ..config import Settings
from ..embedding.encoder import dtype_kwargs

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)

# 官方手动拼接格式（不使用 tokenizer.apply_chat_template：
# Qwen3 新版 chat template 会注入 think 脚手架并错误改写内容，导致 yes/no 打分失真）
_PREFIX = f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


class Qwen3Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.rerank
        self.device = settings.resolve_device()
        self.model_path = settings.resolve_model_path(self.cfg.model)
        self._tok = None
        self._model = None
        self._yes_id: int | None = None
        self._no_id: int | None = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.model_path, **dtype_kwargs(dtype))
            .to(self.device)
            .eval()
        )
        self._yes_id = self._tok.convert_tokens_to_ids("yes")
        self._no_id = self._tok.convert_tokens_to_ids("no")
        if self._yes_id is None or self._no_id is None:
            raise RuntimeError("未能解析 yes/no token id，请检查 tokenizer 是否为 Qwen3 系列")

    # ------------------------------------------------------------------
    def score(self, query: str, documents: list[str]) -> list[float]:
        """为 (query, document) 对逐条打分（越大越相关）。"""
        if not documents:
            return []
        self.load()
        import torch

        prompts = [self._format(query, doc) for doc in documents]
        scores: list[float] = []
        for i in range(0, len(prompts), self.cfg.batch_size):
            batch = prompts[i : i + self.cfg.batch_size]
            inputs = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :]
            yes = logits[:, self._yes_id]
            no = logits[:, self._no_id]
            probs = torch.softmax(torch.stack([no, yes], dim=1).float(), dim=1)[:, 1]
            scores.extend(probs.cpu().tolist())
        return scores

    # ------------------------------------------------------------------
    def _format(self, query: str, document: str) -> str:
        content = f"<Instruct>: {self.cfg.instruct}\n<Query>: {query}\n<Document>: {document}"
        return _PREFIX + content + _SUFFIX