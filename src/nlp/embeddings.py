from __future__ import annotations

import asyncio
import importlib.util

from src.utils.text_cleaner import normalize_for_matching
from src.utils.vector_math import hashed_text_vector


class EmbeddingEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_error: Exception | None = None

    async def encode(self, text: str) -> list[float]:
        if self._should_offload():
            return await asyncio.to_thread(self._encode_sync, text)
        return self._encode_sync(text)

    async def encode_many(self, texts: list[str]) -> list[list[float]]:
        if self._should_offload():
            return await asyncio.to_thread(self._encode_many_sync, texts)
        return self._encode_many_sync(texts)

    def _should_offload(self) -> bool:
        return (
            self.model_name not in {"", "local-hash"}
            and importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _encode_sync(self, text: str) -> list[float]:
        vectors = self._encode_many_sync([text])
        return vectors[0] if vectors else hashed_text_vector(text)

    def _encode_many_sync(self, texts: list[str]) -> list[list[float]]:
        normalized = [normalize_for_matching(text) for text in texts]
        if self.model_name not in {"", "local-hash"}:
            transformer_vectors = self._encode_with_transformer(normalized)
            if transformer_vectors is not None:
                return transformer_vectors
        return [hashed_text_vector(text) for text in normalized]

    def _ensure_transformer_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._torch = torch
            return True
        except Exception as exc:  # pragma: no cover - depends on optional model availability.
            self._load_error = exc
            self._tokenizer = None
            self._model = None
            self._torch = None
            return False

    def _encode_with_transformer(self, texts: list[str]) -> list[list[float]] | None:
        if not self._ensure_transformer_loaded():
            return None

        torch = self._torch
        with torch.no_grad():
            tokens = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            output = self._model(**tokens)
            hidden = output.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            vectors = summed / counts
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return [[float(value) for value in row] for row in vectors.cpu().tolist()]
