from __future__ import annotations


class EmbeddingEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # TODO: load embedding model/tokenizer once at startup.

    async def encode(self, text: str) -> list[float]:
        # TODO: generate dense embedding vector for text.
        raise NotImplementedError
