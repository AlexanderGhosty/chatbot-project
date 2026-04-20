from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalResult:
    answer_text: str
    distance: float
    matched_question: str | None = None


class VectorDatabase:
    def __init__(self, db_path: str, collection_name: str) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        # TODO: initialize ChromaDB client and collection handle.

    async def search_answer(self, embedding: list[float], top_k: int = 1) -> RetrievalResult:
        # TODO: query ChromaDB by embedding and map result object.
        raise NotImplementedError
