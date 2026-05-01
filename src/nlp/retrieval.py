from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.text_cleaner import normalize_for_matching
from src.utils.vector_math import cosine_distance

if TYPE_CHECKING:
    from src.nlp.embeddings import EmbeddingEngine


@dataclass(slots=True)
class RetrievalResult:
    answer_text: str
    distance: float
    matched_question: str | None = None


@dataclass(slots=True)
class _DialogueRecord:
    record_id: str
    question: str
    answer: str
    embedding: list[float]


class VectorDatabase:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        dialogues_path: str = "data/raw/dialogues.txt",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.dialogues_path = Path(dialogues_path)
        self._client = None
        self._collection = None
        self._records: list[_DialogueRecord] = []
        self._ready = False
        self._lock = asyncio.Lock()
        self._init_chroma()

    async def ensure_ready(self, embedding_engine: "EmbeddingEngine") -> None:
        if self._ready:
            return

        async with self._lock:
            if self._ready:
                return

            if self._collection is not None:
                try:
                    if self._collection.count() > 0:
                        self._ready = True
                        return
                except Exception:
                    self._collection = None

            if self._collection is None and self._load_local_index(embedding_engine.model_name):
                self._ready = True
                return

            pairs = load_dialogue_pairs(self.dialogues_path)
            questions = [question for question, _answer in pairs]
            embeddings = await embedding_engine.encode_many(questions)
            self._records = [
                _DialogueRecord(
                    record_id=f"dialogue-{index}",
                    question=question,
                    answer=answer,
                    embedding=embeddings[index],
                )
                for index, (question, answer) in enumerate(pairs)
            ]
            self._persist_records(embedding_engine.model_name)
            self._populate_chroma()
            self._ready = True

    async def search_answer(self, embedding: list[float], top_k: int = 1) -> RetrievalResult:
        if self._collection is not None:
            return await asyncio.to_thread(self._search_sync, embedding, top_k)
        return self._search_sync(embedding, top_k)

    def _init_chroma(self) -> None:
        try:
            import chromadb

            path = Path(self.db_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:  # pragma: no cover - optional dependency / runtime.
            self._client = None
            self._collection = None

    def _populate_chroma(self) -> None:
        if self._collection is None or not self._records:
            return

        try:
            if self._collection.count() > 0:
                return
            self._collection.add(
                ids=[record.record_id for record in self._records],
                embeddings=[record.embedding for record in self._records],
                documents=[record.answer for record in self._records],
                metadatas=[{"question": record.question} for record in self._records],
            )
        except Exception:
            self._collection = None

    def _search_sync(self, embedding: list[float], top_k: int) -> RetrievalResult:
        if self._collection is not None:
            result = self._search_chroma(embedding, top_k)
            if result is not None:
                return result

        if not self._records:
            return RetrievalResult(
                answer_text="Я пока не нашел подходящий ответ. Могу помочь подобрать мебель из каталога.",
                distance=1.0,
            )

        best = min(self._records, key=lambda record: cosine_distance(embedding, record.embedding))
        return RetrievalResult(
            answer_text=best.answer,
            distance=cosine_distance(embedding, best.embedding),
            matched_question=best.question,
        )

    def _search_chroma(self, embedding: list[float], top_k: int) -> RetrievalResult | None:
        try:
            raw = self._collection.query(
                query_embeddings=[embedding],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )
            documents = raw.get("documents") or [[]]
            distances = raw.get("distances") or [[]]
            metadatas = raw.get("metadatas") or [[]]
            if not documents[0]:
                return None
            question = None
            if metadatas[0]:
                question = metadatas[0][0].get("question")
            return RetrievalResult(
                answer_text=documents[0][0],
                distance=float(distances[0][0]) if distances[0] else 1.0,
                matched_question=question,
            )
        except Exception:
            return None

    def _index_path(self) -> Path:
        return Path(self.db_path) / f"{self.collection_name}_fallback_index.json"

    def _load_local_index(self, embedding_model_name: str) -> bool:
        path = self._index_path()
        if not path.exists():
            return False

        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False

        if payload.get("embedding_model_name") != embedding_model_name:
            return False

        records = []
        for item in payload.get("records", []):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            embedding = item.get("embedding")
            if question and answer and isinstance(embedding, list):
                records.append(
                    _DialogueRecord(
                        record_id=str(item.get("id", f"dialogue-{len(records)}")),
                        question=question,
                        answer=answer,
                        embedding=[float(value) for value in embedding],
                    )
                )
        self._records = records
        return bool(self._records)

    def _persist_records(self, embedding_model_name: str) -> None:
        path = self._index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model_name": embedding_model_name,
            "records": [
                {
                    "id": record.record_id,
                    "question": record.question,
                    "answer": record.answer,
                    "embedding": record.embedding,
                }
                for record in self._records
            ],
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)


def load_dialogue_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    if path.exists():
        content = path.read_text(encoding="utf-8")
        for block in content.split("\n\n"):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue

            question = _strip_dialogue_prefix(lines[0])
            answer = _strip_dialogue_prefix(lines[1])
            normalized_question = normalize_for_matching(question)
            if not normalized_question or normalized_question in seen:
                continue
            if len(normalized_question) < 3 or len(answer) < 2:
                continue
            seen.add(normalized_question)
            pairs.append((normalized_question, answer.strip()))

    for question, answer in _seed_dialogues():
        normalized_question = normalize_for_matching(question)
        if normalized_question not in seen:
            pairs.append((normalized_question, answer))
            seen.add(normalized_question)

    return pairs


def _strip_dialogue_prefix(line: str) -> str:
    cleaned = line.strip()
    for prefix in ("- ", "— ", "– ", "q:", "Q:", "вопрос:", "Вопрос:", "a:", "A:", "ответ:", "Ответ:"):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :].strip()
    return cleaned


def _seed_dialogues() -> list[tuple[str, str]]:
    return [
        ("как дела", "Спасибо, хорошо. Готов помочь с вопросами и подбором мебели."),
        ("что ты умеешь", "Я отвечаю на вопросы, поддерживаю диалог и помогаю подобрать мебель для дома."),
        ("какую мебель выбрать для маленькой комнаты", "Для маленькой комнаты лучше выбирать компактные модели: диван-кровать, узкий шкаф и стол с легкой конструкцией."),
        ("есть ли доставка", "Да, можно оформить доставку по городу и согласовать удобный интервал."),
        ("как ухаживать за деревянным столом", "Деревянный стол лучше протирать мягкой влажной тканью и не ставить горячее без подставки."),
        ("как выбрать шкаф", "Ориентируйтесь на ширину ниши, высоту потолка, глубину полок и наличие зеркала."),
        ("какой диван подойдет для гостиной", "Для гостиной хорошо подходит диван с прочной обивкой, удобной посадкой и запасом по ширине."),
        ("можно ли оплатить при получении", "Да, для части заказов доступна оплата при получении. Точные условия зависят от способа доставки."),
    ]
