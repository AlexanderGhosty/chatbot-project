from __future__ import annotations

import asyncio
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.utils.text_cleaner import normalize_for_matching
from src.utils.vector_math import cosine_similarity, hashed_text_vector


@dataclass(slots=True)
class IntentResult:
    label: str
    confidence: float


@dataclass(slots=True)
class SentimentResult:
    label: str
    confidence: float


@dataclass(slots=True)
class _IntentData:
    label: str
    examples: list[str]
    responses: list[str]


class IntentClassifier:
    def __init__(self, model_name: str, intents_path: str = "data/raw/intents.json") -> None:
        self.model_name = model_name
        self.intents_path = Path(intents_path)
        self._intents = self._load_intents()
        self._sklearn_pipeline = self._build_sklearn_pipeline()
        self._centroids = self._build_centroids()
        self._example_index = self._build_example_index()

    async def predict(self, text: str) -> IntentResult:
        return self._predict_sync(text)

    def get_random_response(self, label: str) -> str | None:
        data = self._intents.get(label)
        if not data or not data.responses:
            return None
        return random.choice(data.responses)

    def _predict_sync(self, text: str) -> IntentResult:
        normalized = normalize_for_matching(text)
        if not normalized:
            return IntentResult(label="unknown", confidence=0.0)

        exact = self._example_index.get(normalized)
        if exact:
            return IntentResult(label=exact, confidence=1.0)

        if self._sklearn_pipeline is not None:
            label, confidence = self._predict_sklearn(normalized)
        else:
            label, confidence = self._predict_centroid(normalized)

        if confidence < 0.32:
            return IntentResult(label="unknown", confidence=confidence)
        return IntentResult(label=label, confidence=confidence)

    def _load_intents(self) -> dict[str, _IntentData]:
        if not self.intents_path.exists():
            return _default_intents()

        with self.intents_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        raw_intents = payload.get("intents", payload)
        intents: dict[str, _IntentData] = {}
        for label, data in raw_intents.items():
            examples = [normalize_for_matching(item) for item in data.get("examples", []) if item.strip()]
            responses = [item.strip() for item in data.get("responses", []) if item.strip()]
            if examples:
                intents[label] = _IntentData(label=label, examples=examples, responses=responses)
        return intents or _default_intents()

    def _build_sklearn_pipeline(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError:
            return None

        examples: list[str] = []
        labels: list[str] = []
        for intent in self._intents.values():
            examples.extend(intent.examples)
            labels.extend([intent.label] * len(intent.examples))

        if len(set(labels)) < 2:
            return None

        pipeline = Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        pipeline.fit(examples, labels)
        return pipeline

    def _predict_sklearn(self, normalized: str) -> tuple[str, float]:
        probabilities = self._sklearn_pipeline.predict_proba([normalized])[0]
        classes = list(self._sklearn_pipeline.named_steps["classifier"].classes_)
        best_index = max(range(len(classes)), key=lambda index: probabilities[index])
        return classes[best_index], float(probabilities[best_index])

    def _build_centroids(self) -> dict[str, list[float]]:
        centroids: dict[str, list[float]] = {}
        for intent in self._intents.values():
            vectors = [hashed_text_vector(example) for example in intent.examples]
            if not vectors:
                continue
            dimensions = len(vectors[0])
            centroid = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]
            centroids[intent.label] = centroid
        return centroids

    def _predict_centroid(self, normalized: str) -> tuple[str, float]:
        vector = hashed_text_vector(normalized)
        scored = [
            (label, cosine_similarity(vector, centroid))
            for label, centroid in self._centroids.items()
            if label != "unknown"
        ]
        if not scored:
            return "unknown", 0.0
        label, score = max(scored, key=lambda item: item[1])
        confidence = max(0.0, min(1.0, score))
        return label, confidence

    def _build_example_index(self) -> dict[str, str]:
        return {
            example: intent.label
            for intent in self._intents.values()
            for example in intent.examples
        }


class SentimentClassifier:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._pipeline_error: Exception | None = None

    async def predict(self, text: str) -> SentimentResult:
        if self.model_name not in {"", "local-lexicon"} and importlib.util.find_spec("transformers"):
            return await asyncio.to_thread(self._predict_sync, text)
        return self._predict_sync(text)

    def _predict_sync(self, text: str) -> SentimentResult:
        normalized = normalize_for_matching(text)
        if not normalized:
            return SentimentResult(label="neutral", confidence=0.0)

        if self.model_name not in {"", "local-lexicon"}:
            result = self._predict_with_transformers(normalized)
            if result is not None:
                return result

        positive_words = {
            "отлично",
            "хорошо",
            "классно",
            "нравится",
            "спасибо",
            "супер",
            "удобно",
            "красиво",
            "интересно",
        }
        negative_words = {
            "плохо",
            "ужасно",
            "дорого",
            "ненавижу",
            "раздражает",
            "недоволен",
            "проблема",
            "неудобно",
            "не нравится",
        }

        positive = sum(1 for word in positive_words if word in normalized)
        negative = sum(1 for word in negative_words if word in normalized)
        if positive == negative:
            return SentimentResult(label="neutral", confidence=0.55)
        if positive > negative:
            return SentimentResult(label="positive", confidence=min(0.95, 0.6 + 0.15 * positive))
        return SentimentResult(label="negative", confidence=min(0.95, 0.6 + 0.15 * negative))

    def _predict_with_transformers(self, normalized: str) -> SentimentResult | None:
        try:
            if self._pipeline is None and self._pipeline_error is None:
                from transformers import pipeline

                self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
            if self._pipeline is None:
                return None
            raw = self._pipeline(normalized, truncation=True)[0]
        except Exception as exc:  # pragma: no cover - depends on optional model availability.
            self._pipeline_error = exc
            return None

        label = str(raw.get("label", "neutral")).lower()
        confidence = float(raw.get("score", 0.0))
        if "pos" in label or "positive" in label:
            mapped = "positive"
        elif "neg" in label or "negative" in label:
            mapped = "negative"
        else:
            mapped = "neutral"
        return SentimentResult(label=mapped, confidence=confidence)


def _default_intents() -> dict[str, _IntentData]:
    raw = {
        "greeting": {
            "examples": ["привет", "здравствуйте", "добрый день", "добрый вечер"],
            "responses": ["Здравствуйте! Чем могу помочь по мебели?"],
        },
        "farewell": {
            "examples": ["пока", "до свидания", "увидимся"],
            "responses": ["Спасибо за обращение. Если захотите подобрать мебель, я рядом."],
        },
        "thanks": {
            "examples": ["спасибо", "благодарю", "спасибо большое"],
            "responses": ["Всегда пожалуйста!"],
        },
        "buy_furniture": {
            "examples": ["хочу купить мебель", "подберите мебель", "нужен диван", "покажите каталог"],
            "responses": [],
        },
        "agree": {
            "examples": ["да", "согласен", "покажите", "давайте", "интересно"],
            "responses": [],
        },
        "decline": {
            "examples": ["нет", "не надо", "не интересно", "откажусь"],
            "responses": [],
        },
    }
    return {
        label: _IntentData(
            label=label,
            examples=[normalize_for_matching(item) for item in data["examples"]],
            responses=list(data.get("responses", [])),
        )
        for label, data in raw.items()
    }
