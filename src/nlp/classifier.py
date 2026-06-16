from __future__ import annotations

import asyncio
import csv
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
            from sklearn.neural_network import MLPClassifier
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
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        activation="relu",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
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
    def __init__(self, model_name: str, lexicon_path: str = "data/raw/kartaslovsent.csv") -> None:
        self.model_name = model_name
        self.lexicon_path = Path(lexicon_path)
        self._pipeline = None
        self._pipeline_error: Exception | None = None
        self._lexicon = self._load_lexicon()

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

        score = self._score_with_lexicon(normalized)
        if -0.2 <= score <= 0.2:
            return SentimentResult(label="neutral", confidence=0.55)
        confidence = min(0.95, 0.55 + min(abs(score), 2.0) / 2.0 * 0.4)
        if score > 0.2:
            return SentimentResult(label="positive", confidence=confidence)
        return SentimentResult(label="negative", confidence=confidence)

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

    def _load_lexicon(self) -> dict[str, dict[str, float] | list[str]]:
        default = self._default_lexicon()
        if not self.lexicon_path.exists():
            return default

        if self.lexicon_path.suffix.lower() == ".csv":
            return self._load_kartaslovsent_csv(default)

        try:
            with self.lexicon_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return default

        if not isinstance(payload, dict):
            return default
        for key in ("positive", "negative", "intensifiers"):
            if not isinstance(payload.get(key), dict):
                payload[key] = default[key]
        if not isinstance(payload.get("negations"), list):
            payload["negations"] = default["negations"]
        return payload

    def _default_lexicon(self) -> dict[str, dict[str, float] | list[str]]:
        return {
            "positive": {
                "отлично": 1.0,
                "отличный": 1.0,
                "хорошо": 0.8,
                "хороший": 0.8,
                "классно": 0.8,
                "нравится": 0.9,
                "нравиться": 0.9,
                "люблю": 0.8,
                "спасибо": 0.7,
                "благодарность": 0.7,
                "супер": 1.0,
                "удобно": 0.7,
                "удобный": 0.7,
                "красиво": 0.7,
                "интересно": 0.5,
            },
            "negative": {
                "плохо": -0.8,
                "плохой": -0.8,
                "ужасно": -1.0,
                "ужасный": -1.0,
                "дорого": -0.6,
                "ненавижу": -1.0,
                "раздражает": -0.8,
                "недоволен": -0.8,
                "проблема": -0.7,
                "неудобно": -0.7,
                "не нравится": -0.9,
            },
            "intensifiers": {"очень": 1.4, "слишком": 1.3, "совсем": 1.2},
            "negations": ["не", "нет", "никогда"],
        }

    def _load_kartaslovsent_csv(
        self,
        default: dict[str, dict[str, float] | list[str]],
    ) -> dict[str, dict[str, float] | list[str]]:
        positive = dict(default["positive"]) if isinstance(default["positive"], dict) else {}
        negative = dict(default["negative"]) if isinstance(default["negative"], dict) else {}
        try:
            with self.lexicon_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file, delimiter=";")
                for row in reader:
                    term = normalize_for_matching(str(row.get("term", "")))
                    if not term:
                        continue
                    try:
                        value = float(str(row.get("value", "0")).replace(",", "."))
                    except ValueError:
                        continue
                    if value > 0.2:
                        positive.setdefault(term, value)
                    elif value < -0.2:
                        negative.setdefault(term, value)
        except OSError:
            return default

        return {
            "positive": positive,
            "negative": negative,
            "intensifiers": default["intensifiers"],
            "negations": default["negations"],
        }

    def _score_with_lexicon(self, normalized: str) -> float:
        positive = self._lexicon.get("positive", {})
        negative = self._lexicon.get("negative", {})
        intensifiers = self._lexicon.get("intensifiers", {})
        negations = {str(item) for item in self._lexicon.get("negations", [])}
        weighted = {**positive, **negative}

        words = normalized.split()
        score = 0.0
        consumed: set[int] = set()
        for phrase, raw_weight in sorted(weighted.items(), key=lambda item: len(str(item[0]).split()), reverse=True):
            phrase_words = str(phrase).split()
            if len(phrase_words) <= 1:
                continue
            for index in range(0, len(words) - len(phrase_words) + 1):
                if any(position in consumed for position in range(index, index + len(phrase_words))):
                    continue
                if words[index : index + len(phrase_words)] == phrase_words:
                    score += self._adjust_weight(float(raw_weight), words, index, intensifiers, negations)
                    consumed.update(range(index, index + len(phrase_words)))

        for index, word in enumerate(words):
            if index in consumed or word not in weighted:
                continue
            score += self._adjust_weight(float(weighted[word]), words, index, intensifiers, negations)

        return score

    def _adjust_weight(
        self,
        weight: float,
        words: list[str],
        index: int,
        intensifiers: dict[str, float] | list[str],
        negations: set[str],
    ) -> float:
        if index > 0 and words[index - 1] in negations:
            weight *= -1
        if index > 0 and isinstance(intensifiers, dict) and words[index - 1] in intensifiers:
            weight *= float(intensifiers[words[index - 1]])
        return weight


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
