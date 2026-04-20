from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IntentResult:
    label: str
    confidence: float


@dataclass(slots=True)
class SentimentResult:
    label: str
    confidence: float


class IntentClassifier:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # TODO: load fine-tuned transformer classifier.

    async def predict(self, text: str) -> IntentResult:
        # TODO: run model inference in worker thread (asyncio.to_thread).
        # TODO: apply confidence thresholding and normalization.
        raise NotImplementedError


class SentimentClassifier:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # TODO: load sentiment model.

    async def predict(self, text: str) -> SentimentResult:
        # TODO: run sentiment inference and map to neutral/positive/negative.
        raise NotImplementedError
