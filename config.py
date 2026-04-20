from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(slots=True)
class AppConfig:
    telegram_token: str
    chroma_path: str = "data/chromadb"
    chroma_collection: str = "dialogues"
    embedding_model_name: str = "cointegrated/rubert-tiny2"
    intent_model_name: str = "rubert-intent-placeholder"
    sentiment_model_name: str = "rubert-sentiment-placeholder"
    asr_model_name: str = "gigaam-placeholder"
    tts_model_name: str = "silero-tts-placeholder"
    retrieval_distance_threshold: float = 0.45
    ad_message_threshold: int = 4
    temp_audio_dir: str = "media/temp_audio"


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ConfigError("TELEGRAM_BOT_TOKEN is required")

    # TODO: add strict parsing/validation for optional env overrides.
    return AppConfig(telegram_token=token)
