from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parent)
DEFAULT_INTENTS_PATH = "data/raw/intents.json"
DEFAULT_DIALOGUES_PATH = "data/raw/dialogues.txt"
DEFAULT_CHROMA_PATH = "data/chromadb"
DEFAULT_CHROMA_COLLECTION = "dialogues"
DEFAULT_CHITCHAT_ENABLED = True
DEFAULT_CHITCHAT_DIALOGUES_PATH = "data/raw/chitchat_dialogues.txt"
DEFAULT_CHITCHAT_CHROMA_COLLECTION = "chitchat_dialogues"
DEFAULT_CHITCHAT_RETRIEVAL_DISTANCE_THRESHOLD = 0.22
DEFAULT_EMBEDDING_MODEL_NAME = "cointegrated/rubert-tiny2"
DEFAULT_INTENT_MODEL_NAME = "local-intents"
DEFAULT_SENTIMENT_MODEL_NAME = "local-lexicon"
DEFAULT_ASR_MODEL_NAME = "ctc"
DEFAULT_TTS_MODEL_NAME = "v4_ru"
DEFAULT_TTS_SPEAKER = "xenia"
DEFAULT_TTS_ALLOW_ESPEAK_FALLBACK = False
DEFAULT_VOICE_LOGGING_ENABLED = False
DEFAULT_DIALOGUE_LOGGING_ENABLED = False
DEFAULT_RETRIEVAL_DISTANCE_THRESHOLD = 0.45
DEFAULT_AD_MESSAGE_THRESHOLD = 4
DEFAULT_TEMP_AUDIO_DIR = "media/temp_audio"
DEFAULT_PREWARM_VECTOR_INDEXES = True


@dataclass(slots=True)
class AppConfig:
    telegram_token: str
    project_root: str = DEFAULT_PROJECT_ROOT
    intents_path: str = DEFAULT_INTENTS_PATH
    dialogues_path: str = DEFAULT_DIALOGUES_PATH
    chroma_path: str = DEFAULT_CHROMA_PATH
    chroma_collection: str = DEFAULT_CHROMA_COLLECTION
    chitchat_enabled: bool = DEFAULT_CHITCHAT_ENABLED
    chitchat_dialogues_path: str = DEFAULT_CHITCHAT_DIALOGUES_PATH
    chitchat_chroma_collection: str = DEFAULT_CHITCHAT_CHROMA_COLLECTION
    chitchat_retrieval_distance_threshold: float = DEFAULT_CHITCHAT_RETRIEVAL_DISTANCE_THRESHOLD
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    intent_model_name: str = DEFAULT_INTENT_MODEL_NAME
    sentiment_model_name: str = DEFAULT_SENTIMENT_MODEL_NAME
    asr_model_name: str = DEFAULT_ASR_MODEL_NAME
    tts_model_name: str = DEFAULT_TTS_MODEL_NAME
    tts_speaker: str = DEFAULT_TTS_SPEAKER
    tts_allow_espeak_fallback: bool = DEFAULT_TTS_ALLOW_ESPEAK_FALLBACK
    voice_logging_enabled: bool = DEFAULT_VOICE_LOGGING_ENABLED
    dialogue_logging_enabled: bool = DEFAULT_DIALOGUE_LOGGING_ENABLED
    retrieval_distance_threshold: float = DEFAULT_RETRIEVAL_DISTANCE_THRESHOLD
    ad_message_threshold: int = DEFAULT_AD_MESSAGE_THRESHOLD
    temp_audio_dir: str = DEFAULT_TEMP_AUDIO_DIR
    prewarm_vector_indexes: bool = DEFAULT_PREWARM_VECTOR_INDEXES


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def _get_path(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    _load_dotenv(Path(DEFAULT_PROJECT_ROOT) / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ConfigError("TELEGRAM_BOT_TOKEN is required")

    return AppConfig(
        telegram_token=token,
        intents_path=_get_path("INTENTS_PATH", DEFAULT_INTENTS_PATH),
        dialogues_path=_get_path("DIALOGUES_PATH", DEFAULT_DIALOGUES_PATH),
        chroma_path=_get_path("CHROMA_PATH", DEFAULT_CHROMA_PATH),
        chroma_collection=os.getenv("CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION).strip()
        or DEFAULT_CHROMA_COLLECTION,
        chitchat_enabled=_get_bool("CHITCHAT_ENABLED", DEFAULT_CHITCHAT_ENABLED),
        chitchat_dialogues_path=_get_path("CHITCHAT_DIALOGUES_PATH", DEFAULT_CHITCHAT_DIALOGUES_PATH),
        chitchat_chroma_collection=os.getenv(
            "CHITCHAT_CHROMA_COLLECTION",
            DEFAULT_CHITCHAT_CHROMA_COLLECTION,
        ).strip()
        or DEFAULT_CHITCHAT_CHROMA_COLLECTION,
        chitchat_retrieval_distance_threshold=_get_float(
            "CHITCHAT_RETRIEVAL_DISTANCE_THRESHOLD",
            DEFAULT_CHITCHAT_RETRIEVAL_DISTANCE_THRESHOLD,
        ),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME).strip()
        or DEFAULT_EMBEDDING_MODEL_NAME,
        intent_model_name=os.getenv("INTENT_MODEL_NAME", DEFAULT_INTENT_MODEL_NAME).strip()
        or DEFAULT_INTENT_MODEL_NAME,
        sentiment_model_name=os.getenv("SENTIMENT_MODEL_NAME", DEFAULT_SENTIMENT_MODEL_NAME).strip()
        or DEFAULT_SENTIMENT_MODEL_NAME,
        asr_model_name=os.getenv("ASR_MODEL_NAME", DEFAULT_ASR_MODEL_NAME).strip() or DEFAULT_ASR_MODEL_NAME,
        tts_model_name=os.getenv("TTS_MODEL_NAME", DEFAULT_TTS_MODEL_NAME).strip() or DEFAULT_TTS_MODEL_NAME,
        tts_speaker=os.getenv("TTS_SPEAKER", DEFAULT_TTS_SPEAKER).strip() or DEFAULT_TTS_SPEAKER,
        tts_allow_espeak_fallback=_get_bool("TTS_ALLOW_ESPEAK_FALLBACK", DEFAULT_TTS_ALLOW_ESPEAK_FALLBACK),
        voice_logging_enabled=_get_bool("VOICE_LOGGING_ENABLED", DEFAULT_VOICE_LOGGING_ENABLED),
        dialogue_logging_enabled=_get_bool("DIALOGUE_LOGGING_ENABLED", DEFAULT_DIALOGUE_LOGGING_ENABLED),
        retrieval_distance_threshold=_get_float(
            "RETRIEVAL_DISTANCE_THRESHOLD", DEFAULT_RETRIEVAL_DISTANCE_THRESHOLD
        ),
        ad_message_threshold=_get_int("AD_MESSAGE_THRESHOLD", DEFAULT_AD_MESSAGE_THRESHOLD),
        temp_audio_dir=_get_path("TEMP_AUDIO_DIR", DEFAULT_TEMP_AUDIO_DIR),
        prewarm_vector_indexes=_get_bool("PREWARM_VECTOR_INDEXES", DEFAULT_PREWARM_VECTOR_INDEXES),
    )
