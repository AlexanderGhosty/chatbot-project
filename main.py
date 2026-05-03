from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from config import AppConfig, load_config
from src.bot.router import register_handlers
from src.nlp.classifier import IntentClassifier, SentimentClassifier
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import VectorDatabase
from src.services.ad_campaign import AdCampaignManager
from src.services.dialogue_mgr import DialogueManager
from src.speech import SpeechProcessor
from src.speech.asr import ASRProcessor
from src.speech.tts import TTSProcessor


@dataclass(slots=True)
class ServiceContainer:
    intent_classifier: IntentClassifier
    sentiment_classifier: SentimentClassifier
    embedding_engine: EmbeddingEngine
    vector_db: VectorDatabase
    speech_processor: SpeechProcessor
    ad_campaign_manager: AdCampaignManager
    dialogue_manager: DialogueManager
    temp_audio_dir: str


def build_services(config: AppConfig) -> ServiceContainer:
    """Build service layer objects and wire dependencies."""
    intent_classifier = IntentClassifier(model_name=config.intent_model_name, intents_path=config.intents_path)
    sentiment_classifier = SentimentClassifier(model_name=config.sentiment_model_name)
    embedding_engine = EmbeddingEngine(model_name=config.embedding_model_name)
    vector_db = VectorDatabase(
        db_path=config.chroma_path,
        collection_name=config.chroma_collection,
        dialogues_path=config.dialogues_path,
    )
    speech_processor = SpeechProcessor(
        asr=ASRProcessor(model_name=config.asr_model_name),
        tts=TTSProcessor(
            model_name=config.tts_model_name,
            speaker=config.tts_speaker,
            allow_espeak_fallback=config.tts_allow_espeak_fallback,
        ),
    )
    ad_campaign_manager = AdCampaignManager.default()
    dialogue_manager = DialogueManager(
        intent_classifier=intent_classifier,
        sentiment_classifier=sentiment_classifier,
        embedding_engine=embedding_engine,
        vector_db=vector_db,
        speech_processor=speech_processor,
        ad_campaign_manager=ad_campaign_manager,
        retrieval_distance_threshold=config.retrieval_distance_threshold,
        ad_message_threshold=config.ad_message_threshold,
        voice_logging_enabled=config.voice_logging_enabled,
    )
    return ServiceContainer(
        intent_classifier=intent_classifier,
        sentiment_classifier=sentiment_classifier,
        embedding_engine=embedding_engine,
        vector_db=vector_db,
        speech_processor=speech_processor,
        ad_campaign_manager=ad_campaign_manager,
        dialogue_manager=dialogue_manager,
        temp_audio_dir=config.temp_audio_dir,
    )


async def create_app() -> tuple[Bot, Dispatcher, ServiceContainer]:
    """
    Create bot application runtime.

    Startup sequence:
    1. Load and validate config.
    2. Initialize bot and dispatcher.
    3. Build and connect service layer.
    4. Register handlers and routes.
    """
    config = load_config()
    bot = Bot(token=config.telegram_token)
    dispatcher = Dispatcher(storage=MemoryStorage())
    services = build_services(config)

    register_handlers(dispatcher=dispatcher, services=services)
    return bot, dispatcher, services


async def run() -> None:
    logging.basicConfig(level=logging.INFO)
    bot, dispatcher, _ = await create_app()
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(run())
