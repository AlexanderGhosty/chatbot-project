from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from aiogram.fsm.context import FSMContext

from src.bot.states import DialogueStates
from src.nlp.classifier import IntentClassifier, IntentResult, SentimentClassifier
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import RetrievalResult, VectorDatabase
from src.services.ad_campaign import AdCampaignManager
from src.speech import SpeechProcessor
from src.utils.audio_conv import convert_ogg_to_wav
from src.utils.text_cleaner import normalize_user_text


@dataclass(slots=True)
class BotResponse:
    text: str
    send_voice: bool = False
    voice_path: str | None = None
    image_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DialogueContext:
    chat_id: int
    user_id: int
    message_count: int
    state_name: str | None


class DialogueManager:
    def __init__(
        self,
        *,
        intent_classifier: IntentClassifier,
        sentiment_classifier: SentimentClassifier,
        embedding_engine: EmbeddingEngine,
        vector_db: VectorDatabase,
        speech_processor: SpeechProcessor,
        ad_campaign_manager: AdCampaignManager,
        retrieval_distance_threshold: float,
        ad_message_threshold: int,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.sentiment_classifier = sentiment_classifier
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.speech_processor = speech_processor
        self.ad_campaign_manager = ad_campaign_manager
        self.retrieval_distance_threshold = retrieval_distance_threshold
        self.ad_message_threshold = ad_message_threshold

    async def process_text_message(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        state: FSMContext,
    ) -> BotResponse:
        """
        Text pipeline (placeholder):
        1. Validate/normalize input.
        2. Predict intent + sentiment.
        3. Evaluate FSM state and ad trigger.
        4. Route to ad path, predefined intent, or retrieval.
        """
        started_at = time.perf_counter()

        if not text.strip():
            return BotResponse(text="Пожалуйста, отправьте текстовое сообщение.")

        normalized_text = normalize_user_text(text)
        state_data = await state.get_data()
        message_count = int(state_data.get("message_count", 0)) + 1
        current_state = await state.get_state()
        ctx = DialogueContext(
            chat_id=chat_id,
            user_id=user_id,
            message_count=message_count,
            state_name=current_state,
        )

        intent = await self.intent_classifier.predict(normalized_text)
        _sentiment = await self.sentiment_classifier.predict(normalized_text)

        should_trigger_ad = await self.ad_campaign_manager.should_trigger_ad(
            intent=intent,
            normalized_text=normalized_text,
            message_count=ctx.message_count,
            ad_message_threshold=self.ad_message_threshold,
        )
        if should_trigger_ad and current_state != DialogueStates.ad_offering.state:
            await state.set_state(DialogueStates.ad_offering)
            ad_text, ad_images = await self.ad_campaign_manager.render_ad_offer()
            await state.update_data(message_count=message_count)
            return BotResponse(text=ad_text, image_paths=ad_images)

        if current_state == DialogueStates.ad_offering.state:
            reply_text = await self.ad_campaign_manager.handle_ad_reply(normalized_text, intent)
            await state.update_data(message_count=message_count)
            return BotResponse(text=reply_text)

        direct_response = await self._get_predefined_intent_response(intent)
        if direct_response is not None:
            await state.update_data(message_count=message_count)
            return BotResponse(text=direct_response)

        retrieval_result = await self._resolve_via_retrieval(normalized_text)
        await state.update_data(message_count=message_count)

        # TODO: emit structured log record with route, intent, confidence, and latency.
        _latency_ms = int((time.perf_counter() - started_at) * 1000)
        return BotResponse(text=retrieval_result)

    async def process_voice_message(
        self,
        *,
        chat_id: int,
        user_id: int,
        voice_file_id: str,
        state: FSMContext,
        temp_audio_dir: str,
    ) -> BotResponse:
        """Voice pipeline placeholder: download -> convert -> ASR -> text route -> TTS."""
        temp_dir = Path(temp_audio_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        ogg_path = temp_dir / f"{chat_id}_{user_id}_{voice_file_id}.ogg"
        wav_path = temp_dir / f"{chat_id}_{user_id}_{voice_file_id}.wav"
        out_voice_path = temp_dir / f"{chat_id}_{user_id}_{voice_file_id}_response.ogg"

        try:
            # TODO: download Telegram voice by file_id using Bot API.
            # TODO: persist incoming audio to ogg_path.
            convert_ogg_to_wav(str(ogg_path), str(wav_path))
            transcribed = await self.speech_processor.transcribe_audio(str(wav_path))
            text_response = await self.process_text_message(
                chat_id=chat_id,
                user_id=user_id,
                text=transcribed,
                state=state,
            )

            # TODO: define policy for when voice response should be skipped (e.g., long replies).
            synthesized_path = await self.speech_processor.synthesize_audio(
                text=text_response.text,
                output_path=str(out_voice_path),
            )
            text_response.send_voice = True
            text_response.voice_path = synthesized_path
            return text_response
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            # TODO: map concrete pipeline failures to user-facing fallback messages.
            return BotResponse(text="Не удалось обработать голосовое сообщение. Попробуйте еще раз.")
        finally:
            # TODO: cleanup policy for temp audio files and TTL-based retention.
            pass

    async def _get_predefined_intent_response(self, intent: IntentResult) -> str | None:
        # TODO: move mappings to config/localized resources.
        mapping = {
            "greeting": "Здравствуйте! Чем могу помочь по мебели?",
            "farewell": "Спасибо за обращение. Если захотите подобрать мебель, я рядом.",
            "thanks": "Всегда пожалуйста!",
        }
        if intent.confidence < 0.7:
            return None
        return mapping.get(intent.label)

    async def _resolve_via_retrieval(self, normalized_text: str) -> str:
        embedding = await self.embedding_engine.encode(normalized_text)
        result: RetrievalResult = await self.vector_db.search_answer(embedding=embedding, top_k=1)
        if result.distance > self.retrieval_distance_threshold:
            return "Я не совсем понял запрос. Могу помочь подобрать диван, стол или шкаф."
        return result.answer_text
