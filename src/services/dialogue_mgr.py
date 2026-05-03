from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.bot.states import DialogueStates
from src.nlp.classifier import IntentClassifier, IntentResult, SentimentClassifier
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import RetrievalResult, VectorDatabase
from src.services.ad_campaign import AdCampaignManager
from src.speech import SpeechProcessor
from src.utils.audio_conv import convert_ogg_to_wav
from src.utils.text_cleaner import normalize_user_text

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BotResponse:
    text: str
    send_voice: bool = False
    voice_path: str | None = None
    image_paths: list[str] = field(default_factory=list)
    follow_ups: list["BotResponse"] = field(default_factory=list)


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
        voice_logging_enabled: bool = False,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.sentiment_classifier = sentiment_classifier
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.speech_processor = speech_processor
        self.ad_campaign_manager = ad_campaign_manager
        self.retrieval_distance_threshold = retrieval_distance_threshold
        self.ad_message_threshold = ad_message_threshold
        self.voice_logging_enabled = voice_logging_enabled

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
        if current_state is None:
            await state.set_state(DialogueStates.normal_chat)
            current_state = DialogueStates.normal_chat.state

        ctx = DialogueContext(
            chat_id=chat_id,
            user_id=user_id,
            message_count=message_count,
            state_name=current_state,
        )

        intent = await self.intent_classifier.predict(normalized_text)
        sentiment = await self.sentiment_classifier.predict(normalized_text)

        selected_product_sku = state_data.get("selected_product_sku")
        selected_product_sku = selected_product_sku if isinstance(selected_product_sku, str) else None
        if current_state in {DialogueStates.ad_offering.state, DialogueStates.ad_follow_up.state}:
            ad_reply = await self.ad_campaign_manager.handle_ad_reply(
                normalized_text,
                intent,
                selected_product_sku=selected_product_sku,
            )
            if not ad_reply.handled:
                regular_response = await self._build_regular_text_response(
                    intent=intent,
                    normalized_text=normalized_text,
                    sentiment_label=sentiment.label,
                    sentiment_confidence=sentiment.confidence,
                )
                await state.set_state(DialogueStates.normal_chat)
                await state.update_data(message_count=message_count, ad_declined=True)
                if regular_response is not None:
                    return regular_response
                return BotResponse(text="Могу помочь подобрать диван, стол или шкаф.")

            if ad_reply.declined:
                await state.set_state(DialogueStates.normal_chat)
                await state.update_data(message_count=message_count, ad_declined=True, selected_product_sku=None)
            else:
                selected_product_sku = ad_reply.selected_sku or selected_product_sku
                await state.set_state(DialogueStates.ad_follow_up)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=False,
                    selected_product_sku=selected_product_sku,
                )
            return BotResponse(text=self._apply_sentiment(ad_reply.text, sentiment.label, sentiment.confidence))

        if selected_product_sku and self.ad_campaign_manager.is_product_related(normalized_text, intent):
            ad_reply = await self.ad_campaign_manager.handle_ad_reply(
                normalized_text,
                intent,
                selected_product_sku=selected_product_sku,
            )
            if ad_reply.handled:
                selected_product_sku = ad_reply.selected_sku or selected_product_sku
                await state.set_state(DialogueStates.ad_follow_up)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=False,
                    selected_product_sku=selected_product_sku,
                )
                return BotResponse(text=self._apply_sentiment(ad_reply.text, sentiment.label, sentiment.confidence))

        should_trigger_ad = await self.ad_campaign_manager.should_trigger_ad(
            intent=intent,
            normalized_text=normalized_text,
            message_count=ctx.message_count,
            ad_message_threshold=self.ad_message_threshold,
        )
        explicit_ad_request = self._is_explicit_ad_request(intent, normalized_text)
        if should_trigger_ad and (not state_data.get("ad_declined") or explicit_ad_request):
            selected_product = self.ad_campaign_manager.find_selected_product(normalized_text, intent)
            if selected_product is not None or (explicit_ad_request and selected_product_sku):
                ad_reply = await self.ad_campaign_manager.handle_ad_reply(
                    normalized_text,
                    intent,
                    selected_product_sku=selected_product.sku if selected_product else selected_product_sku,
                )
                await state.set_state(DialogueStates.ad_follow_up)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=False,
                    selected_product_sku=ad_reply.selected_sku
                    or (selected_product.sku if selected_product else selected_product_sku),
                )
                return BotResponse(text=self._apply_sentiment(ad_reply.text, sentiment.label, sentiment.confidence))

            ad_text, ad_images = await self.ad_campaign_manager.render_ad_offer()
            await state.set_state(DialogueStates.ad_offering)
            await state.update_data(message_count=message_count, ad_declined=False, selected_product_sku=None)

            if explicit_ad_request:
                return BotResponse(text=ad_text, image_paths=ad_images)

            primary_response = await self._build_regular_text_response(
                intent=intent,
                normalized_text=normalized_text,
                sentiment_label=sentiment.label,
                sentiment_confidence=sentiment.confidence,
            )
            if primary_response is None:
                return BotResponse(text=ad_text, image_paths=ad_images)
            primary_response.follow_ups.append(BotResponse(text=ad_text, image_paths=ad_images))
            return primary_response

        regular_response = await self._build_regular_text_response(
            intent=intent,
            normalized_text=normalized_text,
            sentiment_label=sentiment.label,
            sentiment_confidence=sentiment.confidence,
        )
        await state.update_data(message_count=message_count)

        _latency_ms = int((time.perf_counter() - started_at) * 1000)
        if regular_response is not None:
            return regular_response
        return BotResponse(text="Могу помочь подобрать диван, стол или шкаф.")

    async def process_voice_message(
        self,
        *,
        chat_id: int,
        user_id: int,
        voice_file_id: str,
        state: FSMContext,
        temp_audio_dir: str,
        bot: Any | None = None,
        source_ogg_path: str | None = None,
    ) -> BotResponse:
        """Voice pipeline: download -> convert -> ASR -> text route -> TTS."""
        temp_dir = Path(temp_audio_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        safe_id = hashlib.sha1(voice_file_id.encode("utf-8")).hexdigest()[:16]
        ogg_path = temp_dir / f"{chat_id}_{user_id}_{safe_id}.ogg"
        wav_path = temp_dir / f"{chat_id}_{user_id}_{safe_id}.wav"
        out_voice_path = temp_dir / f"{chat_id}_{user_id}_{safe_id}_response.ogg"

        try:
            if source_ogg_path:
                source = Path(source_ogg_path)
                if source.resolve() != ogg_path.resolve():
                    shutil.copyfile(source, ogg_path)
            elif bot is not None:
                telegram_file = await bot.get_file(voice_file_id)
                await bot.download_file(telegram_file.file_path, destination=ogg_path)
            else:
                raise FileNotFoundError("voice source is not available")

            convert_ogg_to_wav(str(ogg_path), str(wav_path))
            transcribed = await self.speech_processor.transcribe_audio(str(wav_path))
            if self.voice_logging_enabled:
                logger.info(
                    "Voice transcription recognized: chat_id=%s user_id=%s text=%r",
                    chat_id,
                    user_id,
                    transcribed,
                )
            text_response = await self.process_text_message(
                chat_id=chat_id,
                user_id=user_id,
                text=transcribed,
                state=state,
            )

            try:
                synthesized_path = await self.speech_processor.synthesize_audio(
                    text=text_response.text,
                    output_path=str(out_voice_path),
                )
            except RuntimeError:
                logger.exception("TTS failed, sending text response instead")
                out_voice_path.unlink(missing_ok=True)
                out_voice_path.with_suffix(".wav").unlink(missing_ok=True)
                return text_response

            text_response.send_voice = True
            text_response.voice_path = synthesized_path
            return text_response
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            logger.exception("Voice message processing failed")
            return BotResponse(text="Не удалось обработать голосовое сообщение. Попробуйте еще раз.")
        except Exception:
            logger.exception("Unexpected voice message processing failure")
            return BotResponse(text="Не удалось обработать голосовое сообщение. Попробуйте еще раз.")
        finally:
            for path in (ogg_path, wav_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    async def _get_predefined_intent_response(self, intent: IntentResult) -> str | None:
        if intent.confidence < 0.5:
            return None
        return self.intent_classifier.get_random_response(intent.label)

    async def _build_regular_text_response(
        self,
        *,
        intent: IntentResult,
        normalized_text: str,
        sentiment_label: str,
        sentiment_confidence: float,
    ) -> BotResponse | None:
        direct_response = await self._get_predefined_intent_response(intent)
        if direct_response is not None:
            return BotResponse(
                text=self._apply_sentiment(direct_response, sentiment_label, sentiment_confidence),
            )

        if intent.label in {"buy_furniture", "ask_catalog", "product_sofa", "product_table", "product_wardrobe"}:
            return None

        retrieval_result = await self._resolve_via_retrieval(normalized_text)
        return BotResponse(text=self._apply_sentiment(retrieval_result, sentiment_label, sentiment_confidence))

    async def _resolve_via_retrieval(self, normalized_text: str) -> str:
        await self.vector_db.ensure_ready(self.embedding_engine)
        embedding = await self.embedding_engine.encode(normalized_text)
        result: RetrievalResult = await self.vector_db.search_answer(embedding=embedding, top_k=1)
        if result.distance > self.retrieval_distance_threshold:
            return "Я не совсем понял запрос. Могу помочь подобрать диван, стол или шкаф."
        return result.answer_text

    def _is_explicit_ad_request(self, intent: IntentResult, normalized_text: str) -> bool:
        explicit_intents = {
            "buy_furniture",
            "ask_catalog",
            "product_sofa",
            "product_table",
            "product_wardrobe",
            "product_details",
            "order_product",
        }
        if intent.label in explicit_intents and intent.confidence >= 0.35:
            return True

        product_words = {"диван", "стол", "шкаф"}
        action_markers = {"купить", "заказать", "каталог", "покажи", "хочу", "нужен", "нужна", "нужно", "подбери"}
        words = set(normalized_text.split())
        if words & action_markers:
            return True
        return len(words) <= 2 and bool(words & product_words)

    def _apply_sentiment(self, text: str, sentiment_label: str, confidence: float) -> str:
        if sentiment_label == "negative" and confidence >= 0.65:
            return f"Понимаю, это может раздражать. {text}"
        return text
