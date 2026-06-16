from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.bot.states import DialogueStates
from src.nlp.chitchat_safety import is_safe_chitchat_answer
from src.nlp.classifier import IntentClassifier, IntentResult, SentimentClassifier
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import RetrievalResult, VectorDatabase
from src.nlp.text_analyzer import TextAnalyzer, TextEntity
from src.services.ad_campaign import AdCampaignManager
from src.speech import SpeechProcessor
from src.utils.audio_conv import convert_ogg_to_wav
from src.utils.fuzzy import correct_domain_terms
from src.utils.text_cleaner import normalize_for_matching, normalize_user_text

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
    source: str = "unknown"
    matched_question: str | None = None
    retrieval_distance: float | None = None


@dataclass(slots=True)
class DialogueContext:
    chat_id: int
    user_id: int
    message_count: int
    state_name: str | None


@dataclass(slots=True)
class DialogueRoute:
    topic: str
    chitchat_mode: bool


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
        text_analyzer: TextAnalyzer | None = None,
        chitchat_vector_db: VectorDatabase | None = None,
        chitchat_retrieval_distance_threshold: float = 0.45,
        voice_logging_enabled: bool = False,
        dialogue_logging_enabled: bool = False,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.sentiment_classifier = sentiment_classifier
        self.text_analyzer = text_analyzer or TextAnalyzer(natasha_enabled=False)
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.speech_processor = speech_processor
        self.ad_campaign_manager = ad_campaign_manager
        self.retrieval_distance_threshold = retrieval_distance_threshold
        self.chitchat_vector_db = chitchat_vector_db
        self.chitchat_retrieval_distance_threshold = chitchat_retrieval_distance_threshold
        self.ad_message_threshold = ad_message_threshold
        self.voice_logging_enabled = voice_logging_enabled
        self.dialogue_logging_enabled = dialogue_logging_enabled

    async def process_text_message(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        state: FSMContext,
    ) -> BotResponse:
        started_at = time.perf_counter()

        if not text.strip():
            return BotResponse(text="Пожалуйста, отправьте текстовое сообщение.", source="validation")

        analysis = self.text_analyzer.analyze(text)
        normalized_text = analysis.matching_text
        if not normalized_text:
            return BotResponse(
                text="Я понял реакцию. Можем продолжить разговор текстом или вернуться к выбору мебели.",
                source="empty_normalized_input",
            )
        state_data = await state.get_data()
        message_count = int(state_data.get("message_count", 0)) + 1
        stored_order_context = self._normalize_order_context(state_data.get("order_context"))
        order_context = stored_order_context
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

        raw_intent = await self.intent_classifier.predict(normalized_text)
        intent = self._refine_intent(raw_intent, normalized_text)
        sentiment = await self.sentiment_classifier.predict(normalized_text)
        route = self._classify_route(
            intent=intent,
            normalized_text=normalized_text,
            previous_chitchat_mode=bool(state_data.get("chitchat_mode", False)),
        )
        explicit_ad_request = self._is_explicit_ad_request(intent, normalized_text)

        selected_product_sku = state_data.get("selected_product_sku")
        selected_product_sku = selected_product_sku if isinstance(selected_product_sku, str) else None
        if self._is_negative_feedback(normalized_text):
            await state.set_state(DialogueStates.normal_chat)
            await state.update_data(
                message_count=message_count,
                chitchat_mode=True,
                order_context=stored_order_context,
            )
            response = BotResponse(
                text="Понимаю, ответ прозвучал неудачно. Уточните запрос, и я отвечу точнее.",
                source="negative_feedback",
            )
            self._log_decision(
                ctx=ctx,
                text=normalized_text,
                intent=intent,
                route=route,
                response=response,
                ad_decision="negative_feedback",
                started_at=started_at,
            )
            return response

        if current_state in {DialogueStates.ad_offering.state, DialogueStates.ad_follow_up.state}:
            if self.ad_campaign_manager.is_product_related(normalized_text, intent):
                order_context = self._merge_order_context(stored_order_context, analysis.entities)
            ad_reply = await self.ad_campaign_manager.handle_ad_reply(
                normalized_text,
                intent,
                selected_product_sku=selected_product_sku,
                entities=analysis.entities,
                order_context=order_context,
            )
            if not ad_reply.handled:
                regular_response = await self._build_regular_text_response(
                    intent=intent,
                    normalized_text=normalized_text,
                    sentiment_label=sentiment.label,
                    sentiment_confidence=sentiment.confidence,
                    route=route,
                )
                await state.set_state(DialogueStates.normal_chat)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=True,
                    chitchat_mode=route.chitchat_mode,
                    selected_product_sku=None,
                    order_context=order_context,
                )
                if regular_response is not None:
                    self._log_decision(
                        ctx=ctx,
                        text=normalized_text,
                        intent=intent,
                        route=route,
                        response=regular_response,
                        ad_decision="ad_state_offtopic",
                        started_at=started_at,
                    )
                    return regular_response
                response = BotResponse(text="Могу помочь подобрать диван, стол или шкаф.", source="fallback")
                self._log_decision(
                    ctx=ctx,
                    text=normalized_text,
                    intent=intent,
                    route=route,
                    response=response,
                    ad_decision="ad_state_offtopic",
                    started_at=started_at,
                )
                return response

            if ad_reply.declined:
                await state.set_state(DialogueStates.normal_chat)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=True,
                    selected_product_sku=None,
                    chitchat_mode=route.chitchat_mode,
                    order_context=order_context,
                )
            else:
                selected_product_sku = ad_reply.selected_sku or selected_product_sku
                await state.set_state(DialogueStates.ad_follow_up)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=False,
                    selected_product_sku=selected_product_sku,
                    chitchat_mode=False,
                    order_context=order_context,
                )
            response = BotResponse(
                text=self._apply_sentiment(ad_reply.text, sentiment.label, sentiment.confidence),
                source="ad_reply",
            )
            self._log_decision(
                ctx=ctx,
                text=normalized_text,
                intent=intent,
                route=route,
                response=response,
                ad_decision="ad_state_handled",
                started_at=started_at,
            )
            return response

        if selected_product_sku and self.ad_campaign_manager.is_product_related(normalized_text, intent):
            selected_product = self.ad_campaign_manager.find_selected_product(
                normalized_text,
                intent,
                entities=analysis.entities,
            )
            if selected_product is not None and selected_product.sku != selected_product_sku:
                order_context = self._merge_order_context({}, analysis.entities)
            else:
                order_context = self._merge_order_context(stored_order_context, analysis.entities)
            ad_reply = await self.ad_campaign_manager.handle_ad_reply(
                normalized_text,
                intent,
                selected_product_sku=selected_product_sku,
                entities=analysis.entities,
                order_context=order_context,
            )
            if ad_reply.handled:
                selected_product_sku = ad_reply.selected_sku or selected_product_sku
                await state.set_state(DialogueStates.ad_follow_up)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=False,
                    selected_product_sku=selected_product_sku,
                    chitchat_mode=False,
                    order_context=order_context,
                )
                response = BotResponse(
                    text=self._apply_sentiment(ad_reply.text, sentiment.label, sentiment.confidence),
                    source="ad_reply",
                )
                self._log_decision(
                    ctx=ctx,
                    text=normalized_text,
                    intent=intent,
                    route=route,
                    response=response,
                    ad_decision="selected_product_follow_up",
                    started_at=started_at,
                )
                return response

        should_trigger_ad = await self.ad_campaign_manager.should_trigger_ad(
            intent=intent,
            normalized_text=normalized_text,
            message_count=ctx.message_count,
            ad_message_threshold=self.ad_message_threshold,
        )
        should_trigger_ad = should_trigger_ad or self._is_soft_ad_bridge(normalized_text)
        if should_trigger_ad and self._is_ad_allowed(
            state_data=state_data,
            message_count=message_count,
            route=route,
            explicit_ad_request=explicit_ad_request,
            normalized_text=normalized_text,
        ):
            selected_product = self.ad_campaign_manager.find_selected_product(
                normalized_text,
                intent,
                entities=analysis.entities,
            )
            if selected_product is not None or (explicit_ad_request and selected_product_sku):
                if selected_product is not None and selected_product.sku != selected_product_sku:
                    order_context = self._merge_order_context({}, analysis.entities)
                else:
                    order_context = self._merge_order_context(stored_order_context, analysis.entities)
                ad_reply = await self.ad_campaign_manager.handle_ad_reply(
                    normalized_text,
                    intent,
                    selected_product_sku=selected_product.sku if selected_product else selected_product_sku,
                    entities=analysis.entities,
                    order_context=order_context,
                )
                await state.set_state(DialogueStates.ad_follow_up)
                await state.update_data(
                    message_count=message_count,
                    ad_declined=False,
                    selected_product_sku=ad_reply.selected_sku
                    or (selected_product.sku if selected_product else selected_product_sku),
                    chitchat_mode=False,
                    order_context=order_context,
                )
                response = BotResponse(
                    text=self._apply_sentiment(ad_reply.text, sentiment.label, sentiment.confidence),
                    source="ad_product",
                )
                self._log_decision(
                    ctx=ctx,
                    text=normalized_text,
                    intent=intent,
                    route=route,
                    response=response,
                    ad_decision="product_selected",
                    started_at=started_at,
                )
                return response

            ad_text, ad_images = await self.ad_campaign_manager.render_ad_offer()
            await state.set_state(DialogueStates.ad_offering)
            await state.update_data(
                message_count=message_count,
                ad_declined=False,
                selected_product_sku=None,
                chitchat_mode=False,
                last_ad_message_count=message_count,
                order_context=order_context,
            )

            if explicit_ad_request:
                response = BotResponse(text=ad_text, image_paths=ad_images, source="ad_offer")
                self._log_decision(
                    ctx=ctx,
                    text=normalized_text,
                    intent=intent,
                    route=route,
                    response=response,
                    ad_decision="explicit_offer",
                    started_at=started_at,
                )
                return response

            primary_response = await self._build_regular_text_response(
                intent=intent,
                normalized_text=normalized_text,
                sentiment_label=sentiment.label,
                sentiment_confidence=sentiment.confidence,
                route=route,
            )
            if primary_response is None:
                response = BotResponse(text=ad_text, image_paths=ad_images, source="ad_offer")
                self._log_decision(
                    ctx=ctx,
                    text=normalized_text,
                    intent=intent,
                    route=route,
                    response=response,
                    ad_decision="soft_offer_without_primary",
                    started_at=started_at,
                )
                return response
            primary_response.follow_ups.append(BotResponse(text=ad_text, image_paths=ad_images, source="ad_offer"))
            self._log_decision(
                ctx=ctx,
                text=normalized_text,
                intent=intent,
                route=route,
                response=primary_response,
                ad_decision="soft_offer_follow_up",
                started_at=started_at,
            )
            return primary_response

        regular_response = await self._build_regular_text_response(
            intent=intent,
            normalized_text=normalized_text,
            sentiment_label=sentiment.label,
            sentiment_confidence=sentiment.confidence,
            route=route,
        )
        await state.update_data(
            message_count=message_count,
            chitchat_mode=route.chitchat_mode,
            order_context=stored_order_context,
        )

        if regular_response is not None:
            self._log_decision(
                ctx=ctx,
                text=normalized_text,
                intent=intent,
                route=route,
                response=regular_response,
                ad_decision="none",
                started_at=started_at,
            )
            return regular_response
        response = BotResponse(text="Могу помочь подобрать диван, стол или шкаф.", source="fallback")
        self._log_decision(
            ctx=ctx,
            text=normalized_text,
            intent=intent,
            route=route,
            response=response,
            ad_decision="none",
            started_at=started_at,
        )
        return response

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
        route: DialogueRoute,
    ) -> BotResponse | None:
        direct_response = await self._get_predefined_intent_response(intent)
        if direct_response is not None:
            return BotResponse(
                text=self._apply_sentiment(direct_response, sentiment_label, sentiment_confidence),
                source="intent",
            )

        if intent.label in {"buy_furniture", "ask_catalog", "product_sofa", "product_table", "product_wardrobe"}:
            return None

        domain_rule_response = self._get_domain_rule_response(normalized_text)
        if domain_rule_response is not None and route.topic in {"furniture", "service"}:
            return BotResponse(
                text=self._apply_sentiment(domain_rule_response, sentiment_label, sentiment_confidence),
                source="domain_rule",
            )

        if route.topic in {"smalltalk", "unknown"}:
            chitchat_override = self._resolve_chitchat_override(normalized_text)
            if chitchat_override is not None:
                return BotResponse(
                    text=self._apply_sentiment(chitchat_override, sentiment_label, sentiment_confidence),
                    source="chitchat_override",
                )

            chitchat_result = await self._resolve_chitchat(normalized_text)
            if chitchat_result is not None and self._has_negation_mismatch(
                normalized_text,
                chitchat_result.matched_question,
            ):
                chitchat_result = None
            if chitchat_result is not None:
                return BotResponse(
                    text=self._apply_sentiment(chitchat_result.answer_text, sentiment_label, sentiment_confidence),
                    source="chitchat_retrieval",
                    matched_question=chitchat_result.matched_question,
                    retrieval_distance=chitchat_result.distance,
                )

            return BotResponse(
                text="Расскажите подробнее. Я поддержу разговор, а если понадобится мебель, помогу с выбором.",
                source="chitchat_fallback",
            )

        retrieval_result = await self._resolve_via_retrieval(
            vector_db=self.vector_db,
            threshold=self.retrieval_distance_threshold,
            normalized_text=normalized_text,
        )
        if retrieval_result is not None:
            return BotResponse(
                text=self._apply_sentiment(retrieval_result.answer_text, sentiment_label, sentiment_confidence),
                source="domain_retrieval",
                matched_question=retrieval_result.matched_question,
                retrieval_distance=retrieval_result.distance,
            )

        return BotResponse(text="Я не совсем понял запрос. Могу помочь подобрать диван, стол или шкаф.", source="fallback")

    async def _resolve_via_retrieval(
        self,
        *,
        vector_db: VectorDatabase,
        threshold: float,
        normalized_text: str,
    ) -> RetrievalResult | None:
        await vector_db.ensure_ready(self.embedding_engine)
        embedding = await self.embedding_engine.encode(normalized_text)
        result: RetrievalResult = await vector_db.search_answer(embedding=embedding, top_k=1)
        if result.distance > threshold:
            return None
        return result

    async def _resolve_chitchat(self, normalized_text: str) -> RetrievalResult | None:
        if self.chitchat_vector_db is None:
            return None
        result = await self._resolve_via_retrieval(
            vector_db=self.chitchat_vector_db,
            threshold=self.chitchat_retrieval_distance_threshold,
            normalized_text=normalized_text,
        )
        if result is None or not is_safe_chitchat_answer(result.answer_text):
            return None
        return result

    def _resolve_chitchat_override(self, normalized_text: str) -> str | None:
        overrides = {
            "привет сладенький": "Здравствуйте. Могу просто поговорить или помочь с выбором мебели.",
            "давай просто поговорим о всяком": "Давайте. О чем хотите поговорить?",
            "давай поговорим о всяком": "Давайте. О чем хотите поговорить?",
            "давай просто поговорим": "Давайте. О чем хотите поговорить?",
            "поговорим о всяком": "Давайте. О чем хотите поговорить?",
            "давай поболтаем о разном": "Давайте. Можно поговорить о чем угодно, а если понадобится мебель - подскажу по каталогу.",
            "я тебя понял давай поболтаем о разном": "Давайте. Можно поговорить о чем угодно, а если понадобится мебель - подскажу по каталогу.",
            "мне нравится небо": "Небо правда часто задает настроение. Что вам в нем нравится больше всего?",
            "я нравиться небо": "Небо правда часто задает настроение. Что вам в нем нравится больше всего?",
            "а что тебе нравится": "У меня нет личных предпочтений, но можно поговорить о прогулках, спорте или мебели.",
            "а что ты нравиться": "У меня нет личных предпочтений, но можно поговорить о прогулках, спорте или мебели.",
            "как тебе моя прическа": "Я не вижу вас, но короткая стрижка часто выглядит аккуратно и уверенно.",
            "я люблю гулять": "Прогулки хорошо разгружают голову. Где вам больше нравится гулять?",
            "я любить гулять": "Прогулки хорошо разгружают голову. Где вам больше нравится гулять?",
            "как ты относишься к футболу": "У меня нет личных симпатий, но о футболе можно поговорить. За какую команду болеете?",
            "как ты относиться к футбол": "У меня нет личных симпатий, но о футболе можно поговорить. За какую команду болеете?",
            "давай погуляем завтра": "Звучит как план для прогулки. Лучше ориентироваться на погоду и самочувствие.",
            "давать погулять завтра": "Звучит как план для прогулки. Лучше ориентироваться на погоду и самочувствие.",
            "это замечательный": "Рад, что вам понравилось. Можем продолжить разговор или вернуться к мебели.",
            "по лесу": "Лес хорошо подходит для спокойной прогулки. Там обычно проще переключиться и отдохнуть.",
            "мне нравится короткая стрижка": "Короткая стрижка обычно выглядит аккуратно и проста в уходе.",
            "ты мое солнышко": "Спасибо, приятно слышать. Чем сегодня хотите поделиться?",
            "ты моя солнышко": "Спасибо, приятно слышать. Чем сегодня хотите поделиться?",
            "ты любишь хоккей": "У меня нет личных увлечений, но о хоккее можно поговорить. Вы за какую команду болеете?",
            "новости": "Я не подключен к ленте новостей, но могу обсудить тему, если расскажете, что именно интересно.",
            "нечаянно": "Понимаю. Такое бывает. Что случилось?",
            "странные у тебя ответы": "Да, иногда могу ошибаться в разговоре. Лучше уточните, что именно звучит странно.",
            "ты странный": "Могу звучать странно, потому что я бот. Постараюсь отвечать понятнее.",
            "ты очень странный и ответ у ты кривой": "Понимаю, ответ прозвучал неудачно. Уточните запрос, и я отвечу точнее.",
            "весь с ты понятно ты очень глупый": "Понимаю, ответ был неудачным. Попробую отвечать точнее.",
            "ты ужасный": "Понимаю, что ответ не устроил. Лучше уточните запрос, и я исправлюсь.",
            "а я не рад ты видеть": "Понимаю. Не буду спорить, лучше попробую отвечать полезнее.",
            "а я не рад тебя видеть": "Понимаю. Не буду спорить, лучше попробую отвечать полезнее.",
            "ну ты капец вообще": "Понимаю, что ответ не понравился. Давайте уточним вопрос и попробуем еще раз.",
            "приколдес": "Понимаю, звучит забавно. Продолжим?",
            "лол": "Понял. Смешно получилось.",
            "вчера зарплату получил теперь думаю на что потратить": (
                "Разумно сначала закрыть обязательные расходы, а часть можно оставить на то, "
                "что улучшит повседневный комфорт."
            ),
            "завтра зарплата думаю что то купить": (
                "Можно заранее решить, что действительно улучшит быт, и не тратить деньги на случайные вещи."
            ),
            "завтра зарлата думаю что то купить": (
                "Можно заранее решить, что действительно улучшит быт, и не тратить деньги на случайные вещи."
            ),
        }
        return overrides.get(normalize_for_matching(normalized_text))

    def _get_domain_rule_response(self, normalized_text: str) -> str | None:
        words = set(normalize_for_matching(normalized_text).split())
        if words & {"цена", "цены", "стоимость", "стоит", "сколько", "дорого", "скидки"}:
            return (
                "Цена зависит от модели, размера и материала. "
                "Могу показать базовый каталог и подсказать, какой товар ближе к вашему бюджету."
            )
        if words & {"доставка", "доставляете", "доставить", "привезете"}:
            return "Доставка доступна по городу и области. Интервал и стоимость зависят от адреса и габаритов товара."
        if words & {"оплата", "оплатить", "картой", "наличные", "рассрочка"}:
            return "Обычно доступны оплата картой, наличными при получении и безналичный расчет."
        return None

    def _classify_route(
        self,
        *,
        intent: IntentResult,
        normalized_text: str,
        previous_chitchat_mode: bool,
    ) -> DialogueRoute:
        if self._is_negative_feedback(normalized_text):
            return DialogueRoute(topic="smalltalk", chitchat_mode=True)

        if self._is_furniture_related(normalized_text, intent):
            return DialogueRoute(topic="furniture", chitchat_mode=False)

        if intent.label in {"ask_delivery", "ask_payment", "ask_price"} and intent.confidence >= 0.35:
            return DialogueRoute(topic="service", chitchat_mode=False)

        chitchat_requested = self._is_chitchat_mode_request(normalized_text)
        if previous_chitchat_mode or chitchat_requested or self._has_smalltalk_markers(normalized_text):
            return DialogueRoute(topic="smalltalk", chitchat_mode=True)

        if intent.label in {"greeting", "farewell", "thanks", "help", "ask_weather", "nice_to_meet"}:
            return DialogueRoute(topic="service", chitchat_mode=previous_chitchat_mode)

        return DialogueRoute(topic="smalltalk", chitchat_mode=previous_chitchat_mode)

    def _is_ad_allowed(
        self,
        *,
        state_data: dict[str, Any],
        message_count: int,
        route: DialogueRoute,
        explicit_ad_request: bool,
        normalized_text: str,
    ) -> bool:
        if explicit_ad_request:
            return True
        if self._is_soft_ad_bridge(normalized_text):
            last_ad_message_count = int(state_data.get("last_ad_message_count", -1000))
            cooldown = max(3, self.ad_message_threshold)
            return message_count - last_ad_message_count >= cooldown
        if route.chitchat_mode or route.topic == "smalltalk":
            return False
        if state_data.get("ad_declined"):
            return False
        if route.topic == "service" and message_count < self.ad_message_threshold:
            return False

        last_ad_message_count = int(state_data.get("last_ad_message_count", -1000))
        cooldown = max(3, self.ad_message_threshold)
        return message_count - last_ad_message_count >= cooldown

    def _is_soft_ad_bridge(self, normalized_text: str) -> bool:
        words = set(normalize_for_matching(normalized_text).split())
        return bool(
            words
            & {
                "зарплату",
                "зарплата",
                "деньги",
                "потратить",
                "покупку",
                "покупки",
                "купить",
                "обновить",
                "уют",
                "ремонт",
                "квартира",
                "дом",
                "комната",
            }
        )

    def _is_chitchat_mode_request(self, normalized_text: str) -> bool:
        text = normalize_for_matching(normalized_text)
        phrases = {
            "просто поговорим",
            "поговорим о всяком",
            "давай поговорим",
            "давай просто поговорим",
            "хочу поговорить",
            "поболтаем",
        }
        return any(phrase in text for phrase in phrases)

    def _has_smalltalk_markers(self, normalized_text: str) -> bool:
        words = set(normalize_for_matching(normalized_text).split())
        markers = {
            "гулять",
            "погулять",
            "небо",
            "нравится",
            "нравиться",
            "люблю",
            "любить",
            "прическа",
            "стрижка",
            "солнышко",
            "хоккей",
            "футбол",
            "новости",
            "нечаянно",
            "поговорим",
            "поговорить",
            "всяком",
            "замечательный",
        }
        return bool(words & markers)

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
        if intent.label in explicit_intents and intent.confidence >= 0.35 and self._has_explicit_ad_markers(normalized_text):
            return True

        product_words = {"диван", "стол", "шкаф"}
        action_markers = {"купить", "заказать", "покажи", "хочу", "нужен", "нужна", "нужно", "подбери", "выбери"}
        words = set(correct_domain_terms(normalized_text).split())
        if "каталог" in words:
            return True
        if (words & action_markers) and (words & (product_words | {"мебель"})):
            return True
        return len(words) <= 2 and bool(words & product_words)

    def _is_furniture_related(self, normalized_text: str, intent: IntentResult | None = None) -> bool:
        text = correct_domain_terms(normalized_text)
        furniture_words = {
            "мебель",
            "диван",
            "стол",
            "шкаф",
            "кровать",
            "комод",
            "кухня",
            "гостиная",
            "прихожая",
            "каталог",
        }
        words = set(text.split())
        if intent is not None and intent.label in {
            "buy_furniture",
            "ask_catalog",
            "product_sofa",
            "product_table",
            "product_wardrobe",
            "product_details",
            "order_product",
        } and intent.confidence >= 0.35 and self._has_explicit_ad_markers(text):
            return True
        return bool(words & furniture_words)

    def _log_decision(
        self,
        *,
        ctx: DialogueContext,
        text: str,
        intent: IntentResult,
        route: DialogueRoute,
        response: BotResponse,
        ad_decision: str,
        started_at: float,
    ) -> None:
        if not self.dialogue_logging_enabled:
            return
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Dialogue route: chat_id=%s user_id=%s state=%s source=%s topic=%s "
            "chitchat_mode=%s intent=%s confidence=%.3f ad=%s matched=%r distance=%s latency_ms=%s text=%r",
            ctx.chat_id,
            ctx.user_id,
            ctx.state_name,
            response.source,
            route.topic,
            route.chitchat_mode,
            intent.label,
            intent.confidence,
            ad_decision,
            response.matched_question,
            f"{response.retrieval_distance:.3f}" if response.retrieval_distance is not None else None,
            latency_ms,
            text,
        )

    def _apply_sentiment(self, text: str, sentiment_label: str, confidence: float) -> str:
        if sentiment_label == "negative" and confidence >= 0.65:
            return f"Понимаю, это может раздражать. {text}"
        return text

    def _refine_intent(self, intent: IntentResult, normalized_text: str) -> IntentResult:
        words = set(correct_domain_terms(normalized_text).split())
        if intent.label in {"product_sofa", "product_table", "product_wardrobe"} and not (
            words & {"диван", "софа", "стол", "шкаф", "гардероб"}
        ):
            return IntentResult(label="unknown", confidence=min(intent.confidence, 0.31))

        if intent.label in {"buy_furniture", "ask_catalog", "product_details", "order_product"} and not self._has_explicit_ad_markers(normalized_text):
            return IntentResult(label="unknown", confidence=min(intent.confidence, 0.31))

        if intent.label == "ask_weather" and not (words & {"погода", "дождь", "снег", "температура", "прогноз"}):
            return IntentResult(label="unknown", confidence=min(intent.confidence, 0.31))

        if intent.label == "help" and not (words & {"уметь", "умеешь", "помоги", "помощь", "команда", "команды", "пользоваться", "кто"}):
            return IntentResult(label="unknown", confidence=min(intent.confidence, 0.31))

        if intent.label == "thanks" and not (words & {"спасибо", "благодарю", "спс", "помогли"}):
            return IntentResult(label="unknown", confidence=min(intent.confidence, 0.31))

        if intent.label == "greeting" and not (words & {"привет", "здравствуйте", "добрый", "хай", "начнем"}):
            return IntentResult(label="unknown", confidence=min(intent.confidence, 0.31))

        return intent

    def _has_explicit_ad_markers(self, normalized_text: str) -> bool:
        words = set(correct_domain_terms(normalized_text).split())
        product_words = {"мебель", "диван", "софа", "стол", "шкаф", "гардероб", "каталог"}
        action_words = {
            "купить",
            "заказать",
            "оформить",
            "оформим",
            "беру",
            "возьму",
            "хочу",
            "нужен",
            "нужна",
            "нужно",
            "покажи",
            "подбери",
            "выбери",
            "размер",
            "размеры",
            "доставка",
            "оплата",
            "цена",
            "стоимость",
        }
        return bool(words & product_words) or bool((words & action_words) and (words & {"товар", "заказ"}))

    def _is_negative_feedback(self, normalized_text: str) -> bool:
        text = normalize_for_matching(normalized_text)
        complaint_phrases = {
            "ты очень странный",
            "ты странный",
            "ответ у ты кривой",
            "ответы у ты кривой",
            "ты очень глупый",
            "ты глупый",
            "ты ужасный",
            "ты ужасен",
            "ну ты капец",
            "не рад ты видеть",
            "не рад тебя видеть",
            "не понял мой запрос",
            "почему ты не понять",
        }
        complaint_words = {"глупый", "ужасный", "ужасен", "кривой", "странный", "дизлайк"}
        words = set(text.split())
        return any(phrase in text for phrase in complaint_phrases) or bool(words & complaint_words)

    def _has_negation_mismatch(self, normalized_text: str, matched_question: str | None) -> bool:
        if not matched_question:
            return False
        query_words = set(normalize_for_matching(normalized_text).split())
        matched_words = set(normalize_for_matching(matched_question).split())
        negations = {"не", "нет", "никогда"}
        return bool(query_words & negations) != bool(matched_words & negations)

    def _normalize_order_context(self, current: Any) -> dict[str, str]:
        return dict(current) if isinstance(current, dict) else {}

    def _merge_order_context(self, current: Any, entities: list[TextEntity]) -> dict[str, str]:
        context = self._normalize_order_context(current)
        entity_to_context = {
            "location": "delivery_location",
            "date": "delivery_date",
            "address": "delivery_address",
            "payment": "payment_method",
        }
        for entity in entities:
            key = entity_to_context.get(entity.kind)
            if key and entity.value:
                context[key] = entity.normalized if entity.kind == "payment" else entity.value
        return context
