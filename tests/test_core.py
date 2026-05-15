from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import load_config
from src.bot.handlers import _send_chat_action_safely
from src.bot.states import DialogueStates
from src.nlp.chitchat_safety import is_safe_chitchat_answer, is_safe_chitchat_pair
from src.nlp.classifier import IntentClassifier, SentimentClassifier
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import VectorDatabase, load_dialogue_pairs
from src.services.ad_campaign import AdCampaignManager
from src.services.dialogue_mgr import DialogueManager
from src.speech import SpeechProcessor
from src.speech.asr import ASRProcessor
from src.speech.tts import TTSProcessor, _PROJECT_ROOT
from src.utils.fuzzy import correct_domain_terms, levenshtein_distance
from src.utils.text_cleaner import normalize_for_matching, normalize_user_text
from scripts.import_siberian_persona_chat import is_acceptable_pair


class FakeState:
    def __init__(self) -> None:
        self.data = {}
        self.state = None

    async def get_data(self):
        return dict(self.data)

    async def update_data(self, **kwargs):
        self.data.update(kwargs)

    async def set_data(self, data):
        self.data = dict(data)

    async def get_state(self):
        return self.state

    async def set_state(self, state):
        self.state = getattr(state, "state", str(state))


class FakeChat:
    id = 1


class FailingBot:
    async def send_chat_action(self, **_kwargs):
        raise RuntimeError("telegram timeout")


class FakeMessage:
    bot = FailingBot()
    chat = FakeChat()


class CoreTests(unittest.IsolatedAsyncioTestCase):
    def test_text_normalization(self) -> None:
        self.assertEqual(normalize_user_text("  ПрИвЕт!!!   Как   дела? "), "привет! как дела?")
        self.assertEqual(normalize_for_matching("Шкаф Urban!!!"), "шкаф urban")

    def test_domain_spell_correction(self) -> None:
        self.assertEqual(levenshtein_distance("шкаоф", "шкаф"), 1)
        self.assertEqual(correct_domain_terms("Я хочу купить шкаоф"), "я хочу купить шкаф")
        self.assertEqual(correct_domain_terms("Нужен шкафф"), "нужен шкаф")
        self.assertEqual(correct_domain_terms("Покажи диваан"), "покажи диван")
        self.assertEqual(correct_domain_terms("Нужен стло"), "нужен стол")

    async def test_intent_classifier(self) -> None:
        classifier = IntentClassifier("local-intents", intents_path="data/raw/intents.json")
        result = await classifier.predict("привет")
        self.assertEqual(result.label, "greeting")
        self.assertGreaterEqual(result.confidence, 0.9)

    async def test_retrieval_fallback_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = EmbeddingEngine("local-hash")
            db = VectorDatabase(
                db_path=tmp_dir,
                collection_name="test_dialogues",
                dialogues_path="data/raw/dialogues.txt",
                use_chroma=False,
            )
            await db.ensure_ready(engine)
            query = await engine.encode("как выбрать шкаф")
            result = await db.search_answer(query)
            self.assertLess(result.distance, 0.6)
            self.assertIn("шкаф", result.matched_question or "")

    def test_load_dialogue_pairs_can_skip_seed_dialogues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "chitchat.txt"
            path.write_text("- Ты где?\n- Мой дом - сервер в интернете.\n", encoding="utf-8")

            pairs = load_dialogue_pairs(path, include_seed_dialogues=False)

        self.assertEqual(pairs, [("ты где", "Мой дом - сервер в интернете.")])

    def test_load_dialogue_pairs_can_filter_unsafe_chitchat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "chitchat.txt"
            path.write_text(
                "- Ты говно\n- Когда человек говно то и программы не помогут\n\n"
                "- Ты человек?\n- Я бот, но стараюсь общаться понятно.\n",
                encoding="utf-8",
            )

            pairs = load_dialogue_pairs(path, include_seed_dialogues=False, filter_unsafe_pairs=True)

        self.assertEqual(pairs, [("ты человек", "Я бот, но стараюсь общаться понятно.")])

    async def test_chitchat_fallback_is_used_after_domain_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            domain_path = tmp_path / "domain.txt"
            chitchat_path = tmp_path / "chitchat.txt"
            domain_path.write_text("- Как выбрать шкаф?\n- Ориентируйтесь на размеры ниши.\n", encoding="utf-8")
            chitchat_path.write_text("- Ты где?\n- Мой дом - сервер в интернете.\n", encoding="utf-8")

            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(
                    str(tmp_path / "domain_db"),
                    "domain",
                    str(domain_path),
                    use_chroma=False,
                    include_seed_dialogues=False,
                ),
                chitchat_vector_db=VectorDatabase(
                    str(tmp_path / "chitchat_db"),
                    "chitchat",
                    str(chitchat_path),
                    use_chroma=False,
                    include_seed_dialogues=False,
                ),
                chitchat_retrieval_distance_threshold=0.7,
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.2,
                ad_message_threshold=99,
            )

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Ты где?",
                state=FakeState(),
            )

        self.assertIn("сервер", response.text)

    async def test_dialogue_manager_ad_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=99,
            )
            state = FakeState()
            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Хочу купить диван",
                state=state,
            )
            self.assertIn("Диван Loft", response.text)
            self.assertEqual(state.state, DialogueStates.ad_follow_up.state)
            self.assertEqual(state.data["selected_product_sku"], "sofa-001")

            follow_up = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="покажи стол",
                state=state,
            )
            self.assertIn("Стол Nordic", follow_up.text)
            self.assertEqual(state.data["selected_product_sku"], "table-001")

            details = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Подскажи размеры и стили и сценарий использования.",
                state=state,
            )
            self.assertIn("140 x 80 x 75", details.text)
            self.assertIn("Скандинавский", details.text)

            purchase = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Я хочу купить его.",
                state=state,
            )
            self.assertIn("Стол Nordic", purchase.text)
            self.assertIn("подготовки заказа", purchase.text)

    async def test_ad_threshold_does_not_interrupt_non_furniture_help(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=3,
            )
            state = FakeState()
            await state.update_data(message_count=2)

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Что ты умеешь?",
                state=state,
            )

            self.assertIn("отвечаю на вопросы", response.text)
            self.assertEqual(len(response.follow_ups), 0)
            self.assertEqual(state.state, DialogueStates.normal_chat.state)

    async def test_furniture_context_keeps_regular_answer_and_adds_ad_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=3,
            )
            state = FakeState()
            await state.update_data(message_count=2)

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Сколько стоит мебель?",
                state=state,
            )

            self.assertIn("Цена зависит", response.text)
            self.assertEqual(len(response.follow_ups), 1)
            self.assertIn("Диван Loft", response.follow_ups[0].text)
            self.assertEqual(state.state, DialogueStates.ad_offering.state)

    async def test_spending_bridge_can_add_soft_ad_in_chitchat_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=4,
            )
            state = FakeState()
            await state.update_data(message_count=5, chitchat_mode=True)

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Вчера зарплату получил, теперь думаю на что потратить",
                state=state,
            )

            self.assertIn("повседневный комфорт", response.text)
            self.assertEqual(len(response.follow_ups), 1)
            self.assertIn("Диван Loft", response.follow_ups[0].text)
            self.assertEqual(state.state, DialogueStates.ad_offering.state)

    async def test_chat_log_regressions_use_safe_overrides_or_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                chitchat_vector_db=VectorDatabase(
                    tmp_dir,
                    "chitchat",
                    "data/raw/chitchat_dialogues.txt",
                    use_chroma=False,
                    include_seed_dialogues=False,
                    filter_unsafe_pairs=True,
                ),
                chitchat_retrieval_distance_threshold=0.22,
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=4,
            )
            state = FakeState()
            bad_fragments = ["говно", "глюк", "пизд", "видеокарта", "прогнозу погоды"]

            for text in ("❤️", "По лесу", "Странные у тебя ответы", "Приколдес", "Ты странный", "лол"):
                response = await manager.process_text_message(
                    chat_id=1,
                    user_id=1,
                    text=text,
                    state=state,
                )
                for fragment in bad_fragments:
                    self.assertNotIn(fragment, response.text.lower(), text)

    async def test_chat_log_smalltalk_uses_conversation_routes_without_ads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=3,
            )
            state = FakeState()
            bad_fragments = [
                "подобрать мебель",
                "прогнозу погоды",
                "Диван Loft",
                "каталог",
                "Спасибо, хорошо",
            ]

            for text in (
                "Давай просто поговорим о всяком?",
                "Как тебе моя прическа?",
                "Я люблю гулять",
                "Ты мое солнышко",
                "Ты любишь хоккей?",
                "Новости",
                "Нечаянно",
            ):
                response = await manager.process_text_message(
                    chat_id=1,
                    user_id=1,
                    text=text,
                    state=state,
                )
                self.assertEqual(response.follow_ups, [], text)
                for fragment in bad_fragments:
                    self.assertNotIn(fragment, response.text, text)

    async def test_typo_in_product_request_selects_wardrobe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=99,
            )

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Я хочу купить шкаоф",
                state=FakeState(),
            )

            self.assertIn("Шкаф Urban", response.text)
            self.assertNotIn("Диван Loft", response.text)

    async def test_ad_follow_up_catalog_phrase_shows_other_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=99,
            )
            state = FakeState()
            await state.set_state(DialogueStates.ad_follow_up)
            await state.update_data(message_count=8, selected_product_sku="wardrobe-001")

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Прикольно, а какие еще есть",
                state=state,
            )

            self.assertIn("Вот краткий каталог", response.text)
            self.assertIn("Диван Loft", response.text)
            self.assertIn("Стол Nordic", response.text)

    async def test_other_sofa_request_reports_no_alternative_instead_of_repeating_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=99,
            )
            state = FakeState()
            await state.set_state(DialogueStates.ad_follow_up)
            await state.update_data(message_count=8, selected_product_sku="sofa-001")

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="Есть другой диван",
                state=state,
            )

            self.assertIn("один диван", response.text)
            self.assertIn("Стол Nordic", response.text)
            self.assertIn("Шкаф Urban", response.text)
            self.assertNotIn("Размеры: 190 x 92", response.text)

    async def test_offtopic_in_ad_state_returns_regular_answer_not_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = DialogueManager(
                intent_classifier=IntentClassifier("local-intents", intents_path="data/raw/intents.json"),
                sentiment_classifier=SentimentClassifier("local-lexicon"),
                embedding_engine=EmbeddingEngine("local-hash"),
                vector_db=VectorDatabase(tmp_dir, "dialogues", "data/raw/dialogues.txt", use_chroma=False),
                speech_processor=SpeechProcessor(ASRProcessor("ctc"), TTSProcessor("local-tone")),
                ad_campaign_manager=AdCampaignManager.default(),
                retrieval_distance_threshold=0.7,
                ad_message_threshold=3,
            )
            state = FakeState()
            await state.set_state(DialogueStates.ad_follow_up)
            await state.update_data(message_count=4, selected_product_sku="table-001")

            response = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="А ты знаешь, какая погода в Москве завтра?",
                state=state,
            )

            self.assertIn("не подключен к прогнозу погоды", response.text)
            self.assertNotIn("Вот краткий каталог", response.text)
            self.assertEqual(state.state, DialogueStates.normal_chat.state)

    async def test_speech_sidecar_and_tts_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "voice.wav"
            wav_path.write_bytes(b"placeholder")
            wav_path.with_suffix(".txt").write_text("привет", encoding="utf-8")

            asr = ASRProcessor("ctc")
            self.assertEqual(await asr.transcribe_audio(str(wav_path)), "привет")

            tts = TTSProcessor("local-tone")
            out_path = await tts.synthesize_audio("тестовый ответ", str(Path(tmp_dir) / "answer.wav"))
            self.assertTrue(Path(out_path).exists())

    async def test_production_tts_does_not_generate_tone_on_backend_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "answer.wav"
            tts = TTSProcessor("v4_ru")
            with (
                patch("src.speech.tts.importlib.util.find_spec", return_value=None),
                patch.object(TTSProcessor, "_synthesize_with_silero", return_value=False),
                patch.object(TTSProcessor, "_synthesize_with_espeak", return_value=False) as espeak_mock,
            ):
                with self.assertRaises(RuntimeError):
                    await tts.synthesize_audio("тестовый ответ", str(out_path))
            espeak_mock.assert_not_called()
            self.assertFalse(out_path.exists())

    async def test_espeak_fallback_is_explicitly_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "answer.wav"
            tts = TTSProcessor("v4_ru", allow_espeak_fallback=True)
            with (
                patch("src.speech.tts.importlib.util.find_spec", return_value=None),
                patch.object(TTSProcessor, "_synthesize_with_silero", return_value=False),
                patch.object(TTSProcessor, "_synthesize_with_espeak", return_value=True) as espeak_mock,
                patch.object(TTSProcessor, "_convert_if_needed", return_value=str(out_path)),
            ):
                self.assertEqual(await tts.synthesize_audio("тестовый ответ", str(out_path)), str(out_path))
            espeak_mock.assert_called_once()

    async def test_chat_action_failure_does_not_break_handler_flow(self) -> None:
        await _send_chat_action_safely(message=FakeMessage(), action="record_voice")

    def test_silero_torch_hub_load_hides_project_src_package(self) -> None:
        import sys

        project_src = sys.modules["src"]

        class FakeHub:
            def __init__(self) -> None:
                self.project_src_was_hidden = False
                self.project_root_was_hidden = False

            def load(self, **_kwargs):
                import sys

                self.project_src_was_hidden = sys.modules.get("src") is None
                self.project_root_was_hidden = all(
                    (Path.cwd() if item == "" else Path(item)).resolve() != _PROJECT_ROOT
                    for item in sys.path
                )
                sys.modules["src"] = object()
                sys.modules["src.silero"] = object()
                return object(), "example"

        class FakeTorch:
            hub = FakeHub()

        model, example = TTSProcessor("v4_ru")._load_silero_from_torch_hub(FakeTorch)

        self.assertIsNotNone(model)
        self.assertEqual(example, "example")
        self.assertTrue(FakeTorch.hub.project_src_was_hidden)
        self.assertTrue(FakeTorch.hub.project_root_was_hidden)
        self.assertIs(sys.modules["src"], project_src)
        self.assertNotIn("src.silero", sys.modules)

    def test_voice_logging_can_be_toggled_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TELEGRAM_BOT_TOKEN": "123:test",
                "VOICE_LOGGING_ENABLED": "true",
                "DIALOGUE_LOGGING_ENABLED": "true",
                "TTS_ALLOW_ESPEAK_FALLBACK": "false",
                "CHITCHAT_ENABLED": "true",
                "CHITCHAT_DIALOGUES_PATH": "data/raw/chitchat_dialogues.txt",
            },
            clear=True,
        ):
            config = load_config()

        self.assertTrue(config.voice_logging_enabled)
        self.assertTrue(config.dialogue_logging_enabled)
        self.assertFalse(config.tts_allow_espeak_fallback)
        self.assertTrue(config.chitchat_enabled)
        self.assertEqual(config.chitchat_dialogues_path, "data/raw/chitchat_dialogues.txt")
        self.assertEqual(config.chitchat_retrieval_distance_threshold, 0.22)
        self.assertTrue(config.prewarm_vector_indexes)

    def test_siberian_persona_chat_quality_filter_rejects_bad_pairs(self) -> None:
        self.assertFalse(is_acceptable_pair("Ты любишь хоккей", "В хокей играют настоящие мужчины"))
        self.assertFalse(is_acceptable_pair("Нечаянно", "Я не верю что нечаянно"))
        self.assertFalse(is_acceptable_pair("Новости", "Скажи расскажи мне новости"))
        self.assertFalse(is_acceptable_pair("Ты говно", "Когда человек говно то и программы не помогут"))
        self.assertFalse(is_acceptable_pair("Ты глюк", "Сами вы глюк, я программа"))
        self.assertFalse(is_acceptable_pair("Лол", "Видеокарта у вас слабая"))
        self.assertTrue(is_acceptable_pair("Ты мое солнышко", "Вы тоже очень милый человек"))

    def test_chitchat_safety_filter(self) -> None:
        self.assertFalse(is_safe_chitchat_answer("Когда человек говно то и программы не помогут"))
        self.assertFalse(is_safe_chitchat_answer("Сами вы странные. я - загадочная!"))
        self.assertFalse(is_safe_chitchat_answer("Видеокарта у вас слабая"))
        self.assertFalse(is_safe_chitchat_pair("Ты глюк", "Сами вы глюк, я программа"))
        self.assertTrue(is_safe_chitchat_pair("Ты человек?", "Я бот, но стараюсь общаться понятно."))


if __name__ == "__main__":
    unittest.main()
