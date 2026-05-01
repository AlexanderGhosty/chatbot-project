from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.bot.states import DialogueStates
from src.nlp.classifier import IntentClassifier, SentimentClassifier
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import VectorDatabase
from src.services.ad_campaign import AdCampaignManager
from src.services.dialogue_mgr import DialogueManager
from src.speech import SpeechProcessor
from src.speech.asr import ASRProcessor
from src.speech.tts import TTSProcessor
from src.utils.text_cleaner import normalize_for_matching, normalize_user_text


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


class CoreTests(unittest.IsolatedAsyncioTestCase):
    def test_text_normalization(self) -> None:
        self.assertEqual(normalize_user_text("  ПрИвЕт!!!   Как   дела? "), "привет! как дела?")
        self.assertEqual(normalize_for_matching("Шкаф Urban!!!"), "шкаф urban")

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
            self.assertEqual(state.state, DialogueStates.ad_offering.state)

            follow_up = await manager.process_text_message(
                chat_id=1,
                user_id=1,
                text="покажи стол",
                state=state,
            )
            self.assertIn("Стол Nordic", follow_up.text)

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


if __name__ == "__main__":
    unittest.main()
