from __future__ import annotations

from dataclasses import dataclass

from src.speech.asr import ASRProcessor
from src.speech.tts import TTSProcessor


@dataclass(slots=True)
class SpeechProcessor:
    asr: ASRProcessor
    tts: TTSProcessor

    async def transcribe_audio(self, wav_path: str) -> str:
        return await self.asr.transcribe_audio(wav_path)

    async def synthesize_audio(self, text: str, output_path: str) -> str:
        return await self.tts.synthesize_audio(text=text, output_path=output_path)
