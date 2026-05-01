from __future__ import annotations

import asyncio
import importlib.util
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path


class TTSProcessor:
    def __init__(self, model_name: str, speaker: str = "xenia", sample_rate: int = 48000) -> None:
        self.model_name = model_name
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._model = None
        self._torch = None
        self._load_error: Exception | None = None

    async def synthesize_audio(self, text: str, output_path: str) -> str:
        if self._is_local_tone():
            return self._synthesize_sync(text, output_path)
        if importlib.util.find_spec("torch") is not None:
            return await asyncio.to_thread(self._synthesize_sync, text, output_path)
        return self._synthesize_sync(text, output_path)

    def _synthesize_sync(self, text: str, output_path: str) -> str:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        wav_path = target.with_suffix(".wav") if target.suffix.lower() != ".wav" else target
        if self._is_local_tone():
            self._write_fallback_tone(wav_path, duration_seconds=max(1.0, min(5.0, len(text) / 35.0)))
            return self._convert_if_needed(wav_path, target)

        if self._synthesize_with_silero(text, wav_path):
            return self._convert_if_needed(wav_path, target)
        if self._synthesize_with_espeak(text, wav_path):
            return self._convert_if_needed(wav_path, target)

        self._write_fallback_tone(wav_path, duration_seconds=max(1.0, min(5.0, len(text) / 35.0)))
        return self._convert_if_needed(wav_path, target)

    def _is_local_tone(self) -> bool:
        return self.model_name in {"local-tone", "fallback-tone"}

    def _ensure_silero_loaded(self) -> bool:
        if self._model is not None and self._torch is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch

            model, _example_text = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker=self.model_name,
                trust_repo=True,
            )
            model.to(torch.device("cpu"))
            self._model = model
            self._torch = torch
            return True
        except Exception as exc:  # pragma: no cover - optional dependency/model cache/network.
            self._load_error = exc
            return False

    def _synthesize_with_silero(self, text: str, wav_path: Path) -> bool:
        if not self._ensure_silero_loaded():
            return False

        try:
            audio = self._model.apply_tts(
                text=text[:900],
                speaker=self.speaker,
                sample_rate=self.sample_rate,
            )
            if hasattr(audio, "detach"):
                values = audio.detach().cpu().tolist()
            else:
                values = list(audio)
            self._write_wav(wav_path, values, self.sample_rate)
            return True
        except Exception:
            return False

    def _synthesize_with_espeak(self, text: str, wav_path: Path) -> bool:
        espeak = shutil.which("espeak") or shutil.which("espeak-ng")
        if espeak is None:
            return False
        completed = subprocess.run(
            [espeak, "-v", "ru", "-w", str(wav_path), text[:900]],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0 and wav_path.exists()

    def _convert_if_needed(self, wav_path: Path, target: Path) -> str:
        if target.suffix.lower() == ".wav":
            return str(wav_path)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return str(wav_path)

        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(wav_path),
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and target.exists():
            return str(target)
        return str(wav_path)

    def _write_fallback_tone(self, wav_path: Path, duration_seconds: float) -> None:
        frame_count = int(self.sample_rate * duration_seconds)
        values = [
            0.08 * math.sin(2.0 * math.pi * 440.0 * frame / self.sample_rate)
            for frame in range(frame_count)
        ]
        self._write_wav(wav_path, values, self.sample_rate)

    def _write_wav(self, wav_path: Path, values: list[float], sample_rate: int) -> None:
        with wave.open(str(wav_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            frames = bytearray()
            for value in values:
                clipped = max(-1.0, min(1.0, float(value)))
                frames.extend(struct.pack("<h", int(clipped * 32767)))
            audio_file.writeframes(bytes(frames))
