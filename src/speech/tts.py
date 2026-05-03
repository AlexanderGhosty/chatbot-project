from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)
_SILERO_IMPORT_LOCK = threading.RLock()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TTSProcessor:
    def __init__(
        self,
        model_name: str,
        speaker: str = "xenia",
        sample_rate: int = 48000,
        timeout_seconds: float = 180.0,
        allow_espeak_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.timeout_seconds = timeout_seconds
        self.allow_espeak_fallback = allow_espeak_fallback
        self._model = None
        self._torch = None
        self._load_error: Exception | None = None
        self._last_load_attempt_at = 0.0
        self._load_retry_after_seconds = 60.0
        self._last_error: str | None = None

    async def synthesize_audio(self, text: str, output_path: str) -> str:
        if self._is_local_tone():
            return self._synthesize_sync(text, output_path)
        if importlib.util.find_spec("torch") is not None:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._synthesize_sync, text, output_path),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("TTS synthesis timed out") from exc
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
        if self.allow_espeak_fallback and self._synthesize_with_espeak(text, wav_path):
            return self._convert_if_needed(wav_path, target)

        detail = f": {self._last_error}" if self._last_error else ""
        raise RuntimeError(f"TTS synthesis failed{detail}")

    def _is_local_tone(self) -> bool:
        return self.model_name in {"local-tone", "fallback-tone"}

    def _ensure_silero_loaded(self) -> bool:
        if self._model is not None and self._torch is not None:
            return True

        now = time.monotonic()
        if self._load_error is not None and now - self._last_load_attempt_at < self._load_retry_after_seconds:
            return False

        self._last_load_attempt_at = now
        try:
            import torch

            os.environ.setdefault("TORCH_HOME", str(Path("data/model_cache/torch").resolve()))
            model, _example_text = self._load_silero_from_torch_hub(torch)
            model.to(torch.device("cpu"))
            self._model = model
            self._torch = torch
            self._load_error = None
            logger.info("Silero TTS model loaded: model=%s speaker=%s", self.model_name, self.speaker)
            return True
        except Exception as exc:  # pragma: no cover - optional dependency/model cache/network.
            self._load_error = exc
            self._last_error = f"Silero load error: {exc}"
            logger.exception("Failed to load Silero TTS model")
            return False

    def _load_silero_from_torch_hub(self, torch):
        """
        Load Silero despite this project also being named `src`.

        Silero's hubconf imports `src.silero`. Because the bot package is also
        `src`, Python may resolve that import to `/app/src` and fail with
        `ModuleNotFoundError: No module named 'src.silero'`.

        Silero's repository contains a namespace package named `src`, while
        this project contains a regular package named `src`. A regular package
        wins over a namespace package even when torch.hub prepends Silero's
        repository to sys.path, so both sys.modules and the project root must be
        hidden while hubconf.py is imported.
        """
        with _SILERO_IMPORT_LOCK:
            src_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "src" or name.startswith("src.")
            }
            original_sys_path = list(sys.path)
            for name in src_modules:
                sys.modules.pop(name, None)

            try:
                sys.path = [
                    item
                    for item in sys.path
                    if not self._is_project_import_path(item)
                ]
                return torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker=self.model_name,
                    trust_repo=True,
                )
            finally:
                for name in list(sys.modules):
                    if (name == "src" or name.startswith("src.")) and name not in src_modules:
                        sys.modules.pop(name, None)
                sys.path = original_sys_path
                sys.modules.update(src_modules)

    def _is_project_import_path(self, path_entry: str) -> bool:
        if path_entry == "":
            path = Path.cwd()
        else:
            path = Path(path_entry)

        try:
            return path.resolve() == _PROJECT_ROOT
        except OSError:
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
        except Exception as exc:
            self._last_error = f"Silero synthesis error: {exc}"
            logger.exception("Failed to synthesize speech with Silero")
            return False

    def _synthesize_with_espeak(self, text: str, wav_path: Path) -> bool:
        espeak = shutil.which("espeak") or shutil.which("espeak-ng")
        if espeak is None:
            self._last_error = "espeak/espeak-ng binary is not installed"
            return False
        completed = subprocess.run(
            [espeak, "-v", "ru", "-w", str(wav_path), text[:900]],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and wav_path.exists():
            return True

        self._last_error = completed.stderr.strip() or "espeak exited without output"
        logger.warning("Failed to synthesize speech with espeak: %s", self._last_error)
        return False

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
                "-application",
                "voip",
                "-ar",
                "48000",
                "-ac",
                "1",
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
