from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


class ASRProcessor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._load_error: Exception | None = None

    async def transcribe_audio(self, wav_path: str) -> str:
        sidecar = Path(wav_path).with_suffix(".txt")
        if sidecar.exists() or importlib.util.find_spec("gigaam") is None:
            return self._transcribe_sync(wav_path)
        return await asyncio.to_thread(self._transcribe_sync, wav_path)

    def _transcribe_sync(self, wav_path: str) -> str:
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text

        if not self._ensure_model_loaded():
            raise RuntimeError(
                "GigaAM is not available. Install it with "
                "`pip install git+https://github.com/salute-developers/GigaAM.git`."
            )

        text = self._model.transcribe(str(path))
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)
        text = str(text).strip()
        if not text:
            raise RuntimeError("ASR returned an empty transcription")
        return text

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import gigaam

            self._model = gigaam.load_model(self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - depends on optional dependency/model cache.
            self._load_error = exc
            return False
