from __future__ import annotations


class ASRProcessor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # TODO: load GigaAM model and decoding config.

    async def transcribe_audio(self, wav_path: str) -> str:
        # TODO: run ASR inference in worker thread and return text transcript.
        raise NotImplementedError
