from __future__ import annotations


class TTSProcessor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # TODO: load Silero TTS model and target voice profile.

    async def synthesize_audio(self, text: str, output_path: str) -> str:
        # TODO: synthesize speech and persist result to output_path.
        raise NotImplementedError
