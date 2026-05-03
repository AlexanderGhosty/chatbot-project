from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speech.tts import TTSProcessor


async def main() -> None:
    model_name = os.getenv("TTS_MODEL_NAME", "v4_ru")
    speaker = os.getenv("TTS_SPEAKER", "xenia")
    output_path = Path("/tmp/silero-warmup.wav")

    processor = TTSProcessor(
        model_name=model_name,
        speaker=speaker,
        timeout_seconds=600.0,
        allow_espeak_fallback=False,
    )
    synthesized_path = await processor.synthesize_audio("Проверка синтеза речи.", str(output_path))
    path = Path(synthesized_path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Silero warmup did not produce audio: {path}")
    print(f"Silero TTS warmup completed: model={model_name} speaker={speaker} output={path}")


if __name__ == "__main__":
    asyncio.run(main())
