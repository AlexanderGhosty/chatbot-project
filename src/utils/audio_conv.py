from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def convert_ogg_to_wav(ogg_path: str, wav_path: str) -> None:
    """Convert Telegram OGG/Opus voice to mono 16 kHz WAV for ASR."""
    source = Path(ogg_path)
    target = Path(wav_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file does not exist: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".wav":
        shutil.copyfile(source, target)
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to convert Telegram OGG audio to WAV")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {completed.stderr.strip()}")
