from __future__ import annotations

import re

_SPACE_RE = re.compile(r"\s+")


def normalize_user_text(text: str) -> str:
    # TODO: add stronger normalization (punctuation policy, lowercase, profanity guard).
    cleaned = text.strip()
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned
