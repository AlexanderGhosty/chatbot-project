from __future__ import annotations

import re

_SPACE_RE = re.compile(r"\s+")
_ALLOWED_RE = re.compile(r"[^0-9a-zа-яё\s!?.,:;+\-]", re.IGNORECASE)
_REPEATED_PUNCT_RE = re.compile(r"([!?.,]){2,}")

_PROFANITY_REPLACEMENTS = {
    "блин": "неприятно",
}


def normalize_user_text(text: str) -> str:
    """Normalize text for NLP while preserving enough punctuation for UX."""
    cleaned = text.lower().replace("ё", "е").strip()
    cleaned = _ALLOWED_RE.sub(" ", cleaned)
    cleaned = _REPEATED_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)

    words = []
    for word in cleaned.split():
        words.append(_PROFANITY_REPLACEMENTS.get(word, word))
    cleaned = " ".join(words)
    return cleaned


def normalize_for_matching(text: str) -> str:
    """Return a stricter normalized form for vectorization and intent matching."""
    normalized = normalize_user_text(text)
    normalized = re.sub(r"[^0-9a-zа-яе\s+\-]", " ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()
