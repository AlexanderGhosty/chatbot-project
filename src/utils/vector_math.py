from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[0-9a-zа-яе]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("ё", "е"))


def hashed_text_vector(text: str, *, dimensions: int = 384) -> list[float]:
    """Create a deterministic dense vector from tokens and character n-grams."""
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    features: list[str] = []

    for token in tokens:
        features.append(f"tok:{token}")
        padded = f"_{token}_"
        for size in (3, 4, 5):
            if len(padded) >= size:
                features.extend(f"ch:{padded[index:index + size]}" for index in range(len(padded) - size + 1))

    if not features:
        return vector

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    return l2_normalize(vector)


def l2_normalize(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    score = sum(left[index] * right[index] for index in range(size)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


def cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - cosine_similarity(left, right)
