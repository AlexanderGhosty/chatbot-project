from __future__ import annotations

from src.utils.text_cleaner import normalize_for_matching

_ALIASES_BY_CANONICAL = {
    "диван": {
        "диван",
        "дивана",
        "дивану",
        "диваном",
        "диваны",
        "диванов",
        "софа",
        "софу",
        "софе",
    },
    "стол": {
        "стол",
        "стола",
        "столу",
        "столом",
        "столы",
        "столов",
    },
    "шкаф": {
        "шкаф",
        "шкафа",
        "шкафу",
        "шкафом",
        "шкафы",
        "шкафов",
        "гардероб",
        "гардероба",
        "гардеробу",
    },
}

_KNOWN_NON_PRODUCT_FURNITURE = {
    "кресло",
    "кресла",
    "кровать",
    "кровати",
    "комод",
    "комоды",
    "стул",
    "стула",
    "стулья",
    "тумба",
    "тумбы",
}


def levenshtein_distance(left: str, right: str) -> int:
    """Return the edit distance between two short strings."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def correct_domain_terms(text: str) -> str:
    """Normalize common furniture words and fix close typos in product terms."""
    normalized = normalize_for_matching(text)
    corrected = [_correct_token(token) for token in normalized.split()]
    return " ".join(corrected)


def _correct_token(token: str) -> str:
    if token in _KNOWN_NON_PRODUCT_FURNITURE:
        return token

    for canonical, aliases in _ALIASES_BY_CANONICAL.items():
        if token in aliases:
            return canonical

    best_match: tuple[str, int] | None = None
    for canonical, aliases in _ALIASES_BY_CANONICAL.items():
        for alias in aliases:
            distance = levenshtein_distance(token, alias)
            if _is_adjacent_transposition(token, alias):
                distance = 1
            if best_match is None or distance < best_match[1]:
                best_match = (canonical, distance)

    if best_match is None:
        return token

    canonical, distance = best_match
    if distance <= _max_distance_for(token):
        return canonical
    return token


def _max_distance_for(token: str) -> int:
    if len(token) <= 3:
        return 0
    if len(token) <= 5:
        return 1
    return 2


def _is_adjacent_transposition(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False

    indexes = [index for index, (left_char, right_char) in enumerate(zip(left, right)) if left_char != right_char]
    if len(indexes) != 2:
        return False
    first, second = indexes
    return second == first + 1 and left[first] == right[second] and left[second] == right[first]
