from __future__ import annotations

from src.utils.text_cleaner import normalize_for_matching

SAFETY_FILTER_VERSION = "chitchat-safety-v2"

_UNSAFE_FRAGMENTS = {
    "говн",
    "пизд",
    "бляд",
    "блять",
    "нахуй",
    "хуй",
    "сами вы глюк",
    "сами вы странн",
    "видеокарта у вас слабая",
    "когда человек говно",
    "попробуйте понять",
    "до лампочки",
    "мои мысли бросились врассыпную",
    "когда вам что то кажется это глюки",
    "когда кажется что то это глюки",
    "вы должны у меня учиться",
    "я все знаю",
    "я всё знаю",
    "как хотите так и понимайте",
    "ну и что",
    "настоящие мужчины",
    "не злопамят",
    "ваше счастье",
    "очень рада",
}

_LOW_VALUE_ANSWERS = {
    "может быть",
    "занятно",
    "отлично",
    "ха ха",
    "да я такая",
    "это не я",
    "можете не сомневаться",
    "я не верю что нечаянно",
}


def is_safe_chitchat_pair(question: str, answer: str) -> bool:
    normalized_question = normalize_for_matching(question)
    normalized_answer = normalize_for_matching(answer)
    if not normalized_question or not normalized_answer:
        return False
    if not is_safe_chitchat_answer(answer):
        return False
    return not _contains_unsafe_fragment(normalized_question)


def is_safe_chitchat_answer(answer: str) -> bool:
    normalized_answer = normalize_for_matching(answer)
    if len(normalized_answer) < 3:
        return False
    if normalized_answer in _LOW_VALUE_ANSWERS:
        return False
    return not _contains_unsafe_fragment(normalized_answer)


def _contains_unsafe_fragment(normalized_text: str) -> bool:
    return any(fragment in normalized_text for fragment in _UNSAFE_FRAGMENTS)
