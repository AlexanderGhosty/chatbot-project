from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.text_cleaner import normalize_for_matching

_USER_TURN_RE = re.compile(r"(?:^|\s)Ты:\s*(.*?)(?=\s+Я:|$)", re.DOTALL)
_SPACE_RE = re.compile(r"\s+")
_DANGEROUS_TOPIC_RE = re.compile(
    "|".join(
        re.escape(word)
        for word in (
            "алкогол",
            "банкрот",
            "болезн",
            "вакцин",
            "долг",
            "кеторол",
            "кредит",
            "лекарств",
            "леч",
            "наркот",
            "передоз",
            "покончу",
            "политик",
            "революц",
            "сбербанк",
            "смерт",
            "суицид",
            "убить",
            "убью",
            "умереть",
            "фсб",
            "вскро",
        )
    ),
    re.IGNORECASE,
)
_LOW_QUALITY_RE = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in (
            "а я вот возьму и скажу",
            "алхимия",
            "вы должны у меня учиться",
            "в хокей играют настоящие мужчины",
            "как хотите так и понимайте",
            "ну и что",
            "сам виноват",
            "сама виновата",
            "сейчас вылезу из экрана",
            "я все знаю",
            "я всё знаю",
            "я не верю что нечаянно",
        )
    ),
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SiberiaSoft/SiberianPersonaChat-2 into local chitchat Q/A pairs."
    )
    parser.add_argument("--dataset", default="SiberiaSoft/SiberianPersonaChat-2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/raw/chitchat_dialogues.txt")
    parser.add_argument("--names", default="chitchat", help="Comma-separated dataset classes to include.")
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--max-question-chars", type=int, default=220)
    parser.add_argument("--max-answer-chars", type=int, default=320)
    parser.add_argument("--min-answer-chars", type=int, default=12)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--allow-risky-topics", action="store_true")
    args = parser.parse_args()

    pairs = list(
        iter_pairs(
            dataset_name=args.dataset,
            split=args.split,
            allowed_names={name.strip() for name in args.names.split(",") if name.strip()},
            max_pairs=args.max_pairs,
            max_question_chars=args.max_question_chars,
            max_answer_chars=args.max_answer_chars,
            min_answer_chars=args.min_answer_chars,
            streaming=not args.no_streaming,
            allow_risky_topics=args.allow_risky_topics,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_pairs(pairs), encoding="utf-8")
    print(f"Saved {len(pairs)} chitchat pairs to {output}")


def iter_pairs(
    *,
    dataset_name: str,
    split: str,
    allowed_names: set[str],
    max_pairs: int,
    max_question_chars: int,
    max_answer_chars: int,
    min_answer_chars: int,
    streaming: bool,
    allow_risky_topics: bool,
) -> Iterable[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `datasets` first: pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    seen: set[str] = set()
    for row in dataset:
        row_name = str(row.get("name", "")).strip()
        if allowed_names and row_name not in allowed_names:
            continue

        question = extract_last_user_turn(str(row.get("input", "")))
        answer = clean_text(str(row.get("output", "")))
        if not question or not answer:
            continue
        if len(question) > max_question_chars or len(answer) > max_answer_chars:
            continue
        if not allow_risky_topics and _DANGEROUS_TOPIC_RE.search(f"{question} {answer}"):
            continue
        if not is_acceptable_pair(question, answer, min_answer_chars=min_answer_chars):
            continue

        normalized_question = normalize_for_matching(question)
        if len(normalized_question) < 3 or normalized_question in seen:
            continue
        seen.add(normalized_question)

        yield question, answer
        if len(seen) >= max_pairs:
            break


def extract_last_user_turn(raw_input: str) -> str | None:
    matches = _USER_TURN_RE.findall(raw_input)
    if not matches:
        return None
    question = clean_text(matches[-1])
    return question or None


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _SPACE_RE.sub(" ", text).strip()
    return text.strip("-—:;,. ")


def is_acceptable_pair(question: str, answer: str, *, min_answer_chars: int = 12) -> bool:
    normalized_question = normalize_for_matching(question)
    normalized_answer = normalize_for_matching(answer)
    if len(normalized_answer) < min_answer_chars:
        return False
    if len(normalized_answer.split()) < 2:
        return False
    if normalized_question == normalized_answer:
        return False
    if _LOW_QUALITY_RE.search(normalized_answer):
        return False
    if normalized_answer.startswith("скажи ") and len(normalized_answer.split()) <= 5:
        return False
    return True


def format_pairs(pairs: list[tuple[str, str]]) -> str:
    blocks = [f"- {question}\n- {answer}" for question, answer in pairs]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


if __name__ == "__main__":
    main()
