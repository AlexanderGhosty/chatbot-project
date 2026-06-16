from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.fuzzy import correct_domain_terms
from src.utils.text_cleaner import normalize_for_matching, normalize_user_text


@dataclass(slots=True)
class TextEntity:
    kind: str
    value: str
    normalized: str
    start: int = -1
    end: int = -1


@dataclass(slots=True)
class TextAnalysis:
    clean_text: str
    matching_text: str
    lemmas: list[str] = field(default_factory=list)
    entities: list[TextEntity] = field(default_factory=list)


class TextAnalyzer:
    """Normalize user text, lemmatize it when Natasha is available, and extract light entities."""

    _PRODUCT_ALIASES = {
        "диван": "диван",
        "дивана": "диван",
        "дивану": "диван",
        "диваном": "диван",
        "диваны": "диван",
        "диванов": "диван",
        "софа": "диван",
        "софу": "диван",
        "софе": "диван",
        "стол": "стол",
        "стола": "стол",
        "столу": "стол",
        "столом": "стол",
        "столы": "стол",
        "столов": "стол",
        "шкаф": "шкаф",
        "шкафа": "шкаф",
        "шкафу": "шкаф",
        "шкафом": "шкаф",
        "шкафы": "шкаф",
        "шкафов": "шкаф",
        "гардероб": "шкаф",
        "гардероба": "шкаф",
        "гардеробу": "шкаф",
    }
    _PAYMENT_WORDS = {
        "карта": "карта",
        "картой": "карта",
        "наличные": "наличные",
        "наличными": "наличные",
        "безнал": "безналичный расчет",
        "безналичный": "безналичный расчет",
        "рассрочка": "рассрочка",
        "рассрочку": "рассрочка",
    }
    _DATE_WORDS = {
        "сегодня",
        "завтра",
        "послезавтра",
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
        "понедельника",
        "вторника",
        "среду",
        "четверга",
        "пятницу",
        "субботу",
        "воскресенье",
    }
    _ADDRESS_MARKERS = {"адрес", "улица", "ул", "проспект", "пр", "дом", "квартира", "кв"}
    _KNOWN_LOCATIONS = {
        "москва",
        "санкт-петербург",
        "петербург",
        "казань",
        "новосибирск",
        "екатеринбург",
        "нижний",
        "самара",
        "омск",
        "уфа",
        "пермь",
        "воронеж",
        "краснодар",
    }

    def __init__(self, natasha_enabled: bool = True) -> None:
        self.natasha_enabled = natasha_enabled
        self._natasha_ready = False
        self._natasha_load_failed = False
        self._segmenter: Any = None
        self._morph_vocab: Any = None
        self._morph_tagger: Any = None
        self._ner_tagger: Any = None
        self._doc_cls: Any = None

    def analyze(self, text: str) -> TextAnalysis:
        clean_text = normalize_user_text(text)
        if not clean_text:
            return TextAnalysis(clean_text="", matching_text="")

        lemmas, ner_entities = self._analyze_with_natasha(clean_text)
        if not lemmas:
            lemmas = normalize_for_matching(clean_text).split()

        matching_text = correct_domain_terms(" ".join(lemmas))
        entities = self._extract_entities(clean_text=clean_text, matching_text=matching_text, ner_entities=ner_entities)
        return TextAnalysis(
            clean_text=clean_text,
            matching_text=matching_text,
            lemmas=matching_text.split(),
            entities=entities,
        )

    def _analyze_with_natasha(self, clean_text: str) -> tuple[list[str], list[TextEntity]]:
        if not self.natasha_enabled or not self._ensure_natasha_ready():
            return [], []

        try:
            doc = self._doc_cls(clean_text)
            doc.segment(self._segmenter)
            doc.tag_morph(self._morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self._morph_vocab)

            lemmas = [
                normalize_for_matching(getattr(token, "lemma", "") or getattr(token, "text", ""))
                for token in doc.tokens
            ]
            lemmas = [lemma for lemma in lemmas if lemma]

            doc.tag_ner(self._ner_tagger)
            entities: list[TextEntity] = []
            for span in doc.spans:
                span.normalize(self._morph_vocab)
                kind = self._map_natasha_kind(str(span.type))
                if kind is None:
                    continue
                value = str(getattr(span, "normal", None) or span.text).strip()
                if value:
                    entities.append(
                        TextEntity(
                            kind=kind,
                            value=value,
                            normalized=normalize_for_matching(value),
                            start=int(span.start),
                            end=int(span.stop),
                        )
                    )
            return lemmas, entities
        except Exception:
            return [], []

    def _ensure_natasha_ready(self) -> bool:
        if self._natasha_ready:
            return True
        if self._natasha_load_failed or importlib.util.find_spec("natasha") is None:
            self._natasha_load_failed = True
            return False

        try:
            from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Segmenter

            embedding = NewsEmbedding()
            self._segmenter = Segmenter()
            self._morph_vocab = MorphVocab()
            self._morph_tagger = NewsMorphTagger(embedding)
            self._ner_tagger = NewsNERTagger(embedding)
            self._doc_cls = Doc
            self._natasha_ready = True
            return True
        except Exception:
            self._natasha_load_failed = True
            return False

    def _extract_entities(
        self,
        *,
        clean_text: str,
        matching_text: str,
        ner_entities: list[TextEntity],
    ) -> list[TextEntity]:
        entities = list(ner_entities)
        entities.extend(self._extract_product_entities(clean_text, matching_text))
        entities.extend(self._extract_location_entities(clean_text))
        entities.extend(self._extract_payment_entities(clean_text))
        entities.extend(self._extract_date_entities(clean_text))
        entities.extend(self._extract_address_entities(clean_text))
        return self._deduplicate_entities(entities)

    def _extract_product_entities(self, clean_text: str, matching_text: str) -> list[TextEntity]:
        entities: list[TextEntity] = []
        clean_words = normalize_for_matching(clean_text).split()
        match_words = matching_text.split()
        for index, word in enumerate(clean_words):
            normalized = self._PRODUCT_ALIASES.get(word)
            if normalized is None and index < len(match_words):
                normalized = self._PRODUCT_ALIASES.get(match_words[index])
            if normalized is not None:
                entities.append(TextEntity(kind="product", value=word, normalized=normalized))
        return entities

    def _extract_payment_entities(self, clean_text: str) -> list[TextEntity]:
        entities = []
        for word in normalize_for_matching(clean_text).split():
            normalized = self._PAYMENT_WORDS.get(word)
            if normalized is not None:
                entities.append(TextEntity(kind="payment", value=word, normalized=normalized))
        return entities

    def _extract_location_entities(self, clean_text: str) -> list[TextEntity]:
        words = normalize_for_matching(clean_text).split()
        entities = []
        for index, word in enumerate(words):
            if word in self._KNOWN_LOCATIONS:
                value = word
                if word == "нижний" and index + 1 < len(words) and words[index + 1] == "новгород":
                    value = "нижний новгород"
                entities.append(TextEntity(kind="location", value=value, normalized=value))
            if word in {"город", "г", "в", "во"} and index + 1 < len(words):
                candidate = words[index + 1]
                if candidate in self._KNOWN_LOCATIONS:
                    entities.append(TextEntity(kind="location", value=candidate, normalized=candidate))
        return entities

    def _extract_date_entities(self, clean_text: str) -> list[TextEntity]:
        entities: list[TextEntity] = []
        normalized = normalize_for_matching(clean_text)
        for word in normalized.split():
            if word in self._DATE_WORDS:
                entities.append(TextEntity(kind="date", value=word, normalized=word))
        for match in re.finditer(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b", normalized):
            entities.append(TextEntity(kind="date", value=match.group(0), normalized=match.group(0), start=match.start(), end=match.end()))
        return entities

    def _extract_address_entities(self, clean_text: str) -> list[TextEntity]:
        words = normalize_for_matching(clean_text).split()
        if not words or not (set(words) & self._ADDRESS_MARKERS):
            return []

        start = next((index for index, word in enumerate(words) if word in self._ADDRESS_MARKERS), -1)
        if start < 0:
            return []
        value = " ".join(words[start : min(len(words), start + 6)])
        return [TextEntity(kind="address", value=value, normalized=value)]

    def _map_natasha_kind(self, raw_kind: str) -> str | None:
        return {
            "PER": "person",
            "LOC": "location",
            "ORG": "location",
        }.get(raw_kind)

    def _deduplicate_entities(self, entities: list[TextEntity]) -> list[TextEntity]:
        result = []
        seen = set()
        for entity in entities:
            key = (entity.kind, entity.normalized)
            if entity.normalized and key not in seen:
                result.append(entity)
                seen.add(key)
        return result
