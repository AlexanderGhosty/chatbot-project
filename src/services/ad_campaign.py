from __future__ import annotations

from dataclasses import dataclass

from src.nlp.classifier import IntentResult
from src.utils.text_cleaner import normalize_for_matching


@dataclass(slots=True)
class AdProduct:
    sku: str
    title: str
    description: str
    image_path: str


class AdCampaignManager:
    def __init__(self, products: list[AdProduct]) -> None:
        self.products = products

    @classmethod
    def default(cls) -> "AdCampaignManager":
        return cls(
            products=[
                AdProduct(
                    "sofa-001",
                    "Диван Loft",
                    "Двухместный диван для гостиной, компактный механизм трансформации, износостойкая ткань.",
                    "media/furniture/sofa.jpg",
                ),
                AdProduct(
                    "table-001",
                    "Стол Nordic",
                    "Обеденный стол из массива дерева на 4-6 человек, спокойный скандинавский дизайн.",
                    "media/furniture/table.jpg",
                ),
                AdProduct(
                    "wardrobe-001",
                    "Шкаф Urban",
                    "Трехстворчатый шкаф с зеркалом, штангой для одежды и глубокими полками.",
                    "media/furniture/wardrobe.jpg",
                ),
            ]
        )

    async def should_trigger_ad(
        self,
        *,
        intent: IntentResult,
        normalized_text: str,
        message_count: int,
        ad_message_threshold: int,
    ) -> bool:
        text = normalize_for_matching(normalized_text)
        if intent.label in {"buy_furniture", "ask_catalog", "product_sofa", "product_table", "product_wardrobe"}:
            return intent.confidence >= 0.35

        furniture_keywords = {
            "мебель",
            "диван",
            "стол",
            "шкаф",
            "кровать",
            "комод",
            "гостиная",
            "кухня",
            "интерьер",
            "купить",
            "заказать",
            "каталог",
        }
        if any(keyword in text for keyword in furniture_keywords):
            return True

        if intent.label in {"farewell", "decline"}:
            return False
        return message_count >= ad_message_threshold

    async def render_ad_offer(self) -> tuple[str, list[str]]:
        lines = [
            "Кстати, могу сразу предложить несколько популярных вариантов мебели:",
            "",
        ]
        for index, product in enumerate(self.products, start=1):
            lines.append(f"{index}. {product.title} — {product.description}")
        lines.extend(
            [
                "",
                "Если хотите, напишите «диван», «стол», «шкаф» или «каталог», и я помогу выбрать вариант.",
            ]
        )
        image_paths = [product.image_path for product in self.products if product.image_path]
        return "\n".join(lines), image_paths

    async def handle_ad_reply(self, normalized_text: str, intent: IntentResult) -> str:
        text = normalize_for_matching(normalized_text)
        if intent.label == "decline" or any(marker in text for marker in ("не надо", "нет", "потом", "не интересно")):
            return "Хорошо, не буду отвлекать рекламой. Если понадобится мебель, просто напишите, что ищете."

        selected = self._find_selected_product(text, intent)
        if selected is not None:
            return (
                f"{selected.title}: {selected.description}\n"
                "Могу подсказать размеры, стиль, сценарий использования или подготовить заказ."
            )

        if intent.label == "agree" or any(marker in text for marker in ("да", "каталог", "покажи", "интересно")):
            catalog = "\n".join(
                f"- {product.title}: {product.description}" for product in self.products
            )
            return f"Вот краткий каталог:\n{catalog}\n\nЧто смотрим подробнее?"

        return (
            "Могу показать подробнее диван Loft, стол Nordic или шкаф Urban. "
            "Напишите название товара или задайте вопрос по размерам, доставке и оплате."
        )

    def _find_selected_product(self, text: str, intent: IntentResult) -> AdProduct | None:
        label_to_sku = {
            "product_sofa": "sofa-001",
            "product_table": "table-001",
            "product_wardrobe": "wardrobe-001",
        }
        selected_sku = label_to_sku.get(intent.label)
        for product in self.products:
            title_words = normalize_for_matching(product.title).split()
            if selected_sku == product.sku or any(word and word in text for word in title_words):
                return product

        aliases = {
            "диван": "sofa-001",
            "софа": "sofa-001",
            "стол": "table-001",
            "обеденный": "table-001",
            "шкаф": "wardrobe-001",
            "гардероб": "wardrobe-001",
        }
        for keyword, sku in aliases.items():
            if keyword in text:
                return next((product for product in self.products if product.sku == sku), None)
        return None
