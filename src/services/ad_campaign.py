from __future__ import annotations

from dataclasses import dataclass

from src.nlp.classifier import IntentResult
from src.utils.fuzzy import correct_domain_terms
from src.utils.text_cleaner import normalize_for_matching


@dataclass(slots=True)
class AdProduct:
    sku: str
    title: str
    description: str
    image_path: str
    dimensions: str
    style: str
    use_case: str
    price_hint: str


@dataclass(slots=True)
class AdReply:
    text: str
    selected_sku: str | None = None
    handled: bool = True
    declined: bool = False


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
                    "190 x 92 x 86 см, спальное место 140 x 190 см.",
                    "Современный лофт: нейтральная ткань, простая геометрия, металлические акценты.",
                    "Подойдет для гостиной, студии или комнаты, где диван иногда нужен как спальное место.",
                    "Средний ценовой сегмент; точная цена зависит от ткани и комплектации.",
                ),
                AdProduct(
                    "table-001",
                    "Стол Nordic",
                    "Обеденный стол из массива дерева на 4-6 человек, спокойный скандинавский дизайн.",
                    "media/furniture/table.jpg",
                    "140 x 80 x 75 см, комфортно для 4 человек, допустимо для 6.",
                    "Скандинавский стиль: светлое дерево, лаконичные ножки, матовое покрытие.",
                    "Хорош для кухни-гостиной, семейных ужинов и рабочего места на ноутбуке.",
                    "Средний ценовой сегмент; дороже компактных ЛДСП-моделей за счет массива.",
                ),
                AdProduct(
                    "wardrobe-001",
                    "Шкаф Urban",
                    "Трехстворчатый шкаф с зеркалом, штангой для одежды и глубокими полками.",
                    "media/furniture/wardrobe.jpg",
                    "180 x 60 x 220 см, три секции, глубина полок 55 см.",
                    "Городской минимализм: белые фасады, зеркало в центральной створке, спокойная фурнитура.",
                    "Подойдет для спальни, прихожей или гардеробной зоны в небольшой квартире.",
                    "Средний ценовой сегмент; цена зависит от фасадов и внутреннего наполнения.",
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
        del message_count, ad_message_threshold
        text = correct_domain_terms(normalized_text)
        if intent.label in {"buy_furniture", "ask_catalog", "product_sofa", "product_table", "product_wardrobe"}:
            return intent.confidence >= 0.35

        words = set(text.split())
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
            "каталог",
        }
        commerce_keywords = {"доставка", "оплата", "цена", "стоимость", "скидка", "наличие"}
        if words & furniture_keywords:
            return True
        if words & commerce_keywords:
            return True

        return False

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

    async def render_product_details(self, sku: str) -> tuple[str, str | None]:
        product = self.get_product(sku)
        if product is None:
            return "Не нашел такой товар в каталоге. Могу показать диван Loft, стол Nordic или шкаф Urban.", None
        return self.render_product_summary(product), product.image_path

    def get_product(self, sku: str) -> AdProduct | None:
        return next((product for product in self.products if product.sku == sku), None)

    async def handle_ad_reply(
        self,
        normalized_text: str,
        intent: IntentResult,
        selected_product_sku: str | None = None,
    ) -> AdReply:
        text = correct_domain_terms(normalized_text)
        words = set(text.split())
        if self._is_decline(text, intent):
            return AdReply(
                text="Хорошо, не буду отвлекать рекламой. Если понадобится мебель, просто напишите, что ищете.",
                declined=True,
            )

        selected = self.find_selected_product(text, intent)
        if selected is not None:
            return AdReply(text=self.render_product_summary(selected), selected_sku=selected.sku)

        current = self.get_product(selected_product_sku) if selected_product_sku else None
        if current is not None:
            if self._is_purchase_request(words):
                return AdReply(text=self.render_purchase_prompt(current), selected_sku=current.sku)
            if self._is_detail_request(words):
                return AdReply(text=self.render_product_details_text(current, words), selected_sku=current.sku)

        if self._is_catalog_request(words, intent):
            catalog = "\n".join(
                f"- {product.title}: {product.description}" for product in self.products
            )
            return AdReply(text=f"Вот краткий каталог:\n{catalog}\n\nЧто смотрим подробнее?")

        if not self.is_product_related(text, intent):
            return AdReply(text="", handled=False)

        return AdReply(
            text=(
                "Могу показать подробнее диван Loft, стол Nordic или шкаф Urban. "
                "Напишите название товара или задайте вопрос по размерам, доставке и оплате."
            )
        )

    def render_product_summary(self, product: AdProduct) -> str:
        return (
            f"{product.title}: {product.description}\n"
            f"Размеры: {product.dimensions}\n"
            f"Стиль: {product.style}\n"
            f"Сценарий: {product.use_case}\n"
            "Могу подсказать оплату, доставку или помочь подготовить заказ."
        )

    def render_product_details_text(self, product: AdProduct, words: set[str]) -> str:
        lines = [f"{product.title}:"]
        if words & {"размер", "размеры", "габарит", "габариты", "ширина", "высота", "глубина"}:
            lines.append(f"Размеры: {product.dimensions}")
        if words & {"стиль", "стили", "дизайн", "цвет", "интерьер"}:
            lines.append(f"Стиль: {product.style}")
        if words & {"сценарий", "сценарии", "использование", "подойдет", "куда", "комната"}:
            lines.append(f"Сценарий использования: {product.use_case}")
        if words & {"цена", "стоимость", "сколько", "дорого"}:
            lines.append(f"Цена: {product.price_hint}")
        if words & {"доставка", "доставить", "привезете"}:
            lines.append("Доставка: можно согласовать адрес и удобный интервал.")
        if words & {"оплата", "оплатить", "карта", "наличные"}:
            lines.append("Оплата: карта, наличные при получении или безналичный расчет.")
        if len(lines) == 1:
            lines.extend([f"Размеры: {product.dimensions}", f"Стиль: {product.style}", f"Сценарий: {product.use_case}"])
        lines.append("Если подходит, напишите «хочу купить» или уточните доставку.")
        return "\n".join(lines)

    def render_purchase_prompt(self, product: AdProduct) -> str:
        return (
            f"Отлично, зафиксировал интерес к {product.title}. "
            "Для подготовки заказа нужны город доставки, удобный день и способ оплаты. "
            f"Кратко по товару: {product.description}"
        )

    def find_selected_product(self, text: str, intent: IntentResult) -> AdProduct | None:
        text = correct_domain_terms(text)
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

    def is_product_related(self, text: str, intent: IntentResult) -> bool:
        text = correct_domain_terms(text)
        words = set(text.split())
        if self.find_selected_product(text, intent) is not None:
            return True
        if (
            self._is_catalog_request(words, intent)
            or self._is_detail_request(words)
            or self._is_purchase_request(words)
        ):
            return True
        return bool(words & {"мебель", "доставка", "оплата", "цена", "стоимость", "интерьер"})

    def _is_decline(self, text: str, intent: IntentResult) -> bool:
        words = set(text.split())
        decline_phrases = {"не надо", "не интересно", "не сейчас"}
        return (
            (intent.label == "decline" and intent.confidence >= 0.6)
            or bool(words & {"нет", "потом", "откажусь"})
            or any(phrase in text for phrase in decline_phrases)
        )

    def _is_catalog_request(self, words: set[str], intent: IntentResult) -> bool:
        return (intent.label == "agree" and intent.confidence >= 0.6) or bool(
            words & {"да", "каталог", "покажи", "интересно", "варианты", "какие", "еще", "есть", "другие"}
        )

    def _is_detail_request(self, words: set[str]) -> bool:
        return bool(
            words
            & {
                "размер",
                "размеры",
                "габарит",
                "габариты",
                "стиль",
                "стили",
                "дизайн",
                "сценарий",
                "сценарии",
                "использование",
                "использования",
                "цена",
                "стоимость",
                "доставка",
                "оплата",
            }
        )

    def _is_purchase_request(self, words: set[str]) -> bool:
        return bool(words & {"купить", "заказать", "оформить", "оформим", "беру", "возьму"})
