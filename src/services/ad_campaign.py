from __future__ import annotations

from dataclasses import dataclass

from src.nlp.classifier import IntentResult


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
        # TODO: replace with repository/API-backed catalog source.
        return cls(
            products=[
                AdProduct("sofa-001", "Диван Loft", "Двухместный диван для гостиной", "media/furniture/sofa.jpg"),
                AdProduct("table-001", "Стол Nordic", "Обеденный стол из массива дерева", "media/furniture/table.jpg"),
                AdProduct("wardrobe-001", "Шкаф Urban", "Трехстворчатый шкаф с зеркалом", "media/furniture/wardrobe.jpg"),
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
        # TODO: implement explicit intent + keyword + threshold trigger strategy.
        raise NotImplementedError

    async def render_ad_offer(self) -> tuple[str, list[str]]:
        # TODO: generate ad text/cards with top-N products.
        raise NotImplementedError

    async def handle_ad_reply(self, normalized_text: str, intent: IntentResult) -> str:
        # TODO: interpret user reaction (agree/decline/clarify) and return next ad step.
        raise NotImplementedError
