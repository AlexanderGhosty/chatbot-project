from __future__ import annotations

from collections.abc import Sequence

try:
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
except ImportError:  # Allows service tests to import handlers without aiogram installed.
    class InlineKeyboardButton:  # type: ignore[no-redef]
        def __init__(self, *, text: str, callback_data: str) -> None:
            self.text = text
            self.callback_data = callback_data

    class InlineKeyboardMarkup:  # type: ignore[no-redef]
        def __init__(self, *, inline_keyboard) -> None:
            self.inline_keyboard = inline_keyboard

from src.services.ad_campaign import AdProduct


def product_catalog_keyboard(products: Sequence[AdProduct]) -> InlineKeyboardMarkup:
    """Build product selection controls for the furniture ad flow."""
    rows = [
        [InlineKeyboardButton(text=product.title, callback_data=f"product:{product.sku}")]
        for product in products
    ]
    rows.append([InlineKeyboardButton(text="Каталог", callback_data="catalog")])
    return InlineKeyboardMarkup(inline_keyboard=rows)
