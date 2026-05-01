from __future__ import annotations

from collections.abc import Sequence

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from src.services.ad_campaign import AdProduct


def product_catalog_keyboard(products: Sequence[AdProduct]) -> InlineKeyboardMarkup:
    """Build product selection controls for the furniture ad flow."""
    rows = [
        [InlineKeyboardButton(text=product.title, callback_data=f"product:{product.sku}")]
        for product in products
    ]
    rows.append([InlineKeyboardButton(text="Каталог", callback_data="catalog")])
    return InlineKeyboardMarkup(inline_keyboard=rows)
