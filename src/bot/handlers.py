from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message

from src.bot.keyboards import product_catalog_keyboard
from src.bot.states import DialogueStates
from src.services.dialogue_mgr import BotResponse, DialogueManager

logger = logging.getLogger(__name__)
router = Router(name="core-handlers")


@dataclass(slots=True)
class HandlerDeps:
    dialogue_manager: DialogueManager
    temp_audio_dir: str = "media/temp_audio"


def bind_handlers(deps: HandlerDeps) -> Router:
    """Bind runtime dependencies into aiogram handlers."""
    @router.message(Command("start"))
    async def on_start_command(message: Message, state: FSMContext) -> None:
        await state.set_state(DialogueStates.normal_chat)
        await state.set_data({"message_count": 0})
        await message.answer(
            "Привет! Я помогу с выбором мебели и отвечу на вопросы в текстовом или голосовом формате."
        )

    @router.message(Command("help"))
    async def on_help_command(message: Message) -> None:
        await message.answer(
            "Команды: /start, /help.\n"
            "Поддерживаются текст и голос. При подходящем контексте предложу товары из каталога."
        )

    @router.message(F.voice)
    async def on_voice_message(message: Message, state: FSMContext) -> None:
        if not message.voice:
            return

        await message.bot.send_chat_action(chat_id=message.chat.id, action="record_voice")
        response = await deps.dialogue_manager.process_voice_message(
            chat_id=message.chat.id,
            user_id=message.from_user.id if message.from_user else 0,
            voice_file_id=message.voice.file_id,
            state=state,
            temp_audio_dir=deps.temp_audio_dir,
            bot=message.bot,
        )
        await _send_response(message=message, response=response, deps=deps)

    @router.message(F.text)
    async def on_text_message(message: Message, state: FSMContext) -> None:
        if not message.text:
            return

        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
        response = await deps.dialogue_manager.process_text_message(
            chat_id=message.chat.id,
            user_id=message.from_user.id if message.from_user else 0,
            text=message.text,
            state=state,
        )
        await _send_response(message=message, response=response, deps=deps)

    @router.callback_query(F.data == "catalog")
    async def on_catalog_callback(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if not isinstance(callback.message, Message):
            return

        await state.set_state(DialogueStates.ad_follow_up)
        await state.update_data(ad_declined=False)
        text, image_paths = await deps.dialogue_manager.ad_campaign_manager.render_ad_offer()
        await _send_response(
            message=callback.message,
            response=BotResponse(text=text, image_paths=image_paths),
            deps=deps,
        )

    @router.callback_query(F.data.startswith("product:"))
    async def on_product_callback(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if not isinstance(callback.message, Message):
            return

        sku = (callback.data or "").split(":", 1)[1]
        text, image_path = await deps.dialogue_manager.ad_campaign_manager.render_product_details(sku)
        await state.set_state(DialogueStates.ad_follow_up)
        await state.update_data(ad_declined=False)
        await _send_response(
            message=callback.message,
            response=BotResponse(text=text, image_paths=[image_path] if image_path else []),
            deps=deps,
        )

    @router.errors()
    async def on_handler_error(event) -> None:
        logger.exception("Unhandled update processing error", extra={"event": str(event)})

    return router


async def _send_response(message: Message, response: BotResponse, deps: HandlerDeps) -> None:
    """Deliver assembled response payload to Telegram."""
    keyboard = product_catalog_keyboard(deps.dialogue_manager.ad_campaign_manager.products)

    if response.image_paths:
        await message.answer(response.text, reply_markup=keyboard)
        for image_path in response.image_paths:
            path = Path(image_path)
            if path.exists():
                await message.answer_photo(photo=FSInputFile(path))
        return

    if response.send_voice and response.voice_path:
        voice_path = Path(response.voice_path)
        try:
            await message.answer_voice(voice=FSInputFile(voice_path))
        finally:
            voice_path.unlink(missing_ok=True)
        return

    await message.answer(response.text)
