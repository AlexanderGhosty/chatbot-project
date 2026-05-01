from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile, Message

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
        await _send_response(message=message, response=response)

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
        await _send_response(message=message, response=response)

    @router.errors()
    async def on_handler_error(event) -> None:
        # TODO: replace with centralized error taxonomy + mapping.
        logger.exception("Unhandled update processing error", extra={"event": str(event)})

    return router


async def _send_response(message: Message, response: BotResponse) -> None:
    """Deliver assembled response payload to Telegram."""
    if response.image_paths:
        await message.answer(response.text)
        for image_path in response.image_paths:
            path = Path(image_path)
            if path.exists():
                await message.answer_photo(photo=FSInputFile(path))
        return

    if response.send_voice and response.voice_path:
        await message.answer_voice(voice=FSInputFile(response.voice_path))
        return

    await message.answer(response.text)
