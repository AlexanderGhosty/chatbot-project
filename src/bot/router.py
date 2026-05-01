from __future__ import annotations

from typing import Protocol

from aiogram import Dispatcher

from src.bot.handlers import HandlerDeps, bind_handlers
from src.services.dialogue_mgr import DialogueManager


class ServicesProtocol(Protocol):
    dialogue_manager: DialogueManager
    temp_audio_dir: str


def register_handlers(dispatcher: Dispatcher, services: ServicesProtocol) -> None:
    deps = HandlerDeps(dialogue_manager=services.dialogue_manager, temp_audio_dir=services.temp_audio_dir)
    dispatcher.include_router(bind_handlers(deps))
