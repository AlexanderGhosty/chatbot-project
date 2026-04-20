from aiogram.fsm.state import State, StatesGroup


class DialogueStates(StatesGroup):
    normal_chat = State()
    ad_warmup = State()
    ad_offering = State()
    ad_follow_up = State()
