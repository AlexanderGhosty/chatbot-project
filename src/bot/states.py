try:
    from aiogram.fsm.state import State, StatesGroup
except ImportError:  # Allows core services to be tested without aiogram installed.
    class State:  # type: ignore[no-redef]
        def __set_name__(self, owner, name: str) -> None:
            self.state = f"{owner.__name__}:{name}"

    class StatesGroup:  # type: ignore[no-redef]
        pass


class DialogueStates(StatesGroup):
    normal_chat = State()
    ad_warmup = State()
    ad_offering = State()
    ad_follow_up = State()
