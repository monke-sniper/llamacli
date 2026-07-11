import os
from dataclasses import dataclass, field

import yaml

from . import DEFAULT_CONFIG, STATE_PATH


@dataclass
class AppState:
    active_model: str = ""
    active_adapter: str = ""
    active_template: str = "qwen3"
    active_dataset: str = ""
    training_history: list = field(default_factory=list)
    theme: str = "dark"

    def save(self):
        data = {
            "active_model": self.active_model,
            "active_adapter": self.active_adapter,
            "active_template": self.active_template,
            "active_dataset": self.active_dataset,
            "training_history": self.training_history,
            "theme": self.theme,
        }
        os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        merged = {**DEFAULT_CONFIG, **data}
        return cls(**merged)


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState.load()
    return _state


def reload_state() -> AppState:
    global _state
    _state = AppState.load()
    return _state
