"""Warm gold theme for phronis UI."""

import io
import sys

from rich.console import Console
from rich.theme import Theme

# ── Color constants ────────────────────────────────────────────────────
GOLD = "rgb(255,200,80)"
DIM_GOLD = "rgb(180,140,40)"
ACCENT = "rgb(120,220,255)"

PHRONIS_THEME = Theme(
    {
        "gold": GOLD,
        "dim.gold": DIM_GOLD,
        "accent": ACCENT,
        "muted": "dim",
        "success": "green",
        "error": "bold red",
        "warn": "yellow",
        "info": "dim",
        # Panels / tables
        "panel.border": GOLD,
        "table.header": f"bold {GOLD}",
        "table.border": DIM_GOLD,
        # Menu
        "menu.title": f"bold {GOLD}",
        "menu.item": "white",
        "menu.selected": f"bold {GOLD}",
        # Training
        "train.step": f"bold {GOLD}",
        "train.loss": "white",
        "train.metric": DIM_GOLD,
        # Chat
        "chat.user": f"bold {GOLD}",
        "chat.assistant": f"bold {DIM_GOLD}",
    }
)


def create_themed_console() -> Console:
    """Return a Console with the phronis warm gold theme applied."""
    return Console(
        file=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"),
        force_terminal=True,
        theme=PHRONIS_THEME,
    )
