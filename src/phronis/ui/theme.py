"""Purple and white theme for phronis UI."""

import io
import sys

from rich.console import Console
from rich.theme import Theme

# ── Color constants ────────────────────────────────────────────────────
PURPLE = "rgb(168,85,247)"
DIM_PURPLE = "rgb(120,60,200)"
LIGHT_PURPLE = "rgb(200,160,255)"
ACCENT = "rgb(120,220,255)"

PHRONIS_THEME = Theme(
    {
        "purple": PURPLE,
        "dim.purple": DIM_PURPLE,
        "light.purple": LIGHT_PURPLE,
        "accent": ACCENT,
        "muted": "dim",
        "success": "green",
        "error": "bold red",
        "warn": "yellow",
        "info": "dim",
        # Panels / tables
        "panel.border": PURPLE,
        "table.header": f"bold {PURPLE}",
        "table.border": DIM_PURPLE,
        # Menu
        "menu.title": f"bold {PURPLE}",
        "menu.item": "white",
        "menu.selected": f"bold {PURPLE}",
        # Training
        "train.step": f"bold {PURPLE}",
        "train.loss": "white",
        "train.metric": DIM_PURPLE,
        # Chat
        "chat.user": f"bold {PURPLE}",
        "chat.assistant": f"bold {DIM_PURPLE}",
    }
)


# ── Questionary style ──────────────────────────────────────────────────

try:
    from questionary import Style as QStyle

    PHRONIS_QUESTIONARY_STYLE = QStyle([
        ("qmark", "fg:#a855f7 bold"),
        ("question", "bold"),
        ("answer", "fg:#c8a0ff bold"),
        ("pointer", "fg:#a855f7 bold"),
        ("selected", "fg:#ffffff"),
        ("instruction", "fg:#7830c8"),
        ("text", ""),
    ])
except ImportError:
    PHRONIS_QUESTIONARY_STYLE = None


def create_themed_console() -> Console:
    """Return a Console with the phronis purple theme applied."""
    return Console(
        file=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"),
        force_terminal=True,
        theme=PHRONIS_THEME,
    )
