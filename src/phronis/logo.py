"""Phronis boot sequence — particle logo + CRT system check."""

import sys

from rich.console import Console

from phronis.ui.particle_logo import run_particle_logo
from phronis.ui.crt_boot import run_boot_sequence


def get_logo_text() -> str:
    """Return plain ASCII logo text (backward compat for tests)."""
    try:
        from pyfiglet import Figlet
        fig = Figlet(font="slant")
        return fig.renderText("phronis")
    except Exception:
        return "phronis"


def _gather_system_info() -> list[tuple[str, str]]:
    """Gather system info for the CRT boot display."""
    purple = "rgb(168,85,247)"
    dim = "rgb(120,60,200)"

    lines = [
        ("Initializing phronis...", purple),
    ]

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lines.append((f"  Python: {py_ver}", dim))

    # GPU / CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            lines.append((f"  GPU: {gpu_name}", dim))
            lines.append((f"  CUDA: {torch.version.cuda}", dim))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            lines.append(("  GPU: Apple MPS", dim))
            lines.append(("  CUDA: n/a (MPS)", dim))
        else:
            is_cpu = torch.version.cuda is None or "cpu" in torch.__version__
            label = f"CPU only (torch {torch.__version__})" if is_cpu else "CPU only"
            lines.append((f"  GPU: {label}", dim))
            lines.append(("  CUDA: not available", dim))
    except ImportError:
        lines.append(("  GPU: torch not installed", dim))

    # LLaMA-Factory version
    try:
        import llamafactory
        lf_ver = getattr(llamafactory, "__version__", "unknown")
        lines.append((f"  LLaMA-Factory: {lf_ver}", dim))
    except ImportError:
        lines.append(("  LLaMA-Factory: not installed", dim))

    # Workspace
    try:
        from phronis import PROJECT_ROOT
        lines.append((f"  Workspace: {PROJECT_ROOT}", dim))
    except Exception:
        pass

    lines.append(("", ""))
    lines.append(("Quick Train  Advanced Train  Chat  Export", purple))

    return lines


def print_logo(console: Console) -> None:
    """Run the full phronis boot sequence."""
    # Phase 1: Particle coalesce logo
    run_particle_logo(console, hold_seconds=2.0)

    # Phase 2: Clear and CRT boot
    console.file.write("\033[2J\033[H")
    console.file.flush()

    boot_lines = _gather_system_info()
    run_boot_sequence(console, boot_lines)
