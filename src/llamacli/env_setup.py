"""Isolated workspace environment management.

Ensures llamacli runs inside a compatible Python virtual environment
regardless of the system's default Python version.
"""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .workspace import get_workspace_path


def _venv_dir():
    return os.path.join(get_workspace_path(), ".venv")


def _venv_python():
    return os.path.join(_venv_dir(), "Scripts", "python.exe")


def _venv_pip():
    return os.path.join(_venv_dir(), "Scripts", "pip.exe")


def _venv_cli():
    return os.path.join(_venv_dir(), "Scripts", "llamafactory-cli.exe")


def _venv_llamacli():
    return os.path.join(_venv_dir(), "Scripts", "llamacli.exe")


def is_inside_isolated_venv():
    """Return True if the current interpreter lives inside the workspace venv."""
    return _venv_dir() in sys.executable


def _project_root_for_editable_install():
    """Return the repo root if this package was installed in editable mode."""
    try:
        import llamacli as _pkg  # Import from inside the package to get __file__ path

        inside_src = Path(_pkg.__file__).resolve().parent  # .../src/llamacli
        repo_root = inside_src.parent.parent  # Go up to repo root
        if (repo_root / "pyproject.toml").is_file():
            return str(repo_root)
    except Exception:
        pass
    return None


def _current_python_info():
    """Return (major, minor, micro) for the current interpreter."""
    return sys.version_info[:3]


def is_python_version_compatible(major: int, minor: int):
    """Return True if the given Python version supports CUDA torch wheels."""
    return major == 3 and 11 <= minor <= 13


def _is_torch_compatible(python_exe: str):
    """Return True if torch is installed and has CUDA support."""
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == "CUDA"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _find_compatible_python():
    """Find a Python 3.11–3.13 executable that supports CUDA wheels.

    1. Check current interpreter first.
    2. If current is missing CUDA, but version is OK, return it anyway
       (torch just needs reinstall).
    3. Otherwise try the Windows `py` launcher for 3.12/3.11.
    """
    current = sys.executable
    cur_major, cur_minor, _ = _current_python_info()

    if is_python_version_compatible(cur_major, cur_minor):
        return current

    # Current version is too new.  Search via py launcher.
    for minor in (13, 12, 11):
        try:
            result = subprocess.run(
                ["py", f"-3.{minor}", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None


def ensure_isolated_venv(console: Console):
    """Create the workspace venv and install dependencies.

    Returns True on success, False otherwise.  On success the venv
    is ready to run `llamacli` and `llamafactory-cli train`.
    """
    venv_dir = _venv_dir()
    venv_py = _venv_python()
    venv_pip = _venv_pip()

    # Already exists and is complete
    if os.path.isfile(venv_py) and os.path.isfile(venv_pip):
        return True

    py_exe = _find_compatible_python()
    if not py_exe:
        console.print(
            "[red]No compatible Python interpreter found.[/]"
        )
        console.print(
            "[dim]llamacli needs Python 3.11–3.13 with CUDA support.[/]"
        )
        console.print(
            "[dim]Install Python 3.12 from https://python.org and re-run.[/]"
        )
        return False

    try:
        with console.status(f"[bold green]Creating isolated environment ({py_exe})...", spinner="dots"):
            subprocess.run([py_exe, "-m", "venv", venv_dir], check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Failed to create venv: {exc}[/]")
        return False

    # Upgrade pip
    try:
        with console.status("[bold green]Upgrading pip...", spinner="dots"):
            subprocess.run(
                [venv_pip, "install", "--upgrade", "pip"],
                capture_output=True, timeout=120,
            )
    except subprocess.TimeoutExpired:
        pass  # non-fatal

    # Install torch with CUDA
    try:
        with console.status("[bold green]Installing CUDA PyTorch... (this may take several minutes)", spinner="dots"):
            subprocess.run(
                [
                    venv_pip, "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ],
                check=True, timeout=900,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print(
            "[red]Failed to install CUDA PyTorch. Check your internet.[/]"
        )
        return False

    # Install the llamacli package itself
    repo_root = _project_root_for_editable_install()
    try:
        with console.status("[bold green]Installing llamacli into isolated environment...", spinner="dots"):
            if repo_root:
                subprocess.run(
                    [venv_pip, "install", "-e", repo_root],
                    check=True, timeout=300,
                )
            else:
                subprocess.run(
                    [venv_pip, "install", "llamacli"],
                    check=True, timeout=300,
                )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print(
            "[red]Failed to install llamacli into isolated environment.[/]"
        )
        return False

    _create_wrapper_script(console, venv_dir)
    return True


def _create_wrapper_script(console: Console, venv_dir: str):
    """Write a tiny launcher so users can add it to their PATH."""
    wrapper = os.path.join(get_workspace_path(), "llamacli.cmd")
    venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    cmd_body = (
        "@echo off\n"
        f'"{venv_py}" -m llamacli %*\n'
    )
    try:
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(cmd_body)
    except OSError:
        pass
    else:
        console.print(
            f"[dim]Launcher written to:[/] [bold]{wrapper}[/]"
        )
        console.print(
            "[dim]Add that folder to your PATH for a quicker `llamacli` command.[/]"
        )


def forward_to_venv(argv=None):
    """Re-execute the current command using the isolated venv interpreter.

    Returns only on failure; on success the process is replaced by the
    subprocess call.
    """
    if argv is None:
        argv = sys.argv
    venv_py = _venv_python()
    if not os.path.isfile(venv_py):
        return False
    return subprocess.run([venv_py, "-m", "llamacli"] + argv[1:])
