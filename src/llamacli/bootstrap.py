import os
import shutil
import subprocess
import sys

from rich.console import Console
from rich.table import Table


def check_python_version():
    ok = sys.version_info >= (3, 11)
    return ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_package(pkg_name, import_name=None):
    imp = import_name or pkg_name
    try:
        __import__(imp)
        return True, "installed"
    except ImportError:
        return False, "not installed"


def check_llamafactory_cli():
    cli = shutil.which("llamafactory-cli")
    if cli:
        return True, cli
    bin_dir = os.path.join(os.path.dirname(sys.executable))
    for scripts_dir in ("Scripts", "bin"):
        candidate = os.path.join(bin_dir, scripts_dir, "llamafactory-cli")
        if os.name == "nt":
            candidate += ".exe"
        if os.path.isfile(candidate):
            return True, candidate
    return False, "not found"


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "Apple MPS"
        is_cpu_wheel = torch.version.cuda is None or "cpu" in torch.__version__
        msg = "CPU only"
        if is_cpu_wheel:
            msg += f" (torch wheel: {torch.__version__})"
        return "cpu", msg
    except ImportError:
        return "unknown", "torch not installed"


def check_hf_cli():
    cli = shutil.which("huggingface-cli")
    return bool(cli), cli or "not found"


def run_bootstrap(console: Console, force=False):
    """Run bootstrap setup. Asks for confirmation unless force=True."""
    console.print("\n[bold white]llamacli Setup[/bold white]\n")

    checks = []

    py_ok, py_ver = check_python_version()
    checks.append(("Python >= 3.11", py_ok, py_ver))

    lf_ok, lf_path = check_llamafactory_cli()
    checks.append(("LLaMA-Factory CLI", lf_ok, lf_path))

    hf_ok, hf_path = check_hf_cli()
    checks.append(("HuggingFace CLI", hf_ok, hf_path))

    torch_ok, torch_msg = check_package("torch")
    checks.append(("PyTorch", torch_ok, torch_msg))

    hf_hub_ok, hf_hub_msg = check_package("huggingface_hub")
    checks.append(("huggingface-hub", hf_hub_ok, hf_hub_msg))

    gpu_type, gpu_name = check_gpu()
    checks.append(("GPU", gpu_type != "cpu" and gpu_type != "unknown", gpu_name))

    from .workspace import init_workspace
    try:
        workspace_path, dirs = init_workspace()
        ws_ok = True
        ws_msg = workspace_path
    except Exception as e:
        ws_ok = False
        ws_msg = str(e)
    checks.append(("Workspace", ws_ok, ws_msg))

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Check", style="white")
    table.add_column("Status", width=8)
    table.add_column("Details", style="dim")

    all_ok = True
    for name, ok, details in checks:
        if name in ("GPU",):
            status = "[green]OK[/]" if ok else "[yellow]CPU[/]"
            if not ok:
                all_ok = False
        else:
            status = "[green]OK[/]" if ok else "[red]FAIL[/]"
            if not ok:
                all_ok = False
        table.add_row(name, status, details)

    console.print(table)

    missing = [name for name, ok, _ in checks if not ok and name != "GPU"]
    if not missing:
        console.print("\n[green]All checks passed! You're ready to tune LLMs.[/]")
        if os.path.isfile(os.path.join(ws_msg, "GUIDE.md")):
            console.print(f"[dim]Read the guide: {os.path.join(ws_msg, 'GUIDE.md')}[/]")
        return True

    console.print(f"\n[yellow]Missing: {', '.join(missing)}[/]")

    if force:
        should_install = True
    else:
        try:
            import questionary
            should_install = questionary.confirm(
                "Install missing dependencies now?",
                default=True,
            ).ask()
        except (KeyboardInterrupt, EOFError, ImportError):
            should_install = console.input("[dim]Install missing dependencies now? (y/n): [/]").strip().lower() == "y"

    if not should_install:
        console.print("[dim]You can run setup later with: llamacli setup[/]")
        return False

    _install_missing(console, missing)
    return True


def _install_missing(console: Console, missing: list):
    install_map = {
        "LLaMA-Factory CLI": ["llamafactory"],
        "HuggingFace CLI": ["huggingface-hub"],
        "PyTorch": ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"],
        "huggingface-hub": ["huggingface-hub"],
    }

    for name in missing:
        args = install_map.get(name, [name.lower().replace(" ", "-")])
        try:
            with console.status(f"[bold green]Installing {args[0]}...", spinner="dots"):
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + args,
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                console.print(f"[green]Installed {args[0]}[/]")
            else:
                console.print(f"[red]Failed to install {args[0]}[/]")
                console.print(f"[dim]{result.stderr[-500:]}[/]")
        except Exception as e:
            console.print(f"[red]Error installing {args[0]}: {e}[/]")

    console.print("\n[green]Setup complete![/]")
    console.print("[dim]If anything failed, install manually: pip install llamafactory huggingface-hub torch --index-url https://download.pytorch.org/whl/cu124[/]")
