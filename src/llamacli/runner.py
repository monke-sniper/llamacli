import ast
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel


def _find_cli():
    cli = shutil.which("llamafactory-cli")
    if cli:
        return cli
    bin_dir = os.path.join(os.path.dirname(sys.executable))
    for scripts_dir in ("Scripts", "bin"):
        candidate = os.path.join(bin_dir, scripts_dir, "llamafactory-cli")
        if os.name == "nt":
            candidate += ".exe"
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        "llamafactory-cli not found. Install LLaMA-Factory: pip install llamafactory"
    )


VENV_CLI = _find_cli()

METRICS_RE = re.compile(r"\{'loss': '[^']*'")
TOTAL_STEPS_RE = re.compile(r"Total optimization steps\s*=\s*(\d+)")


def _parse_metrics(raw_line):
    if not METRICS_RE.search(raw_line):
        return None
    for match in re.finditer(r"\{[^}]+\}", raw_line):
        try:
            d = ast.literal_eval(match.group())
            return {
                k: v
                for k, v in d.items()
                if k in ("loss", "grad_norm", "learning_rate", "epoch")
            }
        except (ValueError, SyntaxError):
            continue
    return None


def _format_metrics(metrics, step_total, elapsed):
    parts = []
    current, maximum = (
        step_total if isinstance(step_total, tuple) else (step_total, 0)
    )
    if current > 0:
        if maximum > 0:
            parts.append(f"Step [{current}/{maximum}]")
        else:
            parts.append(f"Step {current}")
    if metrics:
        epoch = metrics.get("epoch")
        if epoch:
            parts.append(f"Epoch {epoch}")
        loss = metrics.get("loss")
        if loss:
            parts.append(f"Loss {loss}")
        gn = metrics.get("grad_norm")
        if gn:
            parts.append(f"GNorm {gn}")
        lr = metrics.get("learning_rate")
        if lr:
            parts.append(f"LR {lr}")
    parts.append(f"Time {elapsed:.0f}s")
    return "  |  ".join(parts)


def run_training(console: Console, config_path: str, output_name: str):
    lines = []
    lock = threading.Lock()
    done = threading.Event()
    returncode = [None]
    step_counter = [0]
    total_steps = [0]

    def _stream():
        try:
            proc = subprocess.Popen(
                [VENV_CLI, "train", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    with lock:
                        lines.append(stripped)
                        m = TOTAL_STEPS_RE.search(stripped)
                        if m:
                            total_steps[0] = int(m.group(1))
                        if METRICS_RE.search(stripped):
                            step_counter[0] += 1
            proc.wait()
            returncode[0] = proc.returncode
        except Exception as e:
            with lock:
                lines.append(f"[ERROR] {e}")
            returncode[0] = 1
        finally:
            done.set()

    thread = threading.Thread(target=_stream, daemon=True)
    thread.start()

    start_time = time.time()
    latest_metrics = {}

    with Live(
        Panel("", title=f"[bold]Training - {output_name}[/]", border_style="white"),
        console=console,
        refresh_per_second=8,
        transient=False,
    ) as live:
        while not done.is_set():
            elapsed = time.time() - start_time
            with lock:
                all_lines = list(lines)
                visible = all_lines[-12:] if all_lines else ["[dim]Starting...[/]"]

            for line in reversed(all_lines):
                parsed = _parse_metrics(line)
                if parsed:
                    latest_metrics = parsed
                    break

            metrics_bar = _format_metrics(
                latest_metrics, (step_counter[0], total_steps[0]), elapsed
            )
            log_body = "\n".join(visible)
            content = metrics_bar + "\n" + "─" * 50 + "\n" + log_body
            live.update(
                Panel(
                    content,
                    title=f"[bold]Training - {output_name}[/]",
                    border_style="white",
                )
            )

    return returncode[0] == 0


def run_export(console: Console, config_path: str):
    console.print("\n[bold]Exporting model...[/]\n")
    try:
        result = subprocess.run(
            [VENV_CLI, "export", config_path],
            text=True,
            capture_output=True,
        )
        output = (result.stdout or "") + (result.stderr or "")
        for line in output.split("\n"):
            if line.strip():
                console.print(f"  [dim]{line.strip()}[/]")
        if result.returncode == 0:
            console.print("\n[green]Export complete![/]")
            return True
        console.print("\n[red]Export failed.[/]")
        return False
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return False
