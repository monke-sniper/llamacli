import ast
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import yaml
from rich.console import Console
from rich.live import Live
from rich.panel import Panel


def _find_cli():
    # 1. Prefer the isolated workspace venv
    try:
        from .env_setup import _venv_cli
        venv_cli = _venv_cli()
        if os.path.isfile(venv_cli):
            return venv_cli
    except Exception:
        pass

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


_cli_path = None


def _get_cli():
    global _cli_path
    if _cli_path is None:
        _cli_path = _find_cli()
    return _cli_path


METRICS_RE = re.compile(r"'loss'\s*:")
TOTAL_STEPS_RE = re.compile(r"Total optimization steps\s*=\s*(\d+)")


def _parse_metrics(raw_line):
    if not METRICS_RE.search(raw_line):
        return None
    for match in re.finditer(r"\{[^}]+\}", raw_line):
        try:
            d = ast.literal_eval(match.group())
            parsed = {}
            for k in ("loss", "grad_norm", "learning_rate", "epoch"):
                v = d.get(k)
                if v is not None:
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        pass  # skip non-numeric values
            return parsed
        except (ValueError, SyntaxError):
            continue
    return None


def _format_metrics(metrics, step_total, elapsed, target_loss=None, best_loss=None, patience_counter=0, patience_limit=0):
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
        if loss is not None:
            try:
                parts.append(f"Loss {float(loss):.4f}")
            except (ValueError, TypeError):
                parts.append(f"Loss {loss}")
        gn = metrics.get("grad_norm")
        if gn:
            try:
                parts.append(f"GNorm {float(gn):.4f}")
            except (ValueError, TypeError):
                parts.append(f"GNorm {gn}")
        lr = metrics.get("learning_rate")
        if lr:
            parts.append(f"LR {lr}")
    if target_loss is not None:
        parts.append(f"Target {target_loss}")
        if best_loss is not None:
            parts.append(f"Best {best_loss:.4f}")
        if patience_limit > 0:
            parts.append(f"Patience {patience_counter}/{patience_limit}")
    parts.append(f"Time {elapsed:.0f}s")
    return "  |  ".join(parts)


def _loss_within_target(loss, target):
    """Return True if loss is at or below the target within a small margin."""
    if loss is None or target is None:
        return False
    margin = max(0.02, abs(target) * 0.05)
    return loss <= (target + margin)


def _find_checkpoints(output_dir):
    """Return a dict mapping step int -> checkpoint dir path."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = {}
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = re.search(r"checkpoint-(\d+)", name)
        if m:
            checkpoints[int(m.group(1))] = path
    return checkpoints


def _restore_checkpoint(console, output_dir, best_step, target_loss):
    """Copy the checkpoint closest to best_step over the final output_dir contents."""
    checkpoints = _find_checkpoints(output_dir)
    if not checkpoints:
        console.print("[yellow]No checkpoints found to restore.[/]")
        return False

    # Pick the highest checkpoint step <= best_step, or closest overall
    candidates = [s for s in checkpoints if s <= best_step]
    if candidates:
        chosen_step = max(candidates)
    else:
        chosen_step = min(checkpoints.keys())

    src = checkpoints[chosen_step]
    console.print(
        f"[dim]Restoring checkpoint {chosen_step} (best observed ~step {best_step}) => {output_dir}[/]"
    )

    # Remove existing adapter files in output_dir (keep checkpoint-* dirs)
    for entry in os.listdir(output_dir):
        full = os.path.join(output_dir, entry)
        if os.path.isfile(full):
            os.remove(full)
        elif os.path.isdir(full) and not entry.startswith("checkpoint-"):
            shutil.rmtree(full)

    # Copy checkpoint contents into output_dir
    for entry in os.listdir(src):
        s = os.path.join(src, entry)
        d = os.path.join(output_dir, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    console.print(f"[green]Restored checkpoint {chosen_step} as final adapter.[/]")
    return True


def run_training(console: Console, config_path: str, output_name: str, target_loss: float = None) -> bool:
    # Warn if training will be CPU-only
    try:
        import torch
        if not torch.cuda.is_available():
            console.print("[yellow]WARNING: PyTorch is running on CPU. Training will be very slow.[/]")
            console.print("[dim]Run [bold]phronis repair[/bold] to rebuild the environment with CUDA support.[/]")
            console.print()
    except Exception:
        pass

    # Load config to find output_dir
    cfg = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        console.print(f"[yellow]Warning: could not read config {config_path}: {exc}[/]")
    output_dir = cfg.get("output_dir", os.path.join("saves", output_name, "lora"))

    lines = []
    lock = threading.Lock()
    done = threading.Event()
    returncode = [None]
    step_counter = [0]
    total_steps = [0]

    proc_ref = [None]

    # Target-loss tracking
    best_loss = [float("inf")]
    best_step = [0]
    patience_counter = [0]
    patience_limit = 30  # logging steps after best loss to wait before stopping
    target_hit = [False]
    early_stopped = [False]

    def _stream():
        try:
            proc = subprocess.Popen(
                [_get_cli(), "train", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            )
            proc_ref[0] = proc
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

            # Target loss evaluation
            if target_loss is not None:
                loss = latest_metrics.get("loss")
                if loss is not None:
                    if _loss_within_target(loss, target_loss):
                        if loss < best_loss[0]:
                            best_loss[0] = loss
                            best_step[0] = step_counter[0]
                        patience_counter[0] = 0
                        target_hit[0] = True
                    else:
                        if target_hit[0]:
                            patience_counter[0] += 1
                    # Stop if patience exceeded
                    if target_hit[0] and patience_counter[0] > patience_limit:
                        console.print(
                            f"\n[yellow]Target loss {target_loss:.4f} reached (best {best_loss[0]:.4f}). "
                            f"Patience exceeded ({patience_counter[0]} > {patience_limit}). Stopping...[/]"
                        )
                        early_stopped[0] = True
                        if proc_ref[0] is not None:
                            try:
                                proc_ref[0].terminate()
                                proc_ref[0].wait(timeout=5)
                            except Exception:
                                pass
                        break

            metrics_bar = _format_metrics(
                latest_metrics,
                (step_counter[0], total_steps[0]),
                elapsed,
                target_loss=target_loss,
                best_loss=best_loss[0] if best_loss[0] != float("inf") else None,
                patience_counter=patience_counter[0],
                patience_limit=patience_limit if target_hit[0] else 0,
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
            time.sleep(0.12)

    # Wait for subprocess to finish if it didn't already
    if not done.is_set() and proc_ref[0] is not None:
        try:
            proc_ref[0].wait(timeout=60)
        except Exception:
            pass
        done.set()
    thread.join(timeout=5)

    success = returncode[0] == 0 or (target_hit[0] and early_stopped[0])

    # If we hit the target but stopped early, restore the best checkpoint
    if target_loss is not None and target_hit[0] and success:
        _restore_checkpoint(console, output_dir, best_step[0], target_loss)
    elif target_loss is not None and not target_hit[0]:
        console.print(f"[yellow]Target loss {target_loss:.4f} was not reached during training.[/]")

    return success


def run_export(console: Console, config_path: str) -> bool:
    try:
        with console.status("[bold green]Exporting / merging model...", spinner="dots"):
            result = subprocess.run(
                [_get_cli(), "export", config_path],
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
