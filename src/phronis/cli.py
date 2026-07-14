from typing import Any

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime

import questionary
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import (
    CONFIGS_DIR,
    DATASET_INFO,
    DATA_DIR,
    HF_CACHE,
    MODELS_DIR,
    PROJECT_ROOT,
    REPO_ROOT,
    SAVES_DIR,
    STATE_PATH,
    YAML_DIR,
)
from .hf import download_model, download_dataset, download_model_interactive, download_dataset_interactive
from .bootstrap import run_bootstrap
from .logo import print_logo
from .repro import gather_repro_metadata, format_repro_header
from .prompts import (
    _count_dataset,
    _ensure_dataset_registered,
    _list_cached_models,
    _list_datasets,
    detect_template,
    prompt_chat_model,
    prompt_dataset,
    prompt_finetuning_type,
    prompt_model,
    prompt_stage,
    prompt_target_loss,
    prompt_training_params,
)
from .runner import run_export, run_training
from .state import get_state, reload_state

app = typer.Typer(name="phronis", help="LLaMA-Factory Interactive CLI", add_completion=False)

console = Console(
    file=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"),
    force_terminal=True,
)

_quiet = False
_verbose = False
_no_input = False

MAIN_MENU = [
    questionary.Choice(title="  Quick Train", value="quick_train"),
    questionary.Choice(title="  Advanced Training", value="advanced_train"),
    questionary.Choice(title="  Train from YAML", value="yaml_train"),
    questionary.Choice(title="  Chat Trained Model", value="chat_trained"),
    questionary.Choice(title="  Quick Chat", value="quick_chat"),
    questionary.Choice(title="  Download Model", value="download_model"),
    questionary.Choice(title="  Download Dataset", value="download_dataset"),
    questionary.Choice(title="  Export Adapter", value="export"),
    questionary.Choice(title="  View Models", value="view_models"),
    questionary.Choice(title="  View Datasets", value="view_datasets"),
    questionary.Choice(title="  Add Dataset", value="add_dataset"),
    questionary.Choice(title="  Workspace Info", value="workspace_info"),
    questionary.Choice(title="  System Check", value="system_check"),
    questionary.Choice(title="  Exit", value="exit"),
]


def show_main_menu() -> str | None:
    try:
        return questionary.select(
            "What would you like to do?",
            choices=MAIN_MENU,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k or arrows to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return "exit"


def _print_breadcrumb(console: Console, title: str, current: int, total: int) -> None:
    parts = []
    parts.append(f"[bold cyan]{title}[/bold cyan]")
    parts.append(f"[dim][{current}/{total}][/dim]")
    console.print(Panel("  ".join(parts), border_style="dim"), highlight=False)
    console.print()


SMART_DEFAULTS = {
    "stage": "sft",
    "finetuning_type": "lora",
    "cutoff_len": 512,
    "max_samples": 10000,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "logging_steps": 5,
    "save_steps": 100,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "report_to": "none",
    "trust_remote_code": True,
    "do_train": True,
}


def _compute_dtype_flags() -> dict[str, bool]:
    """Return dtype flags aligned with the local GPU / CPU setup.

    - CUDA + bf16 support  -> bf16=True
    - CUDA without bf16    -> bf16=False, fp16=True (safe fallback)
    - CPU / no torch       -> bf16=False
    """
    try:
        import torch  # local import: avoid slowing CLI startup
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return {"bf16": True}
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 5:
                return {"bf16": False, "fp16": True}
            return {"bf16": False}
        return {"bf16": False}
    except Exception:
        return {"bf16": False}


def _get_workers_and_pin_memory() -> dict[str, Any]:
    """Return preprocessing workers and pin_memory based on platform + GPU.

    Windows multiprocessing with datasets frequently deadlocks during
    preprocessing, so we force single-threaded preprocessing on Windows
    (especially on CPU).  On Linux/macOS with GPU we keep the default 8.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {"preprocessing_num_workers": 8, "dataloader_pin_memory": True}
    except Exception:
        pass
    if os.name == "nt":  # Windows
        return {"preprocessing_num_workers": 0, "dataloader_pin_memory": False}
    return {"preprocessing_num_workers": 8, "dataloader_pin_memory": True}


def _warn_if_cpu_only(console: Console) -> None:
    """Print a warning when PyTorch is CPU-only so the user knows to reinstall."""
    try:
        import torch
        if torch.cuda.is_available():
            return
    except Exception:
        return
    console.print("[yellow]WARNING: PyTorch is running on CPU. Training will be very slow.[/]")
    console.print("[dim]If you have an NVIDIA GPU, reinstall CUDA PyTorch:[/]")
    console.print("[dim]  pip uninstall torch torchvision torchaudio -y[/]")
    console.print("[dim]  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124[/]")
    console.print()


def _build_config(model: str, template: str, dataset: str, epochs: float, finetuning_type: str, params: dict[str, Any], output_name: str, target_loss: float | None = None) -> dict[str, Any]:
    # Auto-register dataset if it's just a loose JSON file in DATA_DIR
    for ds_name in dataset.split(","):
        ds_name = ds_name.strip()
        if ds_name:
            _ensure_dataset_registered(ds_name)

    config = {
        "model_name_or_path": model,
        "template": template,
        "dataset": dataset,
        "dataset_dir": DATA_DIR,
        "output_dir": os.path.join("saves", output_name, "lora"),
    }
    config.update(SMART_DEFAULTS)
    config.update(_compute_dtype_flags())
    config.update(_get_workers_and_pin_memory())
    config["num_train_epochs"] = epochs
    config["finetuning_type"] = finetuning_type

    if finetuning_type == "lora":
        config["lora_rank"] = params.get("lora_rank", 8)
        config["lora_dropout"] = params.get("lora_dropout", 0.05)
        if params.get("lora_alpha", 0) > 0:
            config["lora_alpha"] = params["lora_alpha"]
        config["lora_target"] = "all"
    if params:
        config["learning_rate"] = params.get("learning_rate", 1e-4)
        config["num_train_epochs"] = params.get("epochs", epochs)
        config["per_device_train_batch_size"] = params.get("batch_size", 2)
        config["gradient_accumulation_steps"] = params.get("grad_accum", 8)
        config["cutoff_len"] = params.get("cutoff_len", 512)
        config["warmup_ratio"] = params.get("warmup_ratio", 0.1)

    if target_loss is not None:
        config["save_steps"] = 1
        config["logging_steps"] = 1

    return config


def _record_training(output_name: str, model: str, dataset: str, stage: str, epochs: float, template: str) -> None:
    state = get_state()
    state.active_model = model
    state.active_dataset = dataset.split(",")[0].strip() if dataset else ""
    state.active_template = template
    state.active_adapter = os.path.join("saves", output_name, "lora")
    record = {
        "name": output_name,
        "output_name": output_name,
        "model": model,
        "adapter": os.path.join("saves", output_name, "lora"),
        "template": template,
        "dataset": dataset,
        "stage": stage,
        "epochs": epochs,
        "config": os.path.join("configs", f"phronis_{output_name}.yaml"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    state.training_history.append(record)
    state.save()

    summary_path = os.path.join(CONFIGS_DIR, f"phronis_{output_name}_summary.yaml")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(record, f, default_flow_style=False, allow_unicode=True)
    except OSError:
        pass


def _write_config_and_train(console: Console, config: dict[str, Any], output_name: str, command: str = "train", target_loss: float | None = None, **cli_args) -> bool:
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f"phronis_{output_name}.yaml")

    # Ensure the output_dir inside the config matches the filename-based run name
    config["output_dir"] = os.path.join("saves", output_name, "lora")

    console.print("[dim]Gathering reproducibility metadata...[/]")
    metadata = gather_repro_metadata(command, output_name=output_name, **cli_args)
    config.setdefault("seed", metadata["random_seed"])

    header = format_repro_header(metadata)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    console.print(f"\n[dim]Config saved: {config_path}[/]")
    return run_training(console, config_path, output_name, target_loss=target_loss)


def quick_train(console: Console) -> None:
    steps = ["model", "dataset", "epochs", "target_loss", "confirm"]
    step_idx = 0
    data = {}

    while 0 <= step_idx < len(steps):
        step = steps[step_idx]
        _print_breadcrumb(console, "Quick Train", step_idx + 1, len(steps))

        if step == "model":
            model, template = prompt_model(console, allow_back=False)
            if not model:
                console.print("[dim]Cancelled.[/]")
                return
            data["model"] = model
            data["template"] = template
            step_idx += 1

        elif step == "dataset":
            dataset = prompt_dataset(console, allow_back=True)
            if dataset == "__back__":
                step_idx -= 1
                continue
            if not dataset:
                console.print("[dim]Cancelled.[/]")
                return
            data["dataset"] = dataset
            step_idx += 1

        elif step == "epochs":
            val = console.input("[dim]Number of epochs (default 3, or type 'back' to go back): [/]").strip()
            if val.lower() in ("back", "b"):
                step_idx -= 1
                continue
            try:
                epochs = float(val) if val else 3.0
            except ValueError:
                console.print("[yellow]Invalid, using 3.0[/]")
                epochs = 3.0
            data["epochs"] = epochs
            step_idx += 1

        elif step == "target_loss":
            target_loss = prompt_target_loss(console, allow_back=True)
            if target_loss == "__back__":
                step_idx -= 1
                continue
            data["target_loss"] = target_loss
            step_idx += 1

        elif step == "confirm":
            config = _build_config(
                data["model"], data["template"], data["dataset"],
                data["epochs"], "lora", {}, "",
                target_loss=data.get("target_loss"),
            )
            console.print("\n[dim]Using smart defaults: LoRA rank=8, LR=1e-4, batch=2, cutoff=512[/]")
            if data.get("target_loss") is not None:
                console.print("[dim]Target-loss mode: save_steps=1, logging_steps=1 for precise checkpointing.[/]")
            table = Table(title="Quick Train Configuration", show_header=False, border_style="white")
            table.add_column("Key", style="bold white", width=22)
            table.add_column("Value", style="white")
            for k in ("model_name_or_path", "template", "dataset", "num_train_epochs",
                      "finetuning_type", "lora_rank", "learning_rate", "cutoff_len"):
                table.add_row(k, str(config.get(k, "")))
            if data.get("target_loss") is not None:
                table.add_row("target_loss", str(data["target_loss"]))
            console.print(table)

            try:
                confirmed = questionary.confirm("Start training?", default=True).ask()
            except (KeyboardInterrupt, EOFError):
                confirmed = False
            if not confirmed:
                console.print("[dim]Cancelled.[/]")
                return

            output_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            success = _write_config_and_train(
                console, config, output_name,
                command="quick_train",
                target_loss=data.get("target_loss"),
                model=data["model"], template=data["template"], dataset=data["dataset"], epochs=data["epochs"],
            )
            if success:
                console.print(f"\n[green]Training complete! Output: saves/{output_name}/lora[/]")
                _record_training(output_name, data["model"], data["dataset"], "sft", data["epochs"], data["template"])
            else:
                console.print("\n[red]Training failed.[/]")
            return


def advanced_train(console: Console) -> None:
    steps = ["model", "dataset", "stage", "finetuning_type", "params", "target_loss", "output_name", "confirm"]
    step_idx = 0
    data = {}

    while 0 <= step_idx < len(steps):
        step = steps[step_idx]
        _print_breadcrumb(console, "Advanced Training", step_idx + 1, len(steps))

        if step == "model":
            model, template = prompt_model(console, allow_back=False)
            if not model:
                console.print("[dim]Cancelled.[/]")
                return
            data["model"] = model
            data["template"] = template
            step_idx += 1

        elif step == "dataset":
            dataset = prompt_dataset(console, allow_back=True)
            if dataset == "__back__":
                step_idx -= 1
                continue
            if not dataset:
                console.print("[dim]Cancelled.[/]")
                return
            data["dataset"] = dataset
            step_idx += 1

        elif step == "stage":
            stage = prompt_stage(console, allow_back=True)
            if stage == "__back__":
                step_idx -= 1
                continue
            if not stage:
                console.print("[dim]Cancelled.[/]")
                return
            data["stage"] = stage
            step_idx += 1

        elif step == "finetuning_type":
            finetuning_type = prompt_finetuning_type(console, allow_back=True)
            if finetuning_type == "__back__":
                step_idx -= 1
                continue
            if not finetuning_type:
                console.print("[dim]Cancelled.[/]")
                return
            data["finetuning_type"] = finetuning_type
            step_idx += 1

        elif step == "params":
            params = prompt_training_params(console, data["finetuning_type"], allow_back=True)
            if params == "__back__":
                step_idx -= 1
                continue
            if not params:
                console.print("[dim]Cancelled.[/]")
                return
            data["params"] = params
            step_idx += 1

        elif step == "target_loss":
            target_loss = prompt_target_loss(console, allow_back=True)
            if target_loss == "__back__":
                step_idx -= 1
                continue
            data["target_loss"] = target_loss
            step_idx += 1

        elif step == "output_name":
            val = console.input(
                f"[dim]Output name (default: run_{datetime.now().strftime('%Y%m%d_%H%M%S')}, or type 'back' to go back): [/]"
            ).strip()
            if val.lower() in ("back", "b"):
                step_idx -= 1
                continue
            if not val:
                val = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            data["output_name"] = val
            step_idx += 1

        elif step == "confirm":
            config = _build_config(
                data["model"], data["template"], data["dataset"],
                data["params"].get("epochs", 3), data["finetuning_type"], data["params"], data["output_name"],
                target_loss=data.get("target_loss"),
            )
            config["stage"] = data["stage"]
            if data.get("target_loss") is not None:
                console.print("[dim]Target-loss mode: save_steps=1, logging_steps=1 for precise checkpointing.[/]")

            table = Table(title="Confirm Training Configuration", show_header=False, border_style="white")
            table.add_column("Key", style="bold white", width=22)
            table.add_column("Value", style="white")
            for k, v in config.items():
                table.add_row(k, str(v))
            console.print(table)

            try:
                confirmed = questionary.confirm("Start training?", default=True).ask()
            except (KeyboardInterrupt, EOFError):
                confirmed = False
            if not confirmed:
                console.print("[dim]Cancelled.[/]")
                return

            success = _write_config_and_train(
                console, config, data["output_name"],
                command="advanced_train",
                target_loss=data.get("target_loss"),
                model=data["model"], template=data["template"], dataset=data["dataset"], stage=data["stage"],
                finetuning_type=data["finetuning_type"], epochs=data["params"].get("epochs", 3),
            )
            if success:
                console.print(f"\n[green]Training complete! Output: saves/{data['output_name']}/lora[/]")
                _record_training(data["output_name"], data["model"], data["dataset"], data["stage"], data["params"].get("epochs", 3), data["template"])
            else:
                console.print("\n[red]Training failed.[/]")
            return


def chat_trained(console: Console) -> None:
    state = get_state()
    console.print("\n[bold white]Chat with Trained Model[/bold white]\n")

    if not state.training_history:
        console.print("[yellow]No trained models yet. Run Quick Train first.[/]")
        return

    last = state.training_history[-1]
    model = last["model"]
    adapter = last.get("adapter", "")
    template = last.get("template", "qwen3")

    if adapter and not os.path.isdir(adapter):
        console.print(f"[yellow]Adapter not found: {adapter}[/]")
        console.print("[dim]Training output may have been moved or deleted.[/]")
        return

    console.print(f"[bold]Run:[/] {last['name']} ({last['timestamp']})")
    console.print(f"[bold]Model:[/] [dim]{model}[/]")
    if adapter:
        console.print(f"[bold]Adapter:[/] [dim]{adapter}[/]")

    _start_chat(console, model, adapter or None, template)


def quick_chat(console: Console) -> None:
    console.print("\n[bold white]Quick Chat[/bold white]\n")
    model, adapter, template = prompt_chat_model(console, allow_back=False)
    if not model:
        console.print("[dim]Cancelled.[/]")
        return
    if model == "__back__":
        console.print("[dim]Cancelled.[/]")
        return
    _start_chat(console, model, adapter, template)


def _start_chat(console: Console, model: str, adapter: str | None, template: str) -> None:
    label = f"{model} + {adapter}" if adapter else model
    console.print(f"\n[bold white]Chat with {label}[/bold white]")
    console.print("[dim]Type messages. /quit to exit, /clear to reset.[/dim]\n")

    try:
        from llamafactory.chat.chat_model import ChatModel
    except ImportError:
        console.print("[red]LLaMA-Factory is not installed.[/]")
        console.print("[dim]Install it with: pip install llamafactory[/]")
        return

    config = {
        "model_name_or_path": model,
        "template": template or "qwen3",
        "finetuning_type": "lora",
    }
    if adapter:
        config["adapter_name_or_path"] = adapter

    try:
        with console.status("[bold green]Loading model into memory...", spinner="dots"):
            chat_model = ChatModel(config)
    except FileNotFoundError:
        console.print(f"[red]Model not found locally: {model}[/]")
        console.print("[dim]Use 'Download Model' from the menu to download it first.[/]")
        return
    except MemoryError:
        console.print("[red]Not enough system memory. Try a smaller model or use a GPU.[/]")
        return
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "oom" in msg.lower():
            console.print("[red]GPU out of memory. Try a smaller model or lower batch size.[/]")
        elif "unexpected" in msg.lower() or "not supported" in msg.lower():
            console.print(f"[red]This model type is not supported for chat: {model}[/]")
            console.print("[dim]Make sure the model is a text-generation model (not embedding, audio, or vision-only).[/]")
        else:
            console.print(f"[red]Failed to load model: {msg}[/]")
        return
    except ValueError as e:
        console.print(f"[red]Invalid model configuration: {e}[/]")
        console.print("[dim]Check that the model path and template are correct.[/]")
        return
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/]")
        return

    console.print("[green]Model loaded![/]\n")
    messages = []

    while True:
        text = console.input("[bold cyan]You:[/] ").strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/clear":
            messages = []
            console.print("[dim]Chat cleared.[/]\n")
            continue

        messages.append({"role": "user", "content": text})
        try:
            response = ""
            console.print("[bold green]Assistant:[/] ", end="")
            for token in chat_model.stream_chat(messages):
                response += token
                console.print(token, end="")
            console.print()
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"\n[red]Error during generation: {e}[/]")


def yaml_train_screen(console: Console) -> None:
    console.print("\n[bold white]Train from YAML[/bold white]\n")
    if not os.path.isdir(YAML_DIR):
        os.makedirs(YAML_DIR, exist_ok=True)
        console.print(f"[dim]Created {YAML_DIR}[/]")

    yaml_files = [f for f in sorted(os.listdir(YAML_DIR)) if f.endswith((".yaml", ".yml"))]
    if not yaml_files:
        console.print("[yellow]No YAML configs found.[/]")
        console.print(f"[dim]Drop saved .yaml configs into {YAML_DIR} to train from them.[/]")
        return

    steps = ["select_yaml", "output_name", "confirm"]
    step_idx = 0
    data = {}

    while 0 <= step_idx < len(steps):
        step = steps[step_idx]
        _print_breadcrumb(console, "Train from YAML", step_idx + 1, len(steps))

        if step == "select_yaml":
            choices = []
            for f in yaml_files:
                choices.append(questionary.Choice(title=f"  {f}", value=os.path.join(YAML_DIR, f)))
            choices.append(questionary.Choice(title="  Custom path...", value="__custom__"))
            choices.append(questionary.Choice(title="  ← Cancel", value="__cancel__"))

            try:
                selected = questionary.select(
                    "Select a YAML config:",
                    choices=choices,
                    pointer=">",
                    use_arrow_keys=True,
                    use_jk_keys=True,
                ).ask()
            except (KeyboardInterrupt, EOFError):
                console.print("[dim]Cancelled.[/]")
                return

            if selected == "__cancel__" or not selected:
                console.print("[dim]Cancelled.[/]")
                return
            if selected == "__custom__":
                path = console.input("[dim]Path to YAML config (or type 'back' to cancel): [/]").strip()
                if path.lower() in ("back", "b") or not path:
                    console.print("[dim]Cancelled.[/]")
                    return
                selected = path
            data["selected"] = selected
            step_idx += 1

        elif step == "output_name":
            val = console.input(
                f"[dim]Output name (default: run_{datetime.now().strftime('%Y%m%d_%H%M%S')}, or type 'back' to go back): [/]"
            ).strip()
            if val.lower() in ("back", "b"):
                step_idx -= 1
                continue
            if not val:
                val = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            data["output_name"] = val
            step_idx += 1

        elif step == "confirm":
            # For YAML training we run the YAML directly without rebuilding a config
            console.print("\n[dim]Training directly from saved YAML.[/]")
            success = run_training(console, data["selected"], data["output_name"])
            if success:
                console.print(f"\n[green]Training complete! Output: saves/{data['output_name']}/lora[/]")
            else:
                console.print("\n[red]Training failed.[/]")
            return


def view_models_screen(console: Console) -> None:
    models = _list_cached_models()
    state = get_state()
    console.print(f"\n[bold white]Cached Models ({len(models)})[/bold white]\n")
    if not models:
        console.print("[dim]No cached models found. Use 'Download Model' from the menu.[/]")
        console.print("[dim]Models are cached in ~/.cache/huggingface/hub[/]")
        return

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="white")
    table.add_column("Size", style="dim", width=12)
    for i, m in enumerate(models, 1):
        mark = " *" if m["repo_id"] == state.active_model else ""
        table.add_row(str(i), m["repo_id"] + mark, f"{m['size_gb']:.1f} GB")
    console.print(table)

    try:
        choice = console.input("\n[dim]Set active model # (or Enter to skip): [/]").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                state.active_model = models[idx]["repo_id"]
                state.save()
                console.print(f"[green]Active model: {state.active_model}[/]")
    except (KeyboardInterrupt, EOFError):
        pass


def view_datasets_screen(console: Console) -> None:
    from . import DATA_DIR

    datasets = _list_datasets()
    state = get_state()
    console.print(f"\n[bold white]Available Datasets ({len(datasets)})[/bold white]\n")
    if not datasets:
        console.print("[dim]No datasets found.[/]")
        console.print(f"[dim]Drop .json or .jsonl files in {DATA_DIR}[/]")
        console.print("[dim]Or use 'Add Dataset' from the menu to register one.[/]")
        return

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("#", style="dim", width=4)
    table.add_column("Dataset", style="white")
    table.add_column("Examples", style="dim", width=10)
    table.add_column("Format", style="dim", width=10)
    table.add_column("Source", style="dim", width=12)
    for i, d in enumerate(datasets, 1):
        mark = " *" if d["name"] == state.active_dataset else ""
        cnt = _count_dataset(d["name"])
        table.add_row(str(i), d["name"] + mark, str(cnt), d["format"], d.get("source", "-"))
    console.print(table)

    try:
        choice = console.input("\n[dim]Set active dataset # (or Enter to skip): [/]").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                state.active_dataset = datasets[idx]["name"]
                state.save()
                console.print(f"[green]Active dataset: {state.active_dataset}[/]")
    except (KeyboardInterrupt, EOFError):
        pass


def add_dataset_screen(console: Console) -> None:
    from . import DATA_DIR, DATASET_INFO
    steps = ["name", "source", "details", "confirm"]
    step_idx = 0
    data = {}

    while 0 <= step_idx < len(steps):
        step = steps[step_idx]
        _print_breadcrumb(console, "Add Dataset", step_idx + 1, len(steps))

        if step == "name":
            name = console.input("[dim]Dataset name (used in configs, or type 'back' to cancel): [/]").strip()
            if name.lower() in ("back", "b"):
                console.print("[dim]Cancelled.[/]")
                return
            if not name:
                console.print("[dim]Cancelled.[/]")
                return
            data["name"] = name
            step_idx += 1

        elif step == "source":
            src = questionary.select(
                "Where is the data?",
                choices=[
                    questionary.Choice("Local file in data/ (e.g. my_data.json)", value="file"),
                    questionary.Choice("HuggingFace dataset (hf_hub_url)", value="hf"),
                    questionary.Choice("← Back", value="__back__"),
                ],
                pointer=">",
                use_arrow_keys=True,
                use_jk_keys=True,
            ).ask()
            if src == "__back__":
                step_idx -= 1
                continue
            if not src:
                console.print("[dim]Cancelled.[/]")
                return
            data["source"] = src
            step_idx += 1

        elif step == "details":
            if data["source"] == "file":
                file_name = console.input(f"[dim]Filename in {DATA_DIR}/ (e.g. my_data.json, or type 'back'): [/]").strip()
                if file_name.lower() in ("back", "b"):
                    step_idx -= 1
                    continue
                if not file_name:
                    console.print("[dim]Cancelled.[/]")
                    return
                data["file_name"] = file_name
            else:
                url = console.input("[dim]HuggingFace dataset URL (or type 'back'): [/]").strip()
                if url.lower() in ("back", "b"):
                    step_idx -= 1
                    continue
                if not url:
                    console.print("[dim]Cancelled.[/]")
                    return
                data["url"] = url

            fmt = questionary.select(
                "Format:",
                choices=[
                    questionary.Choice("alpaca (instruction/input/output)", value="alpaca"),
                    questionary.Choice("sharegpt (messages)", value="sharegpt"),
                    questionary.Choice("← Back", value="__back__"),
                ],
                pointer=">",
                use_arrow_keys=True,
                use_jk_keys=True,
            ).ask()
            if fmt == "__back__":
                step_idx -= 1
                continue
            if not fmt:
                console.print("[dim]Cancelled.[/]")
                return
            data["format"] = fmt
            step_idx += 1

        elif step == "confirm":
            if data["source"] == "file":
                entry = {"file_name": data["file_name"], "formatting": data["format"]}
            else:
                entry = {"hf_hub_url": data["url"], "formatting": data["format"]}

            os.makedirs(DATA_DIR, exist_ok=True)
            registry = {}
            if os.path.isfile(DATASET_INFO):
                with open(DATASET_INFO, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            registry[data["name"]] = entry
            with open(DATASET_INFO, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Dataset '{data['name']}' added to dataset_info.json[/]")
            return


def export_screen(console: Console) -> None:
    state = get_state()
    steps = ["adapter", "dest", "confirm"]
    step_idx = 0
    data = {}

    while 0 <= step_idx < len(steps):
        step = steps[step_idx]
        _print_breadcrumb(console, "Export / Merge Adapter", step_idx + 1, len(steps))

        if step == "adapter":
            adapter = console.input(
                f"[dim]Adapter path (default: {state.active_adapter or 'none'}, or type 'back' to cancel): [/]"
            ).strip()
            if adapter.lower() in ("back", "b"):
                console.print("[dim]Cancelled.[/]")
                return
            if not adapter:
                adapter = state.active_adapter
            if not adapter:
                console.print("[yellow]No adapter specified.[/]")
                return
            data["adapter"] = adapter
            step_idx += 1

        elif step == "dest":
            dest_default = os.path.join(
                "models",
                os.path.basename(data["adapter"].rstrip("/\\").replace("/lora", "").replace("\\lora", "")),
            )
            dest = console.input(f"[dim]Export destination (default: {dest_default}, or type 'back' to go back): [/]").strip()
            if dest.lower() in ("back", "b"):
                step_idx -= 1
                continue
            if not dest:
                dest = dest_default
            data["dest"] = dest
            step_idx += 1

        elif step == "confirm":
            config = {
                "model_name_or_path": state.active_model or "Qwen/Qwen3-0.6B",
                "adapter_name_or_path": data["adapter"],
                "template": state.active_template or "qwen3",
                "finetuning_type": "lora",
                "export_dir": data["dest"],
                "export_size": 2,
                "export_legacy_format": False,
            }

            os.makedirs(CONFIGS_DIR, exist_ok=True)
            config_path = os.path.join(CONFIGS_DIR, "phronis_export_temp.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            try:
                confirmed = questionary.confirm("Proceed with export?", default=True).ask()
            except (KeyboardInterrupt, EOFError):
                console.print("[dim]Cancelled.[/]")
                return
            if not confirmed:
                console.print("[dim]Cancelled.[/]")
                return

            run_export(console, config_path)
            return


def workspace_info_screen(console: Console) -> None:
    from . import PROJECT_ROOT, DATA_DIR, SAVES_DIR, MODELS_DIR, CONFIGS_DIR, YAML_DIR
    console.print("\n[bold white]Workspace Info[/bold white]\n")
    console.print(f"[bold]Project root:[/] [dim]{PROJECT_ROOT}[/]")
    console.print(f"[bold]Data dir:[/]    [dim]{DATA_DIR}[/]")
    console.print(f"[bold]YAML dir:[/]    [dim]{YAML_DIR}[/]")
    console.print(f"[bold]Saves dir:[/]   [dim]{SAVES_DIR}[/]")
    console.print(f"[bold]Models dir:[/] [dim]{MODELS_DIR}[/]")
    console.print(f"[bold]Configs dir:[/] [dim]{CONFIGS_DIR}[/]")

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Directory", style="white")
    table.add_column("Items", style="dim", width=8)
    table.add_column("Size", style="dim", width=12)

    for label, path in [
        ("data/", DATA_DIR),
        ("yaml/", YAML_DIR),
        ("saves/", SAVES_DIR),
        ("models/", MODELS_DIR),
        ("configs/", CONFIGS_DIR),
    ]:
        count = 0
        size = 0
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    count += 1
                    try:
                        size += os.path.getsize(fp)
                    except OSError:
                        pass
        size_str = f"{size / (1024**3):.2f} GB" if size >= 1024**3 else f"{size / (1024**2):.1f} MB" if size >= 1024**2 else f"{size} B"
        table.add_row(label, str(count), size_str)
    console.print(table)


def system_check_screen(console: Console) -> None:
    import shutil
    import sys as _sys

    console.print("\n[bold white]System Check[/bold white]\n")

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Check", style="white")
    table.add_column("Status", style="dim", width=12)
    table.add_column("Details", style="dim")

    py_ok = _sys.version_info >= (3, 11)
    table.add_row(
        "Python >= 3.11",
        "[green]OK[/]" if py_ok else "[red]FAIL[/]",
        f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
    )

    lf_cli = shutil.which("llamafactory-cli")
    table.add_row(
        "LLaMA-Factory",
        "[green]OK[/]" if lf_cli else "[yellow]MISSING[/]",
        lf_cli or "pip install llamafactory",
    )

    hf_cli = shutil.which("huggingface-cli")
    table.add_row(
        "HuggingFace CLI",
        "[green]OK[/]" if hf_cli else "[yellow]MISSING[/]",
        hf_cli or "pip install huggingface-hub",
    )

    try:
        import torch
        gpu_ok = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        gpu_name = ""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_name = "Apple MPS"
        if not gpu_ok:
            is_cpu_wheel = torch.version.cuda is None or "cpu" in torch.__version__
            if is_cpu_wheel:
                gpu_name = f"CPU-only torch wheel ({torch.__version__}) — reinstall with CUDA"
            else:
                gpu_name = gpu_name or "No GPU detected"
        table.add_row(
            "GPU (CUDA/MPS)",
            "[green]OK[/]" if gpu_ok else "[yellow]CPU ONLY[/]",
            gpu_name,
        )
    except ImportError:
        table.add_row("GPU (CUDA/MPS)", "[yellow]UNKNOWN[/]", "torch not installed")

    from . import DATA_DIR, SAVES_DIR, MODELS_DIR, CONFIGS_DIR, YAML_DIR
    for name, d in [("data/", DATA_DIR), ("yaml/", YAML_DIR), ("saves/", SAVES_DIR), ("models/", MODELS_DIR), ("configs/", CONFIGS_DIR)]:
        exists = os.path.isdir(d)
        if not exists:
            os.makedirs(d, exist_ok=True)
            exists = True
        table.add_row(name, "[green]OK[/]" if exists else "[red]FAIL[/]", d)

    console.print(table)


def _ensure_directories() -> None:
    from . import CONFIGS_DIR, DATA_DIR, MODELS_DIR, SAVES_DIR, DATASET_INFO, YAML_DIR

    for d in (DATA_DIR, SAVES_DIR, MODELS_DIR, CONFIGS_DIR, YAML_DIR):
        os.makedirs(d, exist_ok=True)

    readme_path = os.path.join(DATA_DIR, "README.txt")
    if not os.path.isfile(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("Place your training datasets here.\n")
            f.write("\n")
            f.write("Supported formats:\n")
            f.write("  - .json  (JSON array of examples)\n")
            f.write("  - .jsonl (one JSON object per line)\n")
            f.write("\n")
            f.write("Alpaca format (auto-detected):\n")
            f.write('[{"instruction": "...", "input": "...", "output": "..."}]\n')
            f.write("\n")
            f.write("ShareGPT format (auto-detected):\n")
            f.write('[{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]\n')
            f.write("\n")
            f.write("Files dropped here appear in the dataset dropdown automatically.\n")

    if not os.path.isfile(DATASET_INFO):
        with open(DATASET_INFO, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def interactive_loop() -> None:
    _ensure_directories()
    reload_state()
    while True:
        try:
            console.clear()
            print_logo(console)
            console.print()
            choice = show_main_menu()

            if choice == "exit" or choice is None:
                console.print("\n[white]Goodbye.[/]")
                return
            if choice == "quick_train":
                quick_train(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "advanced_train":
                advanced_train(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "yaml_train":
                yaml_train_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "chat_trained":
                chat_trained(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "quick_chat":
                quick_chat(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "download_model":
                download_model_interactive(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "download_dataset":
                download_dataset_interactive(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "export":
                export_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "view_models":
                view_models_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "view_datasets":
                view_datasets_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "add_dataset":
                add_dataset_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "workspace_info":
                workspace_info_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")
            elif choice == "system_check":
                system_check_screen(console)
                console.input("\n[dim]Press Enter to return to menu...[/]")

        except KeyboardInterrupt:
            console.print("\n\n[white]Goodbye.[/]")
            return
        except EOFError:
            return
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
            try:
                console.input("\n[dim]Press Enter to continue...[/]")
            except (KeyboardInterrupt, EOFError):
                return


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    workspace: str = typer.Option(None, "--workspace", help="Override workspace directory"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output"),
    verbose: bool = typer.Option(False, "--verbose", "--debug", help="Verbose output"),
    no_input: bool = typer.Option(False, "--no-input", help="Disable interactive prompts (non-TTY / CI mode)"),
) -> None:
    global console, _quiet, _verbose, _no_input

    _quiet = quiet
    _verbose = verbose
    _no_input = no_input

    if workspace:
        os.environ["PHRONIS_WORKSPACE"] = workspace
        # Re-init module constants for this process without persisting to global config
        import importlib
        import phronis as _pkg
        importlib.reload(_pkg)

    if no_color:
        # Disable colors on the existing console without recreation
        # to avoid file descriptor issues on Windows with PIPE stdout.
        try:
            console._color_system = None
        except Exception:
            pass

    if version:
        from . import PROJECT_ROOT
        console.print(f"[white]phronis[/] [bold]{PROJECT_ROOT}[/]")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        if no_input:
            console.print("[red]--no-input passed but no subcommand given.[/]")
            console.print("[dim]Use subcommands with explicit flags instead of interactive mode:[/]")
            console.print("  phronis train --config <path>")
            console.print("  phronis chat --model <model_id>")
            console.print("  phronis export --adapter <path> --dest <dir>")
            raise typer.Exit(code=1)
        _check_first_run(console)
        interactive_loop()


def _check_first_run(console: Console) -> None:
    from . import PROJECT_ROOT
    marker = os.path.join(PROJECT_ROOT, ".phronis.yaml")
    if not os.path.isfile(marker):
        console.print("\n[bold cyan]Welcome to phronis![/bold cyan]")
        console.print("[dim]Let's check your system before we start.[/]\n")
        run_bootstrap(console)
        console.print()
        # Persist a default state so the welcome screen only runs once
        from .state import get_state
        get_state().save()


@app.command()
def setup() -> None:
    """Run system check and install missing dependencies."""
    run_bootstrap(console)


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    stage: str = typer.Option("sft", "--stage", "-s", help="Training stage"),
    epochs: float = typer.Option(3.0, "--epochs", "-e", help="Number of epochs"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    batch: int = typer.Option(2, "--batch", "-b", help="Batch size"),
    cutoff: int = typer.Option(512, "--cutoff", help="Cutoff length"),
    lora_rank: int = typer.Option(8, "--lora-rank", help="LoRA rank"),
    output: str = typer.Option("", "--output", "-o", help="Output name"),
    template: str = typer.Option("qwen3", "--template", "-t", help="Template"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print config YAML and exit without training"),
    resume: str = typer.Option(None, "--resume", help="Resume training from checkpoint path"),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Push merged model to HuggingFace Hub after training"),
    force: bool = typer.Option(False, "--force", help="Overwrite output directory without asking"),
    grad_accum: int = typer.Option(8, "--grad-accum", help="Gradient accumulation steps"),
    warmup: float = typer.Option(0.1, "--warmup", help="Warmup ratio"),
    scheduler: str = typer.Option("cosine", "--scheduler", help="LR scheduler type (cosine/linear/constant)"),
    method: str = typer.Option("lora", "--method", help="Finetuning method: lora/full/freeze"),
    target_loss: float = typer.Option(None, "--target-loss", help="Stop training when loss reaches approximately this value"),
) -> None:
    # Auto-register dataset if it's just a loose JSON file in DATA_DIR
    for ds_name in dataset.split(","):
        ds_name = ds_name.strip()
        if ds_name:
            _ensure_dataset_registered(ds_name)

    output_name = output or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {
        "model_name_or_path": model,
        "trust_remote_code": True,
        "stage": stage,
        "do_train": True,
        "finetuning_type": method,
        "lora_rank": lora_rank,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "dataset": dataset,
        "dataset_dir": DATA_DIR,
        "template": template,
        "cutoff_len": cutoff,
        "max_samples": 10000,
        "output_dir": f"saves/{output_name}/lora",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "lr_scheduler_type": scheduler,
        "warmup_ratio": warmup,
        **_compute_dtype_flags(),
        **_get_workers_and_pin_memory(),
    }
    if target_loss is not None:
        config["save_steps"] = 1
        config["logging_steps"] = 1
    if resume:
        config["resume_from_checkpoint"] = resume
    if force:
        config["overwrite_output_dir"] = True

    # Record method-specific info
    extra_args = {
        "model": model, "dataset": dataset, "stage": stage, "epochs": epochs,
        "lr": lr, "batch": batch, "cutoff": cutoff, "lora_rank": lora_rank,
        "method": method, "grad_accum": grad_accum, "scheduler": scheduler,
        "warmup": warmup,
    }

    _warn_if_cpu_only(console)

    if dry_run:
        import yaml
        console.print("[bold]Dry run — config that would be used:[/bold]")
        console.print("─" * 50)
        console.print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        console.print("─" * 50)
        raise typer.Exit(0)

    success = _write_config_and_train(
        console, config, output_name,
        command="train",
        target_loss=target_loss,
        **extra_args,
    )
    if success:
        console.print("\n[green]Training complete![/]")
        _record_training(output_name, model, dataset, stage, epochs, template)
        if push_to_hub:
            _push_to_hub(console, config["output_dir"])
    else:
        raise typer.Exit(1)


def _push_to_hub(console: Console, adapter_path: str):
    from .hf import _check_hf
    if not _check_hf(console):
        console.print("[yellow]Skipping push to Hub — huggingface_hub not available[/]")
        return
    try:
        from huggingface_hub import HfApi
        HfApi()
        repo_id = console.input("[dim]HuggingFace repo ID to push to (org/name): [/]").strip()
        if not repo_id:
            console.print("[yellow]No repo ID provided. Skipping push.[/]")
            return
        console.print(f"[dim]Pushing {adapter_path} to {repo_id}...[/]")
        from huggingface_hub import upload_folder
        upload_folder(repo_id=repo_id, folder_path=adapter_path)
        console.print(f"[green]Pushed to https://huggingface.co/{repo_id}[/]")
    except Exception as e:
        console.print(f"[red]Push to Hub failed: {e}[/]")


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Model path (defaults to active_model in state)"),
    adapter: str = typer.Option(None, "--adapter", "-a", help="Adapter path"),
    template: str = typer.Option(None, "--template", "-t", help="Chat template"),
    message: str = typer.Option(None, "--message", help="Single message: non-interactive mode"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max generation tokens"),
) -> None:
    """Chat with a model. Interactive by default, or single-shot with --message."""
    state = get_state()
    model_path = model or state.active_model or "Qwen/Qwen3-0.6B"
    adapter_path = adapter or state.active_adapter
    template_name = template or state.active_template or detect_template(model_path)

    try:
        from llamafactory.chat import ChatModel
    except ImportError:
        console.print("[red]llamafactory not installed. Run: phronis setup[/]")
        raise typer.Exit(1)

    config = {
        "model_name_or_path": model_path,
        "template": template_name,
        "trust_remote_code": True,
    }
    if adapter_path:
        config["adapter_name_or_path"] = adapter_path
        config["finetuning_type"] = "lora"
    config["max_tokens"] = max_tokens

    console.print(f"[dim]Loading model: {model_path}[/dim]")
    try:
        chat_model = ChatModel(config)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/]")
        raise typer.Exit(1)

    console.print(f"[green]Model loaded![/] Template: {template_name}")
    if adapter_path:
        console.print(f"[green]Adapter:[/] {adapter_path}")

    if message:
        # Single-shot mode
        messages = [{"role": "user", "content": message}]
        console.print(f"\n[bold cyan]You:[/] {message}\n")
        console.print("[bold green]Assistant:[/] ", end="")
        try:
            response = ""
            for token in chat_model.stream_chat(messages):
                response += token
                console.print(token, end="")
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
            raise typer.Exit(1)
        return

    # Interactive mode
    _no_input_guard("Use --message for single-shot non-interactive chat.")
    messages = []
    console.print("[dim]Type /quit or /exit to quit. /clear to start a new chat.[/]\n")
    while True:
        try:
            text = console.input("[bold cyan]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            console.print("[dim]Goodbye![/]")
            break
        if text == "/clear":
            messages = []
            console.print("[dim]Chat cleared.[/]\n")
            continue

        messages.append({"role": "user", "content": text})
        try:
            response = ""
            console.print("[bold green]Assistant:[/] ", end="")
            for token in chat_model.stream_chat(messages):
                response += token
                console.print(token, end="")
            console.print()
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")


@app.command()
def export(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Adapter path to export"),
    model: str = typer.Option(None, "--model", "-m", help="Base model (defaults to active_model)"),
    output: str = typer.Option(None, "--output", "-o", help="Export destination directory"),
    template: str = typer.Option(None, "--template", "-t", help="Chat template"),
    size: int = typer.Option(2, "--size", help="Shard size in GB"),
) -> None:
    """Merge a LoRA adapter into a standalone model."""
    state = get_state()
    model_path = model or state.active_model or "Qwen/Qwen3-0.6B"
    template_name = template or state.active_template or detect_template(model_path)
    dest = output or os.path.join(
        "models",
        os.path.basename(adapter.rstrip("/\\").replace("/lora", "").replace("\\lora", "")),
    )

    config = {
        "model_name_or_path": model_path,
        "adapter_name_or_path": adapter,
        "template": template_name,
        "finetuning_type": "lora",
        "export_dir": dest,
        "export_size": size,
        "export_legacy_format": False,
    }

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, "phronis_export_temp.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    run_export(console, config_path)


@app.command("yaml-train")
def yaml_train(
    config: str = typer.Argument(..., help="Path to YAML config, or filename inside yaml/"),
    output: str = typer.Option("", "--output", "-o", help="Output name"),
) -> None:
    """Train a model from an existing YAML config file."""
    # Resolve path
    candidate = os.path.join(YAML_DIR, config)
    if os.path.isfile(candidate):
        config_path = candidate
    elif os.path.isfile(config):
        config_path = config
    else:
        console.print(f"[red]YAML config not found: {config}[/]")
        console.print(f"[dim]Searched: {os.path.abspath(config)} and {os.path.abspath(candidate)}[/]")
        raise typer.Exit(1)

    output_name = output or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    console.print(f"[dim]Training from YAML: {config_path}[/]")
    success = run_training(console, config_path, output_name)
    if success:
        console.print(f"\n[green]Training complete! Output: saves/{output_name}/lora[/]")
    else:
        console.print("\n[red]Training failed.[/]")
        raise typer.Exit(1)


def _no_input_guard(command_hint: str) -> None:
    if _no_input:
        console.print("[red]--no-input is active but this command requires interactivity.[/]")
        console.print(f"[dim]Hint: {command_hint}[/]")
        raise typer.Exit(1)


@app.command()
def download(
    kind: str = typer.Argument(..., help="Type to download: model or dataset"),
    name: str = typer.Argument(..., help="HuggingFace repo ID"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="Skip confirmation prompt"),
) -> None:
    """Download a model or dataset from HuggingFace."""
    kind_lower = kind.lower()
    if kind_lower not in ("model", "dataset"):
        console.print(f"[red]Invalid kind '{kind}'. Must be 'model' or 'dataset'.[/]")
        raise typer.Exit(1)

    if not no_confirm:
        _no_input_guard("Use --no-confirm to skip the confirmation prompt.")

    if kind_lower == "model":
        if no_confirm:
            # Bypass interactive confirmation by calling snapshot_download directly
            console.print(f"[dim]Downloading model {name}...[/]")
            try:
                from huggingface_hub import snapshot_download
                path = snapshot_download(name)
                console.print(f"[green]Downloaded to: {path}[/]")
            except Exception as e:
                console.print(f"[red]Download failed: {e}[/]")
                raise typer.Exit(1)
        else:
            path = download_model(console, name)
            if path:
                console.print("[dim]The model will appear in the model list next time.[/]")
    else:
        if no_confirm:
            console.print(f"[dim]Downloading dataset {name}...[/]")
            try:
                from huggingface_hub import snapshot_download
                safe_name = name.replace("/", "_")
                local_dir = os.path.join(DATA_DIR, safe_name)
                path = snapshot_download(name, repo_type="dataset", local_dir=local_dir)
                console.print(f"[green]Dataset downloaded to: {path}[/]")
            except Exception as e:
                console.print(f"[red]Download failed: {e}[/]")
                raise typer.Exit(1)
        else:
            safe_name = name.replace("/", "_")
            local_dir = os.path.join(DATA_DIR, safe_name)
            path = download_dataset(console, name, local_dir)
            if path:
                # Auto-register
                from .hf import _register_downloaded_dataset
                _register_downloaded_dataset(console, name, safe_name, path)


@app.command()
def list(
    what: str = typer.Argument("models", help="What to list: models, datasets, history, adapters"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List cached models, datasets, training history, or adapters."""
    what_lower = what.lower()

    if what_lower == "models":
        models = _list_cached_models()
        if json_output:
            print(json.dumps(models, indent=2, ensure_ascii=False))
            return
        if not models:
            console.print("[dim]No cached models found.[/]")
            return
        table = Table(show_header=True, header_style="bold white", border_style="white")
        table.add_column("#", style="dim", width=4)
        table.add_column("Model", style="white")
        table.add_column("Size", style="dim", width=12)
        table.add_column("Type", style="dim", width=12)
        for i, m in enumerate(models, 1):
            table.add_row(str(i), m["repo_id"], f"{m['size_gb']:.1f} GB", m.get("model_type", "?"))
        console.print(table)

    elif what_lower == "datasets":
        datasets = _list_datasets()
        if json_output:
            print(json.dumps(datasets, indent=2, ensure_ascii=False))
            return
        if not datasets:
            console.print("[dim]No datasets found.[/]")
            return
        table = Table(show_header=True, header_style="bold white", border_style="white")
        table.add_column("#", style="dim", width=4)
        table.add_column("Dataset", style="white")
        table.add_column("Examples", style="dim", width=10)
        table.add_column("Format", style="dim", width=10)
        table.add_column("Source", style="dim", width=12)
        for i, d in enumerate(datasets, 1):
            cnt = _count_dataset(d["name"])
            table.add_row(str(i), d["name"], str(cnt), d["format"], d.get("source", "-"))
        console.print(table)

    elif what_lower == "history":
        state = get_state()
        history = state.training_history
        if json_output:
            print(json.dumps(history, indent=2, ensure_ascii=False))
            return
        if not history:
            console.print("[dim]No training history.[/]")
            return
        table = Table(show_header=True, header_style="bold white", border_style="white")
        table.add_column("#", style="dim", width=4)
        table.add_column("Run", style="white")
        table.add_column("Model", style="dim")
        table.add_column("Dataset", style="dim")
        table.add_column("Stage", style="dim", width=8)
        table.add_column("Epochs", style="dim", width=8)
        for i, h in enumerate(history, 1):
            table.add_row(
                str(i),
                h.get("output_name", "?"),
                h.get("model", "?"),
                h.get("dataset", "?"),
                h.get("stage", "?"),
                str(h.get("epochs", "?")),
            )
        console.print(table)

    elif what_lower == "adapters":
        adapters = []
        if os.path.isdir(SAVES_DIR):
            for root, dirs, files in os.walk(SAVES_DIR):
                if "adapter_config.json" in files:
                    rel = os.path.relpath(root, SAVES_DIR)
                    adapters.append({"path": rel, "full_path": root})
        if json_output:
            print(json.dumps(adapters, indent=2, ensure_ascii=False))
            return
        if not adapters:
            console.print("[dim]No adapters found.[/]")
            return
        table = Table(show_header=True, header_style="bold white", border_style="white")
        table.add_column("#", style="dim", width=4)
        table.add_column("Adapter", style="white")
        table.add_column("Full Path", style="dim")
        for i, a in enumerate(adapters, 1):
            table.add_row(str(i), a["path"], a["full_path"])
        console.print(table)

    else:
        console.print(f"[red]Unknown list type: {what}. Use models/datasets/history/adapters.[/]")
        raise typer.Exit(1)


@app.command("add")
def add_dataset(
    name: str = typer.Option(..., "--name", help="Dataset name (used in configs)"),
    file: str = typer.Option(None, "--file", help="Local filename in data/"),
    hf_url: str = typer.Option(None, "--hf-url", help="HuggingFace dataset URL"),
    format: str = typer.Option("alpaca", "--format", help="Format: alpaca or sharegpt"),
) -> None:
    """Register a dataset (local file or HuggingFace URL).

    For sharegpt datasets, the actual JSON key ('messages' or 'conversations')
    is auto-detected so LLaMA-Factory reads the correct column.
    """
    if file and hf_url:
        console.print("[red]Cannot specify both --file and --hf-url.[/]")
        raise typer.Exit(1)
    if not file and not hf_url:
        console.print("[red]Must specify --file or --hf-url.[/]")
        raise typer.Exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    registry = {}
    if os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if file:
        entry = {"file_name": file, "formatting": format}
        # Auto-detect sharegpt key from file content
        if format == "sharegpt":
            fpath = os.path.join(DATA_DIR, file)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict):
                            if "messages" in first:
                                entry["columns"] = {"messages": "messages"}
                            elif "conversations" in first:
                                entry["columns"] = {"messages": "conversations"}
                except Exception:
                    pass
    else:
        entry = {"hf_hub_url": hf_url, "formatting": format}

    registry[name] = entry
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Dataset '{name}' added to dataset_info.json[/]")


@app.command()
def info(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show workspace info, active model/dataset, and directory sizes."""
    state = get_state()
    dirs_info = {}
    for label, path in [
        ("data", DATA_DIR),
        ("yaml", YAML_DIR),
        ("saves", SAVES_DIR),
        ("models", MODELS_DIR),
        ("configs", CONFIGS_DIR),
    ]:
        count = 0
        size = 0
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    count += 1
                    try:
                        size += os.path.getsize(fp)
                    except OSError:
                        pass
        dirs_info[label] = {"files": count, "size": size}

    from .env_setup import _venv_dir

    venv_dir = _venv_dir() if os.name == "nt" else None
    if os.name == "nt":
        from .env_setup import _venv_dir as _get_venv_dir
        venv_dir = _get_venv_dir()
    else:
        venv_dir = None

    info_data = {
        "repo_root": REPO_ROOT,
        "workspace": PROJECT_ROOT,
        "state_file": STATE_PATH,
        "dataset_registry": DATASET_INFO,
        "active_model": state.active_model,
        "active_adapter": state.active_adapter,
        "active_template": state.active_template,
        "active_dataset": state.active_dataset,
        "directories": dirs_info,
    }

    if json_output:
        # Use plain print for JSON to avoid Rich syntax-highlighting escape codes
        print(json.dumps(info_data, indent=2, ensure_ascii=False))
        return

    panel_content = (
        f"[bold]Repo Root:[/] {REPO_ROOT}\n"
        f"[bold]Workspace:[/] {PROJECT_ROOT}\n"
        f"[bold]State File:[/] {STATE_PATH}\n"
        f"[bold]Dataset Registry:[/] {DATASET_INFO}\n"
        f"[bold]Active Model:[/] {state.active_model or '(none)'}\n"
        f"[bold]Active Adapter:[/] {state.active_adapter or '(none)'}\n"
        f"[bold]Active Template:[/] {state.active_template}\n"
        f"[bold]Active Dataset:[/] {state.active_dataset or '(none)'}\n"
    )
    console.print(Panel(panel_content, title="[bold]phronis Info[/bold]", border_style="white"))

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Directory", style="white")
    table.add_column("Files", style="dim", width=8)
    table.add_column("Size", style="dim", width=12)
    for label, info in dirs_info.items():
        size = info["size"]
        size_str = f"{size / (1024**3):.2f} GB" if size >= 1024**3 else f"{size / (1024**2):.1f} MB" if size >= 1024**2 else f"{size} B"
        table.add_row(f"{label}/", str(info["files"]), size_str)
    console.print(table)


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Auto-install missing dependencies"),
) -> None:
    """Run a full system diagnostic (like brew doctor)."""
    from . import PROJECT_ROOT
    console.print("\n[bold white]phronis Doctor[/bold white]\n")

    # Use bootstrap for core checks
    ok = run_bootstrap(console, force=fix)

    # Extra checks
    extra_table = Table(show_header=True, header_style="bold white", border_style="white")
    extra_table.add_column("Extra Check", style="white")
    extra_table.add_column("Status", width=12)
    extra_table.add_column("Details", style="dim")

    # Disk space
    try:
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free / (1024**3)
        ds_ok = free_gb > 1
        extra_table.add_row(
            "Disk space",
            "[green]OK[/]" if ds_ok else "[yellow]LOW[/]",
            f"{free_gb:.1f} GB free",
        )
    except Exception as e:
        extra_table.add_row("Disk space", "[yellow]WARN[/]", str(e))

    # State file corruption check
    state_ok = True
    state_msg = "valid"
    from . import STATE_PATH
    if os.path.isfile(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
        except Exception as e:
            state_ok = False
            state_msg = f"corrupted: {e}"
    else:
        state_msg = "missing (will be created)"
    extra_table.add_row(
        "State file",
        "[green]OK[/]" if state_ok else "[red]FAIL[/]",
        state_msg,
    )

    # Dataset registry check
    ds_reg_ok = True
    ds_reg_msg = "valid"
    if os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            ds_reg_ok = False
            ds_reg_msg = f"corrupted: {e}"
    else:
        ds_reg_msg = "missing (will be created)"
    extra_table.add_row(
        "Dataset registry",
        "[green]OK[/]" if ds_reg_ok else "[red]FAIL[/]",
        ds_reg_msg,
    )

    console.print(extra_table)

    if ok and state_ok and ds_reg_ok:
        console.print("\n[green]All diagnostics passed![/]")
    else:
        console.print("\n[yellow]Some issues found. Run with --fix to auto-repair.[/]")


@app.command()
def update(
    check: bool = typer.Option(False, "--check", help="Only check for updates, don't install"),
    force_pip: bool = typer.Option(False, "--force-pip", help="Force PyPI upgrade even if installed from source"),
) -> None:
    """Self-update phronis. Pulls latest source if installed from git, otherwise uses pip."""
    from . import REPO_ROOT

    console.print("[bold]phronis Update[/bold]\n")

    is_source_install = os.path.isdir(os.path.join(REPO_ROOT, ".git"))

    pip_prefix = [sys.executable, "-m", "pip"]

    # Try to fetch latest version info from PyPI
    try:
        result = subprocess.run(
            pip_prefix + ["index", "versions", "phronis"],
            capture_output=True, text=True, timeout=30,
        )
        latest = None
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "LATEST" in line or "Available" in line:
                    latest = line.strip().split()[-1]
                    break
    except Exception:
        latest = None

    if check:
        console.print(f"[bold]Install path:[/] {REPO_ROOT}")
        install_type = "source (git)" if is_source_install else "PyPI / package"
        console.print(f"[bold]Install type:[/] {install_type}")
        if latest:
            console.print(f"[bold]Latest available:[/] {latest}")
        else:
            console.print("[dim]Could not check latest PyPI version.[/]")
        return

    if is_source_install and not force_pip:
        console.print("[dim]Source install detected. Pulling latest code...[/]")
        try:
            result = subprocess.run(
                ["git", "-C", REPO_ROOT, "pull"],
                capture_output=False, text=True, timeout=120,
            )
            if result.returncode != 0:
                console.print("[red]Git pull failed.[/]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Git pull error: {e}[/]")
            raise typer.Exit(1)

        console.print("[dim]Re-installing in editable mode...[/]")
        try:
            result = subprocess.run(
                pip_prefix + ["install", "-e", REPO_ROOT],
                capture_output=False, text=True, timeout=300,
            )
            if result.returncode == 0:
                console.print("\n[green]phronis updated successfully from source![/]")
            else:
                console.print("\n[red]Re-install failed.[/]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]Re-install error: {e}[/]")
            raise typer.Exit(1)
    else:
        if force_pip and is_source_install:
            console.print("[yellow]--force-pip set; skipping git pull and forcing PyPI upgrade.[/]\n")

        console.print("[dim]Installing latest version from PyPI...[/]")
        try:
            result = subprocess.run(
                pip_prefix + ["install", "--upgrade", "phronis"],
                capture_output=False, text=True, timeout=300,
            )
            if result.returncode == 0:
                console.print("\n[green]phronis updated successfully from PyPI![/]")
            else:
                console.print("\n[red]Update failed.[/]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]Update error: {e}[/]")
            raise typer.Exit(1)


@app.command()
def clean(
    what: str = typer.Argument("all", help="What to clean: configs, cache, saves, all"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Clean up workspace: remove old configs, cache files, or saves."""
    what_lower = what.lower()
    valid = ("configs", "cache", "saves", "all")
    if what_lower not in valid:
        console.print(f"[red]Invalid target '{what}'. Use: configs, cache, saves, all.[/]")
        raise typer.Exit(1)

    targets = []
    if what_lower in ("configs", "all"):
        targets.append(("configs", CONFIGS_DIR))
    if what_lower == "cache":
        # HF cache
        from . import HF_CACHE
        targets.append(("HF cache", HF_CACHE))
    if what_lower in ("saves", "all"):
        targets.append(("saves", SAVES_DIR))

    if not targets:
        console.print("[dim]Nothing to clean.[/]")
        return

    if not force:
        console.print("[yellow]Will delete:[/]")
        for label, path in targets:
            console.print(f"  {label}: {path}")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    for label, path in targets:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                console.print(f"[green]Deleted {label}: {path}[/]")
            except Exception as e:
                console.print(f"[red]Failed to delete {label}: {e}[/]")
        else:
            console.print(f"[dim]{label} not found: {path}[/]")


# ─────────────────────────────────────────────────────────────
# Config management
# ─────────────────────────────────────────────────────────────
config_app = typer.Typer()
app.add_typer(config_app, name="config")

@config_app.command("get")
def config_get(key: str) -> None:
    """Get a value from .phronis.yaml state."""
    valid_keys = ("active_model", "active_adapter", "active_template", "active_dataset", "theme")
    if key not in valid_keys:
        console.print(f"[red]Unknown key '{key}'. Valid: {', '.join(valid_keys)}[/]")
        raise typer.Exit(1)
    state = get_state()
    value = getattr(state, key, "")
    console.print(f"{key} = {value}")

@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a value in .phronis.yaml state."""
    valid_keys = ("active_model", "active_adapter", "active_template", "active_dataset", "theme")
    if key not in valid_keys:
        console.print(f"[red]Unknown key '{key}'. Valid: {', '.join(valid_keys)}[/]")
        raise typer.Exit(1)
    state = get_state()
    setattr(state, key, value)
    state.save()
    console.print(f"[green]Set {key} = {value}[/]")


# ─────────────────────────────────────────────────────────────
# Uninstall
# ─────────────────────────────────────────────────────────────
@app.command()
def uninstall(
    workspace: bool = typer.Option(False, "--workspace", help="Also remove the workspace directory"),
    venv: bool = typer.Option(True, "--venv/--no-venv", help="Remove the isolated venv"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Uninstall phronis and optionally its workspace."""
    console.print("[bold]phronis Uninstall[/bold]\n")

    from .env_setup import _venv_dir, is_inside_isolated_venv

    actions = []
    actions.append(("pip package", "phronis"))
    venv_dir = _venv_dir()
    if venv and os.path.isdir(venv_dir):
        actions.append(("isolated venv", venv_dir))
    if workspace:
        actions.append(("workspace", PROJECT_ROOT))

    if not force:
        console.print("[yellow]Will remove:[/]")
        for label, path in actions:
            console.print(f"  {label}: {path}")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    pip_prefix = [sys.executable, "-m", "pip"]
    try:
        subprocess.run(
            pip_prefix + ["uninstall", "phronis", "-y"],
            capture_output=False, text=True, timeout=120,
        )
    except Exception as e:
        console.print(f"[yellow]pip uninstall warning: {e}[/]")

    for label, path in actions[1:]:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                console.print(f"[green]Removed {label}: {path}[/]")
            except Exception as e:
                console.print(f"[red]Failed to remove {label}: {e}[/]")

    console.print("\n[green]phronis has been uninstalled.[/]")
    if not workspace:
        console.print(f"[dim]Workspace preserved at: {PROJECT_ROOT}[/]")


# ─────────────────────────────────────────────────────────────
# Repair
# ─────────────────────────────────────────────────────────────
@app.command()
def repair(
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Rebuild the isolated workspace environment."""
    console.print("[bold]phronis Repair[/bold]\n")

    from .env_setup import _venv_dir, ensure_isolated_venv, is_inside_isolated_venv

    venv_dir = _venv_dir()
    if not force and os.path.isdir(venv_dir):
        console.print(f"[yellow]Will delete and recreate:[/] {venv_dir}")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    if os.path.isdir(venv_dir):
        try:
            shutil.rmtree(venv_dir)
            console.print(f"[green]Removed old venv: {venv_dir}[/]")
        except Exception as e:
            console.print(f"[red]Failed to remove venv: {e}[/]")
            raise typer.Exit(1)

    if ensure_isolated_venv(console):
        console.print("\n[green]Environment repaired successfully.[/]")
    else:
        console.print("\n[red]Repair failed.[/]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────
# Reinstall
# ─────────────────────────────────────────────────────────────
@app.command()
def reinstall(
    force_pip: bool = typer.Option(False, "--force-pip", help="Force PyPI reinstall even for source installs"),
) -> None:
    """Reinstall phronis cleanly (git pull + repair for source, force-reinstall for PyPI)."""
    console.print("[bold]phronis Reinstall[/bold]\n")

    from .env_setup import _venv_dir, ensure_isolated_venv, is_inside_isolated_venv

    is_source = os.path.isdir(os.path.join(REPO_ROOT, ".git"))
    pip_prefix = [sys.executable, "-m", "pip"]

    if is_source and not force_pip:
        console.print("[dim]Source install detected. Pulling latest code...[/]")
        try:
            subprocess.run(
                ["git", "-C", REPO_ROOT, "pull"],
                capture_output=False, text=True, timeout=120,
            )
        except Exception as e:
            console.print(f"[red]Git pull failed: {e}[/]")
            raise typer.Exit(1)

        console.print("[dim]Installing in editable mode...[/]")
        try:
            subprocess.run(
                pip_prefix + ["install", "-e", REPO_ROOT],
                capture_output=False, text=True, timeout=300,
            )
        except Exception as e:
            console.print(f"[red]Editable install failed: {e}[/]")
            raise typer.Exit(1)
    else:
        if force_pip and is_source:
            console.print("[yellow]--force-pip set; forcing PyPI reinstall.[/]")
        console.print("[dim]Reinstalling from PyPI...[/]")
        try:
            subprocess.run(
                pip_prefix + ["install", "--upgrade", "--force-reinstall", "phronis"],
                capture_output=False, text=True, timeout=300,
            )
        except Exception as e:
            console.print(f"[red]PyPI reinstall failed: {e}[/]")
            raise typer.Exit(1)

    venv_dir = _venv_dir()
    if os.path.isdir(venv_dir):
        if is_source and not force_pip:
            console.print("\n[dim]Rebuilding isolated environment...[/]")
        if ensure_isolated_venv(console):
            console.print("\n[green]phronis reinstalled successfully![/]")
        else:
            console.print("\n[yellow]Package reinstalled, but isolated env repair failed.[/]")
            raise typer.Exit(1)
    else:
        console.print("\n[green]phronis reinstalled successfully![/]")


# ─────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────
@app.command()
def reset(
    history: bool = typer.Option(False, "--history", help="Clear training history"),
    state: bool = typer.Option(False, "--state", help="Reset all state to defaults"),
    all: bool = typer.Option(False, "--all", help="Full factory reset including files"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Reset phronis state, history, or workspace files."""
    console.print("[bold]phronis Reset[/bold]\n")

    targets = []
    if history:
        targets.append("training history")
    if state:
        targets.append("app state (.phronis.yaml)")
    if all:
        targets.extend(["configs", "saves", "models", "state", "history"])

    if not targets:
        console.print("[yellow]Nothing to reset. Use --history, --state, or --all.[/]")
        raise typer.Exit(1)

    if not force:
        console.print("[yellow]Will reset:[/]")
        for t in targets:
            console.print(f"  - {t}")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    state_obj = get_state()
    if history or all:
        state_obj.training_history = []
        console.print("[green]Cleared training history.[/]")

    if state or all:
        state_obj.active_model = ""
        state_obj.active_adapter = ""
        state_obj.active_template = "qwen3"
        state_obj.active_dataset = ""
        state_obj.theme = "dark"
        console.print("[green]Reset app state to defaults.[/]")
    state_obj.save()

    if all:
        for d, label in ((CONFIGS_DIR, "configs"), (SAVES_DIR, "saves"), (MODELS_DIR, "models")):
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                    console.print(f"[green]Deleted {label}: {d}[/]")
                except Exception as e:
                    console.print(f"[red]Failed to delete {label}: {e}[/]")
            else:
                console.print(f"[dim]{label} not found: {d}[/]")

    console.print("\n[green]Reset complete.[/]")


# ─────────────────────────────────────────────────────────────
# Backup / Restore
# ─────────────────────────────────────────────────────────────
@app.command()
def backup(
    path: str = typer.Argument(None, help="Output path for backup archive (.zip)"),
) -> None:
    """Backup workspace state to a zip file."""
    import zipfile as zf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_path = os.path.join(os.path.expanduser("~"), f"phronis_backup_{timestamp}.zip")
    out_path = os.path.abspath(path or default_path)

    console.print(f"[bold]phronis Backup[/bold]\n")
    console.print(f"[dim]Creating archive: {out_path}[/]")

    try:
        with zf.ZipFile(out_path, "w", zf.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(PROJECT_ROOT):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, PROJECT_ROOT)
                    z.write(full, arcname)
        console.print(f"\n[green]Backup saved: {out_path}[/]")
    except Exception as e:
        console.print(f"\n[red]Backup failed: {e}[/]")
        raise typer.Exit(1)


@app.command()
def restore(
    path: str = typer.Argument(..., help="Path to backup archive (.zip)"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Restore workspace state from a backup archive."""
    import zipfile as zf

    console.print(f"[bold]phronis Restore[/bold]\n")

    if not os.path.isfile(path):
        console.print(f"[red]Backup not found: {path}[/]")
        raise typer.Exit(1)

    if not force:
        console.print(f"[yellow]Will extract {path} into:[/] {PROJECT_ROOT}")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    try:
        with zf.ZipFile(path, "r") as z:
            z.extractall(PROJECT_ROOT)
        console.print(f"\n[green]Restored to: {PROJECT_ROOT}[/]")
    except Exception as e:
        console.print(f"\n[red]Restore failed: {e}[/]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────
# Delete sub-commands
# ─────────────────────────────────────────────────────────────
delete_app = typer.Typer()
app.add_typer(delete_app, name="delete")

@delete_app.command("dataset")
def delete_dataset(
    name: str = typer.Argument(..., help="Dataset name to remove from registry"),
    keep_files: bool = typer.Option(False, "--keep-files", help="Keep the data file, only unregister"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Unregister and optionally delete a dataset."""
    console.print(f"[bold]Delete Dataset: {name}[/bold]\n")

    if not os.path.isfile(DATASET_INFO):
        console.print("[red]No dataset registry found.[/]")
        raise typer.Exit(1)

    with open(DATASET_INFO, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if name not in registry:
        console.print(f"[red]Dataset '{name}' not found in registry.[/]")
        raise typer.Exit(1)

    entry = registry[name]
    file_name = entry.get("file_name", "")

    if not force:
        action = "Unregister only" if keep_files else "Unregister and delete"
        console.print(f"[yellow]{action} dataset '{name}'?[/]")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    del registry[name]
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    if not keep_files and file_name:
        data_file = os.path.join(DATA_DIR, file_name)
        if os.path.isfile(data_file):
            try:
                os.remove(data_file)
                console.print(f"[green]Deleted data file: {data_file}[/]")
            except Exception as e:
                console.print(f"[yellow]Could not delete data file: {e}[/]")

    console.print(f"[green]Dataset '{name}' removed from registry.[/]")


@delete_app.command("adapter")
def delete_adapter(
    name: str = typer.Argument(..., help="Adapter run name or path to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Delete a LoRA adapter (training run output)."""
    console.print(f"[bold]Delete Adapter: {name}[/bold]\n")

    adapter_dir = name if os.path.isdir(name) else os.path.join(SAVES_DIR, name)
    if not os.path.isdir(adapter_dir):
        console.print(f"[red]Adapter not found: {adapter_dir}[/]")
        raise typer.Exit(1)

    if not force:
        console.print(f"[yellow]Delete adapter directory?[/]\n  {adapter_dir}")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    try:
        shutil.rmtree(adapter_dir)
        console.print(f"[green]Deleted adapter: {adapter_dir}[/]")
    except Exception as e:
        console.print(f"[red]Failed to delete adapter: {e}[/]")
        raise typer.Exit(1)


@delete_app.command("run")
def delete_run(
    name: str = typer.Argument(..., help="Training run name to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Delete a training run (saves, config, and history entry)."""
    console.print(f"[bold]Delete Run: {name}[/bold]\n")

    run_dir = os.path.join(SAVES_DIR, name)
    run_config = os.path.join(CONFIGS_DIR, f"{name}.yaml")

    if not os.path.isdir(run_dir) and not os.path.isfile(run_config):
        console.print(f"[red]Run not found: {name}[/]")
        raise typer.Exit(1)

    if not force:
        console.print(f"[yellow]Delete run '{name}' including saves and config?[/]")
        try:
            ans = input("Proceed? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/]")
            return
        if ans != "y":
            console.print("[dim]Cancelled.[/]")
            return

    if os.path.isdir(run_dir):
        try:
            shutil.rmtree(run_dir)
            console.print(f"[green]Deleted run directory: {run_dir}[/]")
        except Exception as e:
            console.print(f"[red]Failed to delete run directory: {e}[/]")

    if os.path.isfile(run_config):
        try:
            os.remove(run_config)
            console.print(f"[green]Deleted config: {run_config}[/]")
        except Exception as e:
            console.print(f"[yellow]Could not delete config: {e}[/]")

    state = get_state()
    state.training_history = [h for h in state.training_history if h.get("name") != name]
    state.save()
    console.print(f"[green]Removed '{name}' from training history.[/]")
    console.print("[green]Run deleted.[/]")


# ─────────────────────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────────────────────
@app.command()
def logs(
    run_name: str = typer.Argument(..., help="Name of the training run"),
    tail: int = typer.Option(50, "--tail", help="Number of lines to show"),
) -> None:
    """Show training logs for a past run."""
    import glob

    run_dir = os.path.join(SAVES_DIR, run_name)
    if not os.path.isdir(run_dir):
        console.print(f"[red]Run not found: {run_dir}[/]")
        raise typer.Exit(1)

    # Look for any .log files in the run directory
    log_files = glob.glob(os.path.join(run_dir, "*.log"))
    if not log_files:
        # Also check for trainer_state.json or other common artifacts
        console.print(f"[yellow]No .log files found in {run_dir}.[/]")
        console.print("[dim]Showing directory contents:[/]")
        for entry in os.listdir(run_dir):
            console.print(f"  {entry}")
        return

    log_file = log_files[0]
    console.print(f"[dim]Log file: {log_file}[/]\n")
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(0, len(lines) - tail)
        for line in lines[start:]:
            console.print(line.rstrip())
    except Exception as e:
        console.print(f"[red]Could not read log: {e}[/]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────
@app.command()
def evaluate(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to LoRA adapter"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset to evaluate on"),
    template: str = typer.Option(None, "--template", "-t", help="Chat template (default: state.active_template)"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Evaluate a trained adapter on a dataset via LLaMA-Factory."""
    console.print("[bold]phronis Evaluate[/bold]\n")

    state = get_state()
    tpl = template or state.active_template or "qwen3"
    out_dir = output or os.path.join(PROJECT_ROOT, "eval", os.path.basename(adapter))
    os.makedirs(out_dir, exist_ok=True)

    base_model = state.active_model or ""
    adapter_path = adapter if os.path.isdir(adapter) else os.path.join(SAVES_DIR, adapter, "lora")
    if not os.path.isdir(adapter_path):
        console.print(f"[red]Adapter not found: {adapter_path}[/]")
        raise typer.Exit(1)

    # Resolve dataset name to file path
    dataset_file = dataset
    if os.path.isfile(DATASET_INFO):
        with open(DATASET_INFO, "r", encoding="utf-8") as f:
            registry = json.load(f)
        if dataset in registry:
            dataset_file = os.path.join(DATA_DIR, registry[dataset]["file_name"])

    eval_config = {
        "model_name_or_path": base_model,
        "adapter_name_or_path": adapter_path,
        "dataset": dataset_file,
        "template": tpl,
        "output_dir": out_dir,
        "do_train": False,
        "do_eval": True,
        "per_device_eval_batch_size": 1,
        "overwrite_output_dir": True,
    }

    eval_yaml = os.path.join(out_dir, "eval_config.yaml")
    with open(eval_yaml, "w", encoding="utf-8") as f:
        yaml.dump(eval_config, f, default_flow_style=False, allow_unicode=True)

    try:
        from .runner import _get_cli
        cli = _get_cli()
    except Exception as exc:
        console.print(f"[red]llamafactory-cli not found: {exc}[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Running evaluation...[/]")
    try:
        result = subprocess.run(
            [cli, "eval", eval_yaml],
            capture_output=False, text=True, timeout=1800,
        )
        if result.returncode == 0:
            console.print(f"\n[green]Evaluation complete. Results in: {out_dir}[/]")
        else:
            console.print("\n[red]Evaluation failed.[/]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Evaluation error: {e}[/]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────
# Serve
# ─────────────────────────────────────────────────────────────
@app.command()
def serve(
    model: str = typer.Option(..., "--model", "-m", help="Path to exported model"),
    backend: str = typer.Option("vllm", "--backend", help="Inference backend: vllm, tgi"),
    port: int = typer.Option(8000, "--port", help="Server port"),
) -> None:
    """Launch an inference server from an exported model."""
    console.print(f"[bold]phronis Serve[/bold]  ({backend} on port {port})\n")

    model_path = model if os.path.isdir(model) else os.path.join(MODELS_DIR, model)
    if not os.path.isdir(model_path):
        console.print(f"[red]Model not found: {model_path}[/]")
        raise typer.Exit(1)

    if backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            console.print("[red]vLLM is not installed.[/]")
            console.print("[dim]Install: pip install vllm[/]")
            raise typer.Exit(1)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
        ]
    elif backend == "tgi":
        tgi = shutil.which("text-generation-launcher")
        if not tgi:
            console.print("[red]TGI launcher not found.[/]")
            console.print("[dim]Install: https://huggingface.co/docs/text-generation-inference[/]")
            raise typer.Exit(1)
        cmd = [tgi, "--model-id", model_path, "--port", str(port)]
    else:
        console.print(f"[red]Unsupported backend: {backend}. Use: vllm, tgi[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Launching server...[/]")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/]")


# ─────────────────────────────────────────────────────────────
# Convert
# ─────────────────────────────────────────────────────────────
@app.command()
def convert(
    model: str = typer.Option(..., "--model", "-m", help="Path to exported model"),
    format: str = typer.Option("gguf", "--format", help="Target format: gguf, onnx, awq, gptq"),
    output: str = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Convert an exported model to another format."""
    console.print(f"[bold]phronis Convert[/bold]  (format: {format})\n")

    model_path = model if os.path.isdir(model) else os.path.join(MODELS_DIR, model)
    if not os.path.isdir(model_path):
        console.print(f"[red]Model not found: {model_path}[/]")
        raise typer.Exit(1)

    out_path = output or os.path.join(MODELS_DIR, f"{os.path.basename(model_path)}_{format}")
    os.makedirs(out_path, exist_ok=True)

    if format == "gguf":
        convert_script = shutil.which("convert-hf-to-gguf.py")
        if not convert_script:
            # Try llama.cpp repo
            convert_script = shutil.which("convert.py")
        if not convert_script:
            console.print("[red]llama.cpp conversion script not found.[/]")
            console.print("[dim]Install: git clone https://github.com/ggerganov/llama.cpp && pip install -r requirements.txt[/]")
            raise typer.Exit(1)
        cmd = [sys.executable, convert_script, model_path, "--outfile", os.path.join(out_path, "model.gguf")]
    elif format == "onnx":
        try:
            import optimum  # noqa: F401
        except ImportError:
            console.print("[red]optimum is not installed.[/]")
            console.print("[dim]Install: pip install optimum[onnxruntime][/]")
            raise typer.Exit(1)
        cmd = [
            sys.executable, "-m", "optimum-cli", "export", "onnx",
            "--model", model_path, out_path,
        ]
    else:
        console.print(f"[red]Format '{format}' is not yet supported. Use: gguf, onnx[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Converting... This may take a while.[/]")
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)
        if result.returncode == 0:
            console.print(f"\n[green]Conversion complete: {out_path}[/]")
        else:
            console.print("\n[red]Conversion failed.[/]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Conversion error: {e}[/]")
        raise typer.Exit(1)


def entry() -> None:
    """Entry point with transparent isolated-env forwarding."""
    from .env_setup import (
        ensure_isolated_venv,
        forward_to_venv,
        is_inside_isolated_venv,
        is_python_version_compatible,
    )

    cur_major, cur_minor, _ = sys.version_info[:3]
    already_in_venv = is_inside_isolated_venv()
    version_ok = is_python_version_compatible(cur_major, cur_minor)

    if already_in_venv or version_ok:
        app()
        return

    # Wrong python version. Set up (or reuse) the isolated venv and forward.
    console.print(
        f"[yellow]Python {cur_major}.{cur_minor} detected. "
        "CUDA PyTorch wheels are unavailable for this version.[/]"
    )
    console.print("[dim]Setting up an isolated workspace environment...[/]")
    if ensure_isolated_venv(console):
        console.print("[green]Environment ready — restarting inside isolated venv...[/]\n")
        result = forward_to_venv()
        if result is None:
            console.print(
                "[red]Could not launch phronis inside isolated venv. "
                "Try deleting your workspace .venv and re-running.[/]"
            )
            sys.exit(1)
        if result.returncode != 0:
            sys.exit(result.returncode)
        sys.exit(0)
    else:
        console.print(
            "[red]Could not create isolated environment. "
            "Install Python 3.12 and try again.[/]"
        )
        sys.exit(1)


if __name__ == "__main__":
    app()
