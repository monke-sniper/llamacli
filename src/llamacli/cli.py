import io
import json
import os
import sys
from datetime import datetime

import questionary
import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import CONFIGS_DIR, MODELS_DIR, SAVES_DIR
from .hf import download_model_interactive, download_dataset_interactive
from .bootstrap import run_bootstrap
from .logo import print_logo
from .prompts import (
    _count_dataset,
    _list_cached_models,
    _list_datasets,
    detect_template,
    prompt_chat_model,
    prompt_dataset,
    prompt_finetuning_type,
    prompt_model,
    prompt_stage,
    prompt_training_params,
)
from .runner import run_export, run_training
from .state import get_state, reload_state

app = typer.Typer(name="llamacli", help="LLaMA-Factory Interactive CLI", add_completion=False)

console = Console(
    file=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"),
    force_terminal=True,
)

MAIN_MENU = [
    questionary.Choice(title="  Quick Train", value="quick_train"),
    questionary.Choice(title="  Advanced Training", value="advanced_train"),
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


def show_main_menu():
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


SMART_DEFAULTS = {
    "stage": "sft",
    "finetuning_type": "lora",
    "cutoff_len": 512,
    "max_samples": 10000,
    "preprocessing_num_workers": 8,
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


def _build_config(model, template, dataset, epochs, finetuning_type, params, output_name):
    config = {
        "model_name_or_path": model,
        "template": template,
        "dataset": dataset,
        "output_dir": os.path.join("saves", output_name, "lora"),
    }
    config.update(SMART_DEFAULTS)
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

    return config


def _record_training(output_name, model, dataset, stage, epochs, template):
    state = get_state()
    state.active_model = model
    state.active_dataset = dataset.split(",")[0].strip() if dataset else ""
    state.active_template = template
    state.active_adapter = os.path.join("saves", output_name, "lora")
    record = {
        "name": output_name,
        "model": model,
        "adapter": os.path.join("saves", output_name, "lora"),
        "template": template,
        "dataset": dataset,
        "stage": stage,
        "epochs": epochs,
        "config": os.path.join("configs", f"llamacli_{output_name}.yaml"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    state.training_history.append(record)
    state.save()

    summary_path = os.path.join(CONFIGS_DIR, f"llamacli_{output_name}_summary.yaml")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(record, f, default_flow_style=False, allow_unicode=True)
    except OSError:
        pass


def _write_config_and_train(console, config, output_name):
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f"llamacli_{output_name}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    console.print(f"\n[dim]Config saved: {config_path}[/]")
    return run_training(console, config_path, output_name)


def quick_train(console):
    state = get_state()
    console.print("\n[bold white]Quick Train[/bold white]\n")

    model, template = prompt_model(console)
    if not model:
        console.print("[dim]Cancelled.[/]")
        return

    dataset = prompt_dataset(console)
    if not dataset:
        console.print("[dim]Cancelled.[/]")
        return

    epochs_val = console.input("[dim]Number of epochs (default 3): [/]").strip()
    try:
        epochs = float(epochs_val) if epochs_val else 3.0
    except ValueError:
        console.print("[yellow]Invalid, using 3.0[/]")
        epochs = 3.0

    config = _build_config(model, template, dataset, epochs, "lora", {}, "")
    console.print("\n[dim]Using smart defaults: LoRA rank=8, LR=1e-4, batch=2, cutoff=512[/]")
    table = Table(title="Quick Train Configuration", show_header=False, border_style="white")
    table.add_column("Key", style="bold white", width=22)
    table.add_column("Value", style="white")
    for k in ("model_name_or_path", "template", "dataset", "num_train_epochs",
              "finetuning_type", "lora_rank", "learning_rate", "cutoff_len"):
        table.add_row(k, str(config.get(k, "")))
    console.print(table)

    try:
        confirmed = questionary.confirm("Start training?", default=True).ask()
    except (KeyboardInterrupt, EOFError):
        confirmed = False
    if not confirmed:
        console.print("[dim]Cancelled.[/]")
        return

    output_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    success = _write_config_and_train(console, config, output_name)
    if success:
        console.print(f"\n[green]Training complete! Output: saves/{output_name}/lora[/]")
        _record_training(output_name, model, dataset, "sft", epochs, template)
    else:
        console.print("\n[red]Training failed.[/]")


def advanced_train(console):
    state = get_state()
    console.print("\n[bold white]Advanced Training[/bold white]\n")

    model, template = prompt_model(console)
    if not model:
        console.print("[dim]Cancelled.[/]")
        return

    dataset = prompt_dataset(console)
    if not dataset:
        console.print("[dim]Cancelled.[/]")
        return

    stage = prompt_stage(console)
    if not stage:
        console.print("[dim]Cancelled.[/]")
        return

    finetuning_type = prompt_finetuning_type(console)
    if not finetuning_type:
        console.print("[dim]Cancelled.[/]")
        return

    params = prompt_training_params(console, finetuning_type)
    if not params:
        console.print("[dim]Cancelled.[/]")
        return

    output_name = console.input(
        f"[dim]Output name (default: run_{datetime.now().strftime('%Y%m%d_%H%M%S')}): [/]"
    ).strip()
    if not output_name:
        output_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = _build_config(model, template, dataset, params.get("epochs", 3), finetuning_type, params, output_name)
    config["stage"] = stage

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

    success = _write_config_and_train(console, config, output_name)
    if success:
        console.print(f"\n[green]Training complete! Output: saves/{output_name}/lora[/]")
        _record_training(output_name, model, dataset, stage, params.get("epochs", 3), template)
    else:
        console.print("\n[red]Training failed.[/]")


def chat_trained(console):
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


def quick_chat(console):
    console.print("\n[bold white]Quick Chat[/bold white]\n")
    model, adapter, template = prompt_chat_model(console)
    if not model:
        console.print("[dim]Cancelled.[/]")
        return
    _start_chat(console, model, adapter, template)


def _start_chat(console, model, adapter, template):
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

    console.print("[dim]Loading model...[/]")
    try:
        chat_model = ChatModel(config)
    except FileNotFoundError as e:
        console.print(f"[red]Model not found locally: {model}[/]")
        console.print(f"[dim]Use 'Download Model' from the menu to download it first.[/]")
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


def view_models_screen(console):
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


def view_datasets_screen(console):
    from . import DATA_DIR

    datasets = _list_datasets()
    state = get_state()
    console.print(f"\n[bold white]Available Datasets ({len(datasets)})[/bold white]\n")
    if not datasets:
        console.print(f"[dim]No datasets found.[/]")
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


def add_dataset_screen(console):
    from . import DATA_DIR, DATASET_INFO
    console.print("\n[bold white]Add Dataset[/bold white]\n")

    name = console.input("[dim]Dataset name (used in configs): [/]").strip()
    if not name:
        console.print("[dim]Cancelled.[/]")
        return

    console.print("[dim]Source type:[/]")
    src = questionary.select(
        "Where is the data?",
        choices=[
            questionary.Choice("Local file in data/ (e.g. my_data.json)", value="file"),
            questionary.Choice("HuggingFace dataset (hf_hub_url)", value="hf"),
        ],
        pointer=">",
        use_arrow_keys=True,
        use_jk_keys=True,
    ).ask()

    if not src:
        return

    if src == "file":
        file_name = console.input(f"[dim]Filename in {DATA_DIR}/ (e.g. my_data.json): [/]").strip()
        if not file_name:
            return
        fmt = questionary.select(
            "Format:",
            choices=[
                questionary.Choice("alpaca (instruction/input/output)", value="alpaca"),
                questionary.Choice("sharegpt (messages)", value="sharegpt"),
            ],
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
        ).ask()
        if not fmt:
            return
        entry = {"file_name": file_name, "formatting": fmt}
    else:
        url = console.input("[dim]HuggingFace dataset URL: [/]").strip()
        if not url:
            return
        fmt = questionary.select(
            "Format:",
            choices=[
                questionary.Choice("alpaca", value="alpaca"),
                questionary.Choice("sharegpt", value="sharegpt"),
            ],
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
        ).ask()
        if not fmt:
            return
        entry = {"hf_hub_url": url, "formatting": fmt}

    os.makedirs(DATA_DIR, exist_ok=True)
    registry = {}
    if os.path.isfile(DATASET_INFO):
        with open(DATASET_INFO, "r", encoding="utf-8") as f:
            registry = json.load(f)
    registry[name] = entry
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Dataset '{name}' added to dataset_info.json[/]")


def export_screen(console):
    state = get_state()
    console.print("\n[bold white]Export / Merge Adapter[/bold white]\n")

    adapter = console.input(
        f"[dim]Adapter path (default: {state.active_adapter or 'none'}): [/]"
    ).strip()
    if not adapter:
        adapter = state.active_adapter
    if not adapter:
        console.print("[yellow]No adapter specified.[/]")
        return

    dest_default = os.path.join(
        "models",
        os.path.basename(adapter.rstrip("/\\").replace("/lora", "").replace("\\lora", "")),
    )
    dest = console.input(f"[dim]Export destination (default: {dest_default}): [/]").strip()
    if not dest:
        dest = dest_default

    config = {
        "model_name_or_path": state.active_model or "Qwen/Qwen3-0.6B",
        "adapter_name_or_path": adapter,
        "template": state.active_template or "qwen3",
        "finetuning_type": "lora",
        "export_dir": dest,
        "export_size": 2,
        "export_legacy_format": False,
    }

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, "llamacli_export_temp.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    run_export(console, config_path)


def workspace_info_screen(console):
    from . import PROJECT_ROOT, DATA_DIR, SAVES_DIR, MODELS_DIR, CONFIGS_DIR
    console.print("\n[bold white]Workspace Info[/bold white]\n")
    console.print(f"[bold]Project root:[/] [dim]{PROJECT_ROOT}[/]")
    console.print(f"[bold]Data dir:[/]    [dim]{DATA_DIR}[/]")
    console.print(f"[bold]Saves dir:[/]   [dim]{SAVES_DIR}[/]")
    console.print(f"[bold]Models dir:[/] [dim]{MODELS_DIR}[/]")
    console.print(f"[bold]Configs dir:[/] [dim]{CONFIGS_DIR}[/]")

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Directory", style="white")
    table.add_column("Items", style="dim", width=8)
    table.add_column("Size", style="dim", width=12)

    for label, path in [
        ("data/", DATA_DIR),
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


def system_check_screen(console):
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
        table.add_row(
            "GPU (CUDA/MPS)",
            "[green]OK[/]" if gpu_ok else "[yellow]CPU ONLY[/]",
            gpu_name or "No GPU detected",
        )
    except ImportError:
        table.add_row("GPU (CUDA/MPS)", "[yellow]UNKNOWN[/]", "torch not installed")

    from . import DATA_DIR, SAVES_DIR, MODELS_DIR, CONFIGS_DIR
    for name, d in [("data/", DATA_DIR), ("saves/", SAVES_DIR), ("models/", MODELS_DIR), ("configs/", CONFIGS_DIR)]:
        exists = os.path.isdir(d)
        if not exists:
            os.makedirs(d, exist_ok=True)
            exists = True
        table.add_row(name, "[green]OK[/]" if exists else "[red]FAIL[/]", d)

    console.print(table)


def _ensure_directories():
    from . import CONFIGS_DIR, DATA_DIR, MODELS_DIR, SAVES_DIR, DATASET_INFO

    for d in (DATA_DIR, SAVES_DIR, MODELS_DIR, CONFIGS_DIR):
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
            f.write('  [{"instruction": "...", "input": "...", "output": "..."}]\n')
            f.write("\n")
            f.write("ShareGPT format (auto-detected):\n")
            f.write('  [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]\n')
            f.write("\n")
            f.write("Files dropped here appear in the dataset dropdown automatically.\n")

    if not os.path.isfile(DATASET_INFO):
        with open(DATASET_INFO, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def interactive_loop():
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
):
    if version:
        from . import PROJECT_ROOT
        console.print(f"[white]llamacli[/] [bold]{PROJECT_ROOT}[/]")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _check_first_run(console)
        interactive_loop()


def _check_first_run(console):
    from . import PROJECT_ROOT
    marker = os.path.join(PROJECT_ROOT, ".llamacli.yaml")
    if not os.path.isfile(marker):
        console.print("\n[bold cyan]Welcome to llamacli![/bold cyan]")
        console.print("[dim]Let's check your system before we start.[/]\n")
        run_bootstrap(console)
        console.print()


@app.command()
def setup():
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
):
    output_name = output or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {
        "model_name_or_path": model,
        "trust_remote_code": True,
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "dataset": dataset,
        "template": template,
        "cutoff_len": cutoff,
        "max_samples": 10000,
        "preprocessing_num_workers": 8,
        "output_dir": f"saves/{output_name}/lora",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": 8,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
    }
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f"llamacli_{output_name}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    console.print(f"[dim]Config: {config_path}[/]")
    success = run_training(console, config_path, output_name)
    if success:
        console.print(f"\n[green]Training complete![/]")
        _record_training(output_name, model, dataset, stage, epochs, template)
    else:
        raise typer.Exit(1)


def entry():
    app()


if __name__ == "__main__":
    app()
