import os

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from huggingface_hub import HfApi, list_models, snapshot_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def _check_hf(console: Console):
    if not _HF_AVAILABLE:
        console.print("[red]huggingface_hub is not installed. Install it: pip install huggingface-hub[/]")
        return False
    return True


def search_models(console: Console, query: str):
    if not _check_hf(console):
        return None

    console.print(f'\n[dim]Searching HuggingFace for "{query}"...[/]\n')
    models = []
    try:
        for m in list_models(search=query, sort="downloads", direction=-1, limit=20):
            tags = getattr(m, "tags", None) or []
            pipeline_tag = getattr(m, "pipeline_tag", "") or ""
            if "text-generation" not in tags and pipeline_tag != "text-generation":
                continue
            models.append(m)
            if len(models) >= 10:
                break
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/]")
        console.print("[dim]Check your internet connection and try again.[/]")
        return None

    if not models:
        console.print("[yellow]No text-generation models found. Try a different query.[/]")
        return None

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID", style="white")
    table.add_column("Downloads", style="dim", width=14)
    table.add_column("Likes", style="dim", width=8)

    for i, m in enumerate(models, 1):
        table.add_row(
            str(i),
            m.modelId,
            f"{getattr(m, 'downloads', 0) or 0:,}",
            str(getattr(m, 'likes', 0) or 0),
        )
    console.print(table)

    console.print("\n[dim]Enter a number to download, or press Enter to skip.[/]")
    choice = console.input("[dim]# [/]").strip()
    if not choice or not choice.isdigit():
        return None

    idx = int(choice) - 1
    if idx < 0 or idx >= len(models):
        return None
    return models[idx].modelId


def download_model(console: Console, model_id: str, cache_dir: str = None):
    if not _check_hf(console):
        return None

    console.print(f"\n[bold]Downloading {model_id}...[/]\n")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"[dim]Fetching {model_id}...[/]", total=None)
            path = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                resume_download=True,
            )
            progress.update(task, description=f"[green]Done: {model_id}[/]")
        console.print(f"\n[green]Downloaded to: {path}[/]")
        return path
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            console.print(f"[red]Access denied: {model_id} may require authentication.[/]")
            console.print("[dim]Run: huggingface-cli login[/]")
        elif "404" in msg or "not found" in msg.lower():
            console.print(f"[red]Model not found: {model_id}[/]")
            console.print("[dim]Check the model ID and try again.[/]")
        elif "connection" in msg.lower() or "timeout" in msg.lower():
            console.print("[red]Network error. Check your internet connection.[/]")
        elif "disk" in msg.lower() or "space" in msg.lower():
            console.print("[red]Not enough disk space to download this model.[/]")
        else:
            console.print(f"[red]Download error: {msg}[/]")
        return None


def download_model_interactive(console: Console):
    console.print("\n[bold white]Download Model from HuggingFace[/bold white]\n")
    query = console.input("[dim]Search query (e.g. qwen3, llama3, mistral): [/]").strip()
    if not query:
        console.print("[dim]Cancelled.[/]")
        return

    model_id = search_models(console, query)
    if not model_id:
        return

    path = download_model(console, model_id)
    if path:
        console.print("[dim]The model will appear in the model list next time.[/]")
