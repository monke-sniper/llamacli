from typing import Any

import json
import os

from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm as tqdm_auto

try:
    from huggingface_hub import HfApi, snapshot_download
    _HF_AVAILABLE = True
    _api = HfApi()
except ImportError:
    _HF_AVAILABLE = False
    _api = None


class RichTqdm(tqdm_auto):
    """Custom tqdm class that bridges huggingface_hub progress to Rich console."""

    def __init__(self, *args, **kwargs):
        console = kwargs.pop("console", None)
        self._rich_console = console
        self._files_done = 0
        self._files_total = kwargs.get("total", 0) or 0
        kwargs["disable"] = False
        kwargs.setdefault("unit", "B")
        kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("unit_divisor", 1024)
        kwargs.setdefault("desc", "Downloading")
        super().__init__(*args, **kwargs)

    def refresh(self, nolock=False, lock_args=None):
        if self._rich_console and self.n > 0:
            pct = self.format_percentage(self.n, self.total)
            speed = self.format_speed(self._rate())
            eta = self.format_eta(self._eta())
            size = self.format_size(self.n)
            total = self.format_size(self.total) if self.total else "?"
            bar = self.format_bar(self.n, self.total)
            self._rich_console.print(
                f"\r{bar} {pct} | {size}/{total} | {speed} | {eta}",
                end="",
            )
        super().refresh(nolock, lock_args)

    @staticmethod
    def format_bar(n, total):
        if not total:
            return "[>...]"
        filled = int(20 * n / total)
        return "[" + "=" * filled + ">" + " " * (19 - filled) + "]"

    @staticmethod
    def format_percentage(n, total):
        if total:
            return f"{100 * n / total:5.1f}%"
        return "  ?  %"

    @staticmethod
    def format_size(n):
        if n is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    @staticmethod
    def format_speed(rate):
        if rate is None or rate <= 0:
            return "? B/s"
        return RichTqdm.format_size(rate) + "/s"

    @staticmethod
    def format_eta(eta):
        if eta is None or eta <= 0:
            return "~? min left"
        mins = int(eta // 60)
        secs = int(eta % 60)
        if mins > 0:
            return f"~{mins}m {secs}s left"
        return f"~{secs}s left"

    def _rate(self):
        return self.format_dict.get("rate", None)

    def _eta(self):
        return self.format_dict.get("eta", None)


def _check_hf(console: Console) -> bool:
    if not _HF_AVAILABLE:
        console.print("[red]huggingface_hub is not installed. Install it: pip install huggingface-hub[/]")
        return False
    return True


def _get_repo_files(repo_id: str, repo_type: str = "model") -> list[dict[str, Any]]:
    """Get list of files and their sizes from a HuggingFace repo."""
    try:
        if repo_type == "model":
            info = _api.model_info(repo_id)
        else:
            info = _api.dataset_info(repo_id)
        files = []
        for sibling in info.siblings:
            size = getattr(sibling, "size", None)
            files.append({"name": sibling.rfilename, "size": size})
        return files
    except Exception:
        return []


def _format_total_size(files: list[dict[str, Any]]) -> str:
    """Format total size from a list of file dicts."""
    total = sum(f.get("size", 0) or 0 for f in files)
    if total == 0:
        return "unknown"
    return RichTqdm.format_size(total)


def search_models(console: Console, query: str) -> str | None:
    if not _check_hf(console):
        return None

    models = []
    try:
        with console.status(f'[bold green]Searching HuggingFace for "{query}"...', spinner="dots"):
            results = list(_api.list_models(search=query, sort="downloads", limit=20))
        results.reverse()
        for m in results:
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


def search_datasets(console: Console, query: str) -> str | None:
    if not _check_hf(console):
        return None

    datasets = []
    try:
        with console.status(f'[bold green]Searching HuggingFace datasets for "{query}"...', spinner="dots"):
            results = list(_api.list_datasets(search=query, sort="downloads", limit=20))
        results.reverse()
        for d in results:
            datasets.append(d)
            if len(datasets) >= 10:
                break
    except Exception as e:
        console.print(f"[red]Dataset search failed: {e}[/]")
        console.print("[dim]Check your internet connection and try again.[/]")
        return None

    if not datasets:
        console.print("[yellow]No datasets found. Try a different query.[/]")
        return None

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("#", style="dim", width=4)
    table.add_column("Dataset ID", style="white")
    table.add_column("Downloads", style="dim", width=14)
    table.add_column("Likes", style="dim", width=8)

    for i, d in enumerate(datasets, 1):
        table.add_row(
            str(i),
            d.id,
            f"{getattr(d, 'downloads', 0) or 0:,}",
            str(getattr(d, 'likes', 0) or 0),
        )
    console.print(table)

    console.print("\n[dim]Enter a number to download, or press Enter to skip.[/]")
    choice = console.input("[dim]# [/]").strip()
    if not choice or not choice.isdigit():
        return None

    idx = int(choice) - 1
    if idx < 0 or idx >= len(datasets):
        return None
    return datasets[idx].id


def _show_file_sizes(console: Console, repo_id: str, repo_type: str = "model") -> bool:
    """Show file sizes before download and ask for confirmation."""
    files = _get_repo_files(repo_id, repo_type)
    if not files:
        console.print("[dim]Could not fetch file sizes. Proceeding with download...[/]")
        return True

    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="white",
        title=f"Files in {repo_id}",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="white")
    table.add_column("Size", style="dim", width=12)

    total = 0
    for i, f in enumerate(files, 1):
        size = f.get("size") or 0
        total += size
        table.add_row(str(i), f["name"], RichTqdm.format_size(size))

    console.print(table)
    console.print(f"\n[bold]Total size: {RichTqdm.format_size(total)}[/]")
    console.print()

    try:
        import questionary
        confirmed = questionary.confirm(
            "Proceed with download?",
            default=True,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return False
    return confirmed if confirmed is not None else False


def _make_rich_tqdm(console):
    """Return a RichTqdm subclass that hardcodes the console instance."""
    class _RichTqdm(RichTqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("console", console)
            super().__init__(*args, **kwargs)
    return _RichTqdm


def download_model(console: Console, model_id: str, cache_dir: str | None = None) -> str | None:
    if not _check_hf(console):
        return None

    if not _show_file_sizes(console, model_id, "model"):
        console.print("[dim]Download cancelled.[/]")
        return None

    console.print(f"\n[bold]Downloading {model_id}...[/]\n")
    try:
        path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            tqdm_class=_make_rich_tqdm(console),
        )
        console.print()
        console.print(f"[green]Downloaded to: {path}[/]")
        return path
    except Exception as exc:
        kind = _classify_hf_error(exc)
        if kind == "auth":
            console.print(f"[red]Access denied: {model_id} may require authentication.[/]")
            console.print("[dim]Run: huggingface-cli login[/]")
        elif kind == "not_found":
            console.print(f"[red]Model not found: {model_id}[/]")
            console.print("[dim]Check the model ID and try again.[/]")
        elif kind == "network":
            console.print("[red]Network error. Check your internet connection.[/]")
        elif kind == "disk":
            console.print("[red]Not enough disk space to download this model.[/]")
        else:
            console.print(f"[red]Download error: {exc}[/]")
        return None


def download_dataset(console: Console, dataset_id: str, local_dir: str | None = None) -> str | None:
    if not _check_hf(console):
        return None

    if not _show_file_sizes(console, dataset_id, "dataset"):
        console.print("[dim]Download cancelled.[/]")
        return None

    console.print(f"\n[bold]Downloading dataset {dataset_id}...[/]\n")
    try:
        path = snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_dir=local_dir,
            tqdm_class=_make_rich_tqdm(console),
        )
        console.print()
        console.print(f"[green]Dataset downloaded to: {path}[/]")
        return path
    except Exception as exc:
        kind = _classify_hf_error(exc)
        if kind == "auth":
            console.print(f"[red]Access denied: {dataset_id} may require authentication.[/]")
            console.print("[dim]Run: huggingface-cli login[/]")
        elif kind == "not_found":
            console.print(f"[red]Dataset not found: {dataset_id}[/]")
        elif kind == "network":
            console.print("[red]Network error. Check your internet connection.[/]")
        elif kind == "disk":
            console.print("[red]Not enough disk space to download this dataset.[/]")
        else:
            console.print(f"[red]Download error: {exc}[/]")
        return None


def _classify_hf_error(exc: Exception):
    """Classify a huggingface_hub error for user-friendly messages."""
    from requests.exceptions import HTTPError, ConnectionError, Timeout

    status = None
    if isinstance(exc, HTTPError) and exc.response is not None:
        status = exc.response.status_code

    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        GatedRepoError = None
        RepositoryNotFoundError = None

    if GatedRepoError is not None and isinstance(exc, GatedRepoError):
        return "auth"
    if RepositoryNotFoundError is not None and isinstance(exc, RepositoryNotFoundError):
        return "not_found"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if isinstance(exc, (ConnectionError, Timeout)):
        return "network"
    if isinstance(exc, OSError) and ("disk" in str(exc).lower() or "space" in str(exc).lower()):
        return "disk"
    return "unknown"


def download_model_interactive(console: Console) -> None:
    console.print("\n[bold white]Download Model from HuggingFace[/bold white]\n")
    console.print("[dim]Enter a search query or a full model path (e.g. Qwen/Qwen3-0.6B):[/]")
    query = console.input("[dim]Search or model ID: [/]").strip()
    if not query:
        console.print("[dim]Cancelled.[/]")
        return

    if "/" in query and " " not in query:
        model_id = query
    else:
        model_id = search_models(console, query)
        if not model_id:
            return

    path = download_model(console, model_id)
    if path:
        console.print("[dim]The model will appear in the model list next time.[/]")


def download_dataset_interactive(console: Console) -> None:
    console.print("\n[bold white]Download Dataset from HuggingFace[/bold white]\n")
    console.print("[dim]Enter a search query or a full dataset path (e.g. tatsu-lab/alpaca):[/]")
    query = console.input("[dim]Search or dataset ID: [/]").strip()
    if not query:
        console.print("[dim]Cancelled.[/]")
        return

    if "/" in query and " " not in query:
        dataset_id = query
    else:
        dataset_id = search_datasets(console, query)
        if not dataset_id:
            return

    from . import DATA_DIR
    safe_name = dataset_id.replace("/", "_")
    local_dir = os.path.join(DATA_DIR, safe_name)

    path = download_dataset(console, dataset_id, local_dir)
    if path:
        _register_downloaded_dataset(console, dataset_id, safe_name, path)


def _register_downloaded_dataset(console: Console, dataset_id: str, safe_name: str, path: str):
    """Auto-detect format of downloaded dataset and register it in dataset_info.json."""
    from . import DATA_DIR, DATASET_INFO

    json_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".json") or f.endswith(".jsonl"):
                json_files.append(os.path.join(root, f))

    if not json_files:
        console.print("[yellow]No JSON files found in dataset. Manual registration required.[/]")
        return

    fmt = None
    for jf in json_files:
        try:
            if jf.endswith(".jsonl"):
                with open(jf, "r", encoding="utf-8-sig") as f:
                    first_line = f.readline().strip()
                if first_line:
                    first = json.loads(first_line)
            else:
                with open(jf, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                if not isinstance(data, list) or len(data) == 0:
                    continue
                first = data[0]

            if isinstance(first, dict):
                if "instruction" in first and "output" in first:
                    fmt = "alpaca"
                    break
                elif "messages" in first or "conversations" in first:
                    fmt = "sharegpt"
                    break
        except Exception:
            continue

    if not fmt:
        fmt = "alpaca"

    entry = {"file_name": os.path.relpath(json_files[0], DATA_DIR), "formatting": fmt}
    os.makedirs(DATA_DIR, exist_ok=True)
    registry = {}
    if os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    registry[safe_name] = entry
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Dataset '{safe_name}' registered as {fmt} format.[/]")
    console.print("[dim]It will appear in the dataset list next time.[/]")
