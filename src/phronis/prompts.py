import json
import os
import glob

import questionary
from rich.console import Console

from . import (
    DATA_DIR,
    DATASET_INFO,
    HF_CACHE,
    MODELS_DIR,
    SAVES_DIR,
    BUNDLED_DATA_DIR,
    BUNDLED_DATASET_INFO,
)
from .state import get_state


def _list_cached_models():
    return _list_cached_models_impl()


STAGES = [
    questionary.Choice("sft  - Supervised Fine-Tuning", value="sft"),
    questionary.Choice("dpo  - Direct Preference Optimization", value="dpo"),
    questionary.Choice("ppo  - Proximal Policy Optimization", value="ppo"),
    questionary.Choice("grpo - Group Relative Policy Optimization", value="grpo"),
    questionary.Choice("kto  - Kahneman-Tversky Optimization", value="kto"),
    questionary.Choice("pt   - Pre-Training", value="pt"),
    questionary.Choice("rm   - Reward Modeling", value="rm"),
]

FINETUNING_TYPES = [
    questionary.Choice("lora  - Low-Rank Adaptation (recommended)", value="lora"),
    questionary.Choice("full  - Full Fine-Tuning (needs more VRAM)", value="full"),
    questionary.Choice("freeze - Freeze layers, train partial", value="freeze"),
]

TEMPLATES = [
    "qwen3", "qwen3_nothink", "llama3", "llama2", "mistral",
    "chatml", "gemma", "deepseek", "deepseek3", "phi", "yi",
    "baichuan", "internlm2", "falcon", "default",
]

TEXT_GEN_MODEL_TYPES = {
    "qwen2", "qwen3", "qwen3_moe", "qwen2_vl", "qwen2_audio", "qwen2_5_vl",
    "llama", "mistral", "gemma", "gemma2", "gemma3_text", "gemma3",
    "gpt_neox", "gpt2", "gpt_bigcode", "gpt_oss", "gptj",
    "falcon", "phi", "phi3", "phi4",
    "stablelm", "baichuan", "deepseek", "deepseek_v2", "deepseek_v3",
    "internlm2", "chatglm", "bloom", "mpt", "olmo", "olmo2", "olmoe",
    "cohere", "cohere2", "dbrx", "mixtral",
    "exaone", "arctic", "jamba", "mamba", "mamba2",
    "nemotron", "minicpm", "minicpm3", "minicpmo",
    "llava", "llava_next", "pixtral", "paligemma",
    "xverse", "yuan", "orion", "skywork",
    "command_r", "dolly",
    "refinedweb", "rwkv", "rwkv6", "rwkv7",
    "plamo2",
    "cogvlm", "cogvlm2",
}

MODEL_TYPE_TO_TEMPLATE = {
    "qwen3": "qwen3", "qwen3_moe": "qwen3",
    "qwen2": "qwen", "qwen2_vl": "qwen2_vl", "qwen2_audio": "qwen2_audio",
    "llama": "llama3",
    "mistral": "mistral",
    "gemma": "gemma", "gemma2": "gemma", "gemma3": "gemma3", "gemma3_text": "gemma3",
    "deepseek": "deepseek", "deepseek_v2": "deepseek", "deepseek_v3": "deepseek3",
    "phi": "phi", "phi3": "phi", "phi4": "phi4",
    "internlm2": "internlm2",
    "chatglm": "chatglm3",
    "baichuan": "baichuan",
    "falcon": "falcon",
    "yi": "yi",
}


def _list_cached_models_impl():
    models = []
    if not os.path.isdir(HF_CACHE):
        return models
    for entry in os.listdir(HF_CACHE):
        if not os.path.isdir(os.path.join(HF_CACHE, entry)) or "--" not in entry:
            continue
        repo_id = "/".join(entry.split("--")[1:])
        snapshots_dir = os.path.join(HF_CACHE, entry, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
        if not snapshots:
            continue
        snapshot = os.path.join(snapshots_dir, snapshots[0])
        config_file = os.path.join(snapshot, "config.json")
        model_type = "?"
        if os.path.isfile(config_file):
            try:
                with open(config_file, "r") as f:
                    cfg = json.load(f)
                model_type = cfg.get("model_type", "?")
            except Exception:
                pass
        if model_type != "?" and model_type not in TEXT_GEN_MODEL_TYPES:
            continue
        model_files = glob.glob(os.path.join(snapshot, "*.safetensors")) + glob.glob(os.path.join(snapshot, "*.bin"))
        size_gb = sum(os.path.getsize(f) for f in model_files if os.path.isfile(f)) / (1024 ** 3)
        models.append({"repo_id": repo_id, "size_gb": size_gb, "model_type": model_type})
    models.sort(key=lambda m: m["repo_id"].lower())
    return models


def _get_model_type_for_repo(repo_id):
    if not os.path.isdir(HF_CACHE):
        return None
    for entry in os.listdir(HF_CACHE):
        if not os.path.isdir(os.path.join(HF_CACHE, entry)):
            continue
        parts = entry.split("--")
        if len(parts) < 2:
            continue
        candidate = "/".join(parts[1:])
        if candidate.lower() == repo_id.lower():
            snapshots_dir = os.path.join(HF_CACHE, entry, "snapshots")
            if os.path.isdir(snapshots_dir):
                snaps = sorted(os.listdir(snapshots_dir), reverse=True)
                if snaps:
                    config_file = os.path.join(snapshots_dir, snaps[0], "config.json")
                    if os.path.isfile(config_file):
                        try:
                            with open(config_file, "r", encoding="utf-8") as f:
                                return json.load(f).get("model_type", "?")
                        except Exception:
                            pass
    return None


def detect_template(repo_id):
    lower = repo_id.lower()

    # Check reasoning model names first — these share model_type with base
    # models (e.g. deepseek-r1-distill-qwen has model_type=qwen2) so the
    # model_type lookup alone returns the wrong template.
    if "deepseek-r1" in lower or "deepseekr1" in lower:
        return "deepseekr1"

    model_type = _get_model_type_for_repo(repo_id)
    if model_type and model_type in MODEL_TYPE_TO_TEMPLATE:
        return MODEL_TYPE_TO_TEMPLATE[model_type]

    if "qwen3" in lower:
        return "qwen3"
    if "qwen2" in lower:
        return "qwen2"
    if "qwen" in lower:
        return "qwen3"
    if "llama-4" in lower or "llama4" in lower:
        return "llama4"
    if "llama-3" in lower or "llama3" in lower:
        return "llama3"
    if "llama-2" in lower or "llama2" in lower:
        return "llama2"
    if "mistral" in lower:
        return "mistral"
    if "gemma-3" in lower or "gemma3" in lower:
        return "gemma3"
    if "gemma-2" in lower or "gemma2" in lower:
        return "gemma"
    if "gemma" in lower:
        return "gemma"
    if "deepseek-v3" in lower or "deepseekv3" in lower:
        return "deepseek3"
    if "deepseek-v2" in lower or "deepseekv2" in lower:
        return "deepseek"
    if "deepseek" in lower:
        return "deepseek"
    if "phi-4" in lower or "phi4" in lower:
        return "phi4"
    if "phi" in lower:
        return "phi"
    if "falcon" in lower:
        return "falcon"
    if "yi" in lower:
        return "yi"
    if "baichuan" in lower:
        return "baichuan"
    if "internlm" in lower:
        return "internlm2"
    return "qwen3"


def _detect_format(data):
    """Detect dataset format from data content.

    Returns a tuple (format, columns, tags) where:
      - format is 'alpaca' or 'sharegpt'
      - columns is None for alpaca, or {"messages": key_name} for sharegpt
      - tags is None for alpaca, or a dict of sharegpt tags for sharegpt
    """
    if not isinstance(data, list) or len(data) == 0:
        return None, None, None
    first = data[0]
    if not isinstance(first, dict):
        return None, None, None
    if "instruction" in first and "output" in first:
        return "alpaca", None, None
    if "messages" in first:
        messages = first["messages"]
        if messages and isinstance(messages[0], dict):
            if "role" in messages[0]:
                return "sharegpt", {"messages": "messages"}, {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                }
            if "from" in messages[0]:
                return "sharegpt", {"messages": "messages"}, {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system",
                }
        return "sharegpt", {"messages": "messages"}, None
    if "conversations" in first:
        messages = first["conversations"]
        if messages and isinstance(messages[0], dict):
            if "role" in messages[0]:
                return "sharegpt", {"messages": "conversations"}, {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                }
            if "from" in messages[0]:
                return "sharegpt", {"messages": "conversations"}, {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system",
                }
        return "sharegpt", {"messages": "conversations"}, None
    if "prompt" in first and "completion" in first:
        return "alpaca", None, None
    if "text" in first:
        return "alpaca", None, None
    return None, None, None


def _cleanup_stale_datasets() -> None:
    """Remove registry entries whose data files no longer exist."""
    if not os.path.isfile(DATASET_INFO):
        return
    try:
        with open(DATASET_INFO, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    cleaned = {}
    for name, info in registry.items():
        file_name = info.get("file_name", "")
        if file_name:
            fpath = os.path.join(DATA_DIR, file_name)
            if os.path.isfile(fpath):
                cleaned[name] = info
            # else: file gone → drop entry
        else:
            # hf_hub_url entries have no local file; keep them
            cleaned[name] = info

    if len(cleaned) != len(registry):
        try:
            with open(DATASET_INFO, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
        except OSError:
            pass


def _ensure_dataset_registered(name: str) -> bool:
    """Register a dataset by scanning DATA_DIR for a matching JSON/JSONL file.

    Returns True if the dataset is now registered (or was already), False otherwise.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Ensure dataset_info.json exists with valid JSON
    if not os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except OSError:
            pass

    # Check if already registered
    registry = {}
    if os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            registry = {}

    if name in registry:
        return True

    # Scan DATA_DIR for a matching file
    candidates = [f"{name}.json", f"{name}.jsonl"]
    for cand in candidates:
        fpath = os.path.join(DATA_DIR, cand)
        if os.path.isfile(fpath):
            # Detect format
            try:
                fmt = None
                cols = None
                tags = None
                if cand.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        first_line = f.readline().strip()
                    if first_line:
                        first = json.loads(first_line)
                        if "instruction" in first and "output" in first:
                            fmt = "alpaca"
                            cols = None
                        elif "messages" in first or "conversations" in first:
                            fmt = "sharegpt"
                            msg_key = "messages" if "messages" in first else "conversations"
                            cols = {"messages": msg_key}
                            msgs = first.get(msg_key, [])
                            if msgs and isinstance(msgs[0], dict):
                                if "role" in msgs[0]:
                                    tags = {
                                        "role_tag": "role",
                                        "content_tag": "content",
                                        "user_tag": "user",
                                        "assistant_tag": "assistant",
                                        "system_tag": "system",
                                    }
                                elif "from" in msgs[0]:
                                    tags = {
                                        "role_tag": "from",
                                        "content_tag": "value",
                                        "user_tag": "human",
                                        "assistant_tag": "gpt",
                                        "system_tag": "system",
                                    }
                        elif "prompt" in first and "completion" in first:
                            fmt = "alpaca"
                            cols = None
                        elif "text" in first:
                            fmt = "alpaca"
                            cols = None
                        else:
                            continue
                    else:
                        continue
                else:
                    try:
                        with open(fpath, "r", encoding="utf-8-sig") as f:
                            data = json.load(f)
                        if not isinstance(data, list) or len(data) == 0:
                            continue
                        fmt, cols, tags = _detect_format(data)
                    except (json.JSONDecodeError, OSError):
                        # Fallback: might be JSONL with .json extension
                        try:
                            with open(fpath, "r", encoding="utf-8-sig") as f:
                                first_line = f.readline().strip()
                            if first_line:
                                first = json.loads(first_line)
                                if isinstance(first, dict):
                                    if "instruction" in first and "output" in first:
                                        fmt = "alpaca"
                                        cols = None
                                    elif "messages" in first or "conversations" in first:
                                        fmt = "sharegpt"
                                        msg_key = "messages" if "messages" in first else "conversations"
                                        cols = {"messages": msg_key}
                                        msgs = first.get(msg_key, [])
                                        if msgs and isinstance(msgs[0], dict):
                                            if "role" in msgs[0]:
                                                tags = {
                                                    "role_tag": "role",
                                                    "content_tag": "content",
                                                    "user_tag": "user",
                                                    "assistant_tag": "assistant",
                                                    "system_tag": "system",
                                                }
                                            elif "from" in msgs[0]:
                                                tags = {
                                                    "role_tag": "from",
                                                    "content_tag": "value",
                                                    "user_tag": "human",
                                                    "assistant_tag": "gpt",
                                                    "system_tag": "system",
                                                }
                                    elif "prompt" in first and "completion" in first:
                                        fmt = "alpaca"
                                        cols = None
                                    elif "text" in first:
                                        fmt = "alpaca"
                                        cols = None
                        except Exception:
                            pass
                        if not fmt:
                            continue

                entry = {"file_name": cand, "formatting": fmt}
                if cols:
                    entry["columns"] = cols
                if tags:
                    entry["tags"] = tags
                registry[name] = entry
                with open(DATASET_INFO, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=2, ensure_ascii=False)
                return True
            except Exception:
                pass

    return False


def _list_datasets():
    datasets = {}
    os.makedirs(DATA_DIR, exist_ok=True)

    # Auto-create empty dataset_info.json if missing
    if not os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except OSError:
            pass

    # Remove stale entries first
    _cleanup_stale_datasets()

    if os.path.isfile(DATASET_INFO):
        try:
            with open(DATASET_INFO, "r", encoding="utf-8") as f:
                registry = json.load(f)
            for name, info in registry.items():
                datasets[name] = {
                    "name": name,
                    "format": info.get("formatting", "alpaca"),
                    "source": "registered",
                }
        except (json.JSONDecodeError, OSError):
            pass

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith(".") or fname == "dataset_info.json" or fname == "README.txt":
            continue
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        name = os.path.splitext(fname)[0]
        if name in datasets:
            continue
        if fname.endswith(".json"):
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                fmt, cols, tags = _detect_format(data)
                if fmt:
                    datasets[name] = {
                        "name": name,
                        "format": fmt,
                        "source": "auto",
                    }
            except (json.JSONDecodeError, OSError):
                # Fallback: might be JSONL with .json extension
                try:
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        first_line = f.readline().strip()
                    if first_line:
                        first = json.loads(first_line)
                        if isinstance(first, dict):
                            if "instruction" in first and "output" in first:
                                datasets[name] = {"name": name, "format": "alpaca", "source": "auto"}
                            elif "messages" in first:
                                datasets[name] = {"name": name, "format": "sharegpt", "source": "auto"}
                            elif "conversations" in first:
                                datasets[name] = {"name": name, "format": "sharegpt", "source": "auto"}
                except (json.JSONDecodeError, OSError):
                    pass
        elif fname.endswith(".jsonl"):
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    first_line = f.readline().strip()
                if first_line:
                    first = json.loads(first_line)
                    if isinstance(first, dict):
                        if "instruction" in first and "output" in first:
                            datasets[name] = {
                                "name": name,
                                "format": "alpaca",
                                "source": "auto",
                            }
                        elif "messages" in first:
                            datasets[name] = {
                                "name": name,
                                "format": "sharegpt",
                                "source": "auto",
                            }
                        elif "conversations" in first:
                            datasets[name] = {
                                "name": name,
                                "format": "sharegpt",
                                "source": "auto",
                            }
            except (json.JSONDecodeError, OSError):
                pass

    return [v for _, v in sorted(datasets.items())]


def _count_dataset(name):
    candidates = [
        os.path.join(DATA_DIR, f"{name}.json"),
        os.path.join(DATA_DIR, f"{name}.jsonl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                if path.endswith(".jsonl"):
                    with open(path, "r", encoding="utf-8") as f:
                        return sum(1 for line in f if line.strip())
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return len(data) if isinstance(data, list) else 1
            except (json.JSONDecodeError, OSError):
                # Fallback: might be JSONL with .json extension
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return sum(1 for line in f if line.strip())
                except Exception:
                    return 0
            except Exception:
                return 0

    if os.path.isfile(DATASET_INFO):
        with open(DATASET_INFO, "r", encoding="utf-8") as f:
            registry = json.load(f)
        info = registry.get(name, {})
        file_name = info.get("file_name", "")
        if file_name:
            fpath = os.path.join(DATA_DIR, file_name)
            if os.path.isfile(fpath):
                try:
                    if fpath.endswith(".jsonl"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            return sum(1 for line in f if line.strip())
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return len(data) if isinstance(data, list) else 1
                except (json.JSONDecodeError, OSError):
                    # Fallback: might be JSONL with .json extension
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            return sum(1 for line in f if line.strip())
                    except Exception:
                        pass
                except Exception:
                    pass
    return 0


def _list_demo_datasets():
    datasets = {}
    if not os.path.isdir(BUNDLED_DATA_DIR):
        return []
    if os.path.isfile(BUNDLED_DATASET_INFO):
        try:
            with open(BUNDLED_DATASET_INFO, "r", encoding="utf-8") as f:
                registry = json.load(f)
            for name, info in registry.items():
                datasets[name] = {
                    "name": name,
                    "format": info.get("formatting", "alpaca"),
                    "source": "demo",
                }
        except (json.JSONDecodeError, OSError):
            pass

    for fname in sorted(os.listdir(BUNDLED_DATA_DIR)):
        if fname.startswith(".") or fname == "dataset_info.json":
            continue
        fpath = os.path.join(BUNDLED_DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        name = os.path.splitext(fname)[0]
        if name in datasets:
            continue
        if fname.endswith(".json"):
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                fmt, cols, tags = _detect_format(data)
                if fmt:
                    datasets[name] = {"name": name, "format": fmt, "source": "demo"}
            except (json.JSONDecodeError, OSError):
                # Fallback: might be JSONL with .json extension
                try:
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        first_line = f.readline().strip()
                    if first_line:
                        first = json.loads(first_line)
                        if isinstance(first, dict):
                            if "instruction" in first and "output" in first:
                                datasets[name] = {"name": name, "format": "alpaca", "source": "demo"}
                            elif "messages" in first:
                                datasets[name] = {"name": name, "format": "sharegpt", "source": "demo"}
                            elif "conversations" in first:
                                datasets[name] = {"name": name, "format": "sharegpt", "source": "demo"}
                except (json.JSONDecodeError, OSError):
                    pass
        elif fname.endswith(".jsonl"):
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    first_line = f.readline().strip()
                if first_line:
                    first = json.loads(first_line)
                    if isinstance(first, dict):
                        if "instruction" in first and "output" in first:
                            datasets[name] = {"name": name, "format": "alpaca", "source": "demo"}
                        elif "messages" in first:
                            datasets[name] = {"name": name, "format": "sharegpt", "source": "demo"}
                        elif "conversations" in first:
                            datasets[name] = {"name": name, "format": "sharegpt", "source": "demo"}
            except (json.JSONDecodeError, OSError):
                pass

    return [v for _, v in sorted(datasets.items())]


def _count_demo_dataset(name):
    candidates = [
        os.path.join(BUNDLED_DATA_DIR, f"{name}.json"),
        os.path.join(BUNDLED_DATA_DIR, f"{name}.jsonl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                if path.endswith(".jsonl"):
                    with open(path, "r", encoding="utf-8") as f:
                        return sum(1 for line in f if line.strip())
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return len(data) if isinstance(data, list) else 1
            except Exception:
                return 0
    return 0


def prompt_model(console: Console, allow_back: bool = False):
    state = get_state()

    while True:
        cached = _list_cached_models()
        if not cached:
            console.print("[yellow]No cached models found.[/]")
            console.print("[dim]Use 'Download Model' from the menu to search HuggingFace.[/]")
            path = console.input("[dim]Or enter a HuggingFace model ID (e.g. Qwen/Qwen3-0.6B): [/]").strip()
            if not path:
                return ("__back__", None) if allow_back else (None, None)
            detected = detect_template(path)
            return path, detected

        choices = []
        if allow_back:
            choices.append(questionary.Choice(title="← Back", value="__back__"))
        for m in cached:
            label = f"{m['repo_id']} ({m['size_gb']:.1f} GB)"
            if m["repo_id"] == state.active_model:
                label += " [active]"
            choices.append(questionary.Choice(title=label, value=m["repo_id"]))
        choices.append(questionary.Choice(title="Custom path...", value="__custom__"))

        try:
            selected = questionary.select(
                "Select a model:",
                choices=choices,
                default=choices[1] if allow_back and len(choices) > 1 else (choices[0] if choices else None),
                pointer=">",
                use_arrow_keys=True,
                use_jk_keys=True,
                instruction="(j/k to move, Enter to select)",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return ("__back__", None) if allow_back else (None, None)

        if selected == "__back__":
            return "__back__", None
        if selected == "__custom__":
            path = console.input("[dim]Enter HuggingFace model path: [/]").strip()
            if not path:
                if allow_back:
                    continue
                return None, None
            selected = path
        if not selected:
            if allow_back:
                continue
            return None, None

        detected = detect_template(selected)
        template_choices = []
        if allow_back:
            template_choices.append(questionary.Choice(title="← Back", value="__back__"))
        template_choices.append(questionary.Choice(title="auto (detect)", value="__auto__"))
        for t in TEMPLATES:
            template_choices.append(questionary.Choice(title=t, value=t))
        try:
            template = questionary.select(
                f"Select template (auto-detected: {detected}):",
                choices=template_choices,
                default="__auto__",
                pointer=">",
                use_arrow_keys=True,
                use_jk_keys=True,
                instruction="(template affects chat format)",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return ("__back__", None) if allow_back else (selected, detected)

        if template == "__back__":
            continue
        if template == "__auto__":
            template = detected
        if not template:
            if allow_back:
                continue
            return selected, detected
        return selected, template


def prompt_dataset(console: Console, allow_back: bool = False):
    from . import DATA_DIR

    datasets = _list_datasets()
    state = get_state()
    if not datasets:
        demo_datasets = _list_demo_datasets()
        if demo_datasets:
            console.print("[yellow]No personal datasets found.[/]")
            console.print(f"[dim]You can use a built-in demo dataset below, drop files in {DATA_DIR}, or use 'Add Dataset'.[/]")
            console.print()
            choices = []
            if allow_back:
                choices.append(questionary.Choice(title="← Back", value="__back__"))
            for d in demo_datasets:
                cnt = _count_demo_dataset(d["name"])
                label = f"{d['name']} ({cnt} ex, {d['format']}) [demo]"
                checked = d["name"] == state.active_dataset
                choices.append(questionary.Choice(title=label, value=d["name"], checked=checked))
            choices.append(questionary.Choice(title="Custom dataset...", value="__custom__"))

            try:
                selected = questionary.checkbox(
                    "Select dataset(s) (Space to toggle):",
                    choices=choices,
                    pointer=">",
                    use_arrow_keys=True,
                    use_jk_keys=True,
                    instruction="(Space to select, Enter to confirm)",
                ).ask()
            except (KeyboardInterrupt, EOFError):
                return None

            if selected is None:
                return None
            if "__back__" in (selected or []):
                return "__back__"
            if isinstance(selected, list) and len(selected) == 0:
                console.print("[yellow]No dataset selected. Press Space on a dataset to select it, then Enter to confirm.[/]")
                return None
            filtered = [s for s in selected if s not in ("__custom__", "__back__")]
            if "__custom__" in selected:
                custom = console.input("[dim]Custom dataset name: [/]").strip()
                if custom:
                    filtered.append(custom)
            return ",".join(filtered) if filtered else None

        console.print("[yellow]No datasets found.[/]")
        console.print(f"[dim]Drop .json or .jsonl files in {DATA_DIR}[/]")
        console.print("[dim]Format: [{instruction: ..., output: ...}, ...] for alpaca, or [{messages: [...]}] for sharegpt[/]")
        console.print()
        if allow_back:
            console.print("[dim]Enter a dataset name/path, type 'back' to go back, or press Enter to cancel.[/]")
        dataset = console.input("[dim]Dataset name/path: [/]").strip()
        if allow_back and dataset.lower() in ("back", "b"):
            return "__back__"
        return dataset if dataset else None

    choices = []
    if allow_back:
        choices.append(questionary.Choice(title="← Back", value="__back__"))
    for d in datasets:
        cnt = _count_dataset(d["name"])
        tag = "[auto]" if d["source"] == "auto" else ""
        label = f"{d['name']} ({cnt} ex, {d['format']}) {tag}"
        checked = d["name"] == state.active_dataset
        choices.append(questionary.Choice(
            title=label,
            value=d["name"],
            checked=checked,
        ))

    choices.append(questionary.Choice(title="Custom dataset...", value="__custom__"))

    try:
        selected = questionary.checkbox(
            "Select dataset(s) (Space to toggle):",
            choices=choices,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(Space to select, Enter to confirm)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None

    if selected is None:
        return None
    if "__back__" in (selected or []):
        return "__back__"
    if isinstance(selected, list) and len(selected) == 0:
        console.print("[yellow]No dataset selected. Press Space on a dataset to select it, then Enter to confirm.[/]")
        return None
    filtered = [s for s in selected if s not in ("__custom__", "__back__")]
    if "__custom__" in selected:
        custom = console.input("[dim]Custom dataset name: [/]").strip()
        if custom:
            filtered.append(custom)
    return ",".join(filtered) if filtered else None


def prompt_stage(console: Console, allow_back: bool = False):
    choices = list(STAGES)
    if allow_back:
        choices.insert(0, questionary.Choice(title="← Back", value="__back__"))
    try:
        selected = questionary.select(
            "Training stage:",
            choices=choices,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    return selected


def prompt_finetuning_type(console: Console, allow_back: bool = False):
    choices = list(FINETUNING_TYPES)
    if allow_back:
        choices.insert(0, questionary.Choice(title="← Back", value="__back__"))
    try:
        selected = questionary.select(
            "Fine-tuning method:",
            choices=choices,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    return selected


def prompt_training_params(console: Console, finetuning_type, allow_back: bool = False):
    params = {}

    if finetuning_type == "lora":
        lora_choices = [
            questionary.Choice("4  - Smaller adapter, less VRAM", value=4),
            questionary.Choice("8  - Balanced (recommended)", value=8),
            questionary.Choice("16 - More capacity, more VRAM", value=16),
        ]
        if allow_back:
            lora_choices.insert(0, questionary.Choice(title="← Back", value="__back__"))
        try:
            rank = questionary.select(
                "LoRA rank:",
                choices=lora_choices,
                default=lora_choices[1] if allow_back and len(lora_choices) > 1 else lora_choices[0],
                pointer=">",
                use_arrow_keys=True,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None
        if rank == "__back__":
            return "__back__"
        params["lora_rank"] = rank if rank else 8
        params["lora_dropout"] = _input_float(console, "LoRA dropout", 0.05, allow_back=allow_back)
        if params["lora_dropout"] == "__back__":
            return "__back__"
        params["lora_alpha"] = _input_int(console, "LoRA alpha (0 = auto)", 0, allow_back=allow_back)
        if params["lora_alpha"] == "__back__":
            return "__back__"

    params["epochs"] = _input_float(console, "Number of epochs", 3.0, allow_back=allow_back)
    if params["epochs"] == "__back__":
        return "__back__"
    params["learning_rate"] = _input_float(console, "Learning rate", 1e-4, allow_back=allow_back)
    if params["learning_rate"] == "__back__":
        return "__back__"
    params["batch_size"] = _input_int(console, "Batch size per device", 2, allow_back=allow_back)
    if params["batch_size"] == "__back__":
        return "__back__"
    params["grad_accum"] = _input_int(console, "Gradient accumulation steps", 8, allow_back=allow_back)
    if params["grad_accum"] == "__back__":
        return "__back__"
    params["cutoff_len"] = _input_int(console, "Cutoff length (tokens)", 512, allow_back=allow_back)
    if params["cutoff_len"] == "__back__":
        return "__back__"

    params["warmup_ratio"] = _input_float(console, "Warmup ratio", 0.1, allow_back=allow_back)
    if params["warmup_ratio"] == "__back__":
        return "__back__"
    return params


def _input_int(console, label, default, allow_back: bool = False):
    prompt_text = f"[dim]{label} (default {default}"
    if allow_back:
        prompt_text += ", or type 'back'"
    prompt_text += "): [/]"
    val = console.input(prompt_text).strip()
    if allow_back and val.lower() in ("back", "b"):
        return "__back__"
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        console.print(f"[yellow]Invalid, using {default}[/]")
        return default


def _input_float(console, label, default, allow_back: bool = False):
    prompt_text = f"[dim]{label} (default {default}"
    if allow_back:
        prompt_text += ", or type 'back'"
    prompt_text += "): [/]"
    val = console.input(prompt_text).strip()
    if allow_back and val.lower() in ("back", "b"):
        return "__back__"
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        console.print(f"[yellow]Invalid, using {default}[/]")
        return default


def prompt_target_loss(console: Console, allow_back: bool = False):
    prompt_text = "[dim]Target loss (optional, e.g. 0.9 1.0 2.5"
    if allow_back:
        prompt_text += ", or type 'back'"
    prompt_text += "; Enter to skip): [/]"
    val = console.input(prompt_text).strip()
    if allow_back and val.lower() in ("back", "b"):
        return "__back__"
    if not val:
        return None
    try:
        target = float(val)
        if target <= 0:
            console.print("[yellow]Target loss must be positive, skipping.[/]")
            return None
        return target
    except ValueError:
        console.print("[yellow]Invalid target loss, skipping.[/]")
        return None


def prompt_chat_model(console: Console, allow_back: bool = False):
    models = _list_cached_models()
    adapter_map = {}
    if os.path.isdir(SAVES_DIR):
        for root, dirs, files in os.walk(SAVES_DIR):
            if "adapter_config.json" in files:
                adapter_rel = os.path.relpath(root, os.path.dirname(SAVES_DIR))
                adapter_cfg_path = os.path.join(root, "adapter_config.json")
                with open(adapter_cfg_path, "r") as f:
                    try:
                        acfg = json.load(f)
                        base_model = acfg.get("base_model_name_or_path", "")
                    except Exception:
                        base_model = ""
                key = base_model or "unknown"
                if key not in adapter_map:
                    adapter_map[key] = []
                adapter_map[key].append(adapter_rel)

    merged_models = []
    if os.path.isdir(MODELS_DIR):
        for entry in sorted(os.listdir(MODELS_DIR)):
            full = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
                merged_models.append(entry)

    if not models and not merged_models:
        console.print("[yellow]No models cached locally.[/]")
        console.print("[dim]Use 'Download Model' from the main menu to download one from HuggingFace.[/]")
        console.print()
        if allow_back:
            console.print("[dim]Enter a model ID, type 'back' to go back, or press Enter to cancel.[/]")
        path = console.input("[dim]Or enter a HuggingFace model ID (e.g. Qwen/Qwen3-0.6B): [/]").strip()
        if allow_back and path.lower() in ("back", "b"):
            return "__back__", None, None
        if not path:
            return None, None, None
        template = detect_template(path)
        return path, None, template

    choices = []
    if allow_back:
        choices.append(questionary.Choice(title="← Back", value="__back__"))
    for m in models:
        choices.append(questionary.Choice(
            title=f"{m['repo_id']} (base, no adapter)",
            value=json.dumps({"model": m["repo_id"], "adapter": None}),
        ))
        for adapter in adapter_map.get(m["repo_id"], []):
            choices.append(questionary.Choice(
                title=f"{m['repo_id']} + {adapter} (adapter)",
                value=json.dumps({"model": m["repo_id"], "adapter": adapter}),
            ))

    for name in merged_models:
        path = os.path.join(MODELS_DIR, name)
        choices.append(questionary.Choice(
            title=f"{name} (merged, models/)",
            value=json.dumps({"model": path, "adapter": None}),
        ))

    try:
        result = questionary.select(
            "Select model to chat with:",
            choices=choices,
            default=choices[1] if allow_back and len(choices) > 1 else choices[0] if choices else None,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return ("__back__", None, None) if allow_back else (None, None, None)

    if result == "__back__":
        return "__back__", None, None
    if not result:
        return None, None, None

    data = json.loads(result)
    model = data["model"]
    adapter = data.get("adapter")
    template = detect_template(model)
    return model, adapter, template
