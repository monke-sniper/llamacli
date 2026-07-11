import json
import os
import glob

import questionary
from rich.console import Console

from . import DATA_DIR, DATASET_INFO, HF_CACHE, MODELS_DIR, SAVES_DIR
from .state import get_state


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


def _list_cached_models():
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
                            with open(config_file, "r") as f:
                                return json.load(f).get("model_type", "?")
                        except Exception:
                            pass
    return None


def detect_template(repo_id):
    model_type = _get_model_type_for_repo(repo_id)
    if model_type and model_type in MODEL_TYPE_TO_TEMPLATE:
        return MODEL_TYPE_TO_TEMPLATE[model_type]
    lower = repo_id.lower()
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
    if "deepseek-r1" in lower or "deepseekr1" in lower:
        return "deepseekr1"
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


def _list_datasets():
    datasets = {}
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isfile(DATASET_INFO):
        with open(DATASET_INFO, "r", encoding="utf-8") as f:
            registry = json.load(f)
        for name, info in registry.items():
            datasets[name] = {
                "name": name,
                "format": info.get("formatting", "alpaca"),
                "source": "registered",
            }

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith(".") or fname == "dataset_info.json":
            continue
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        name = os.path.splitext(fname)[0]
        if name in datasets:
            continue
        if fname.endswith(".json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
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
            except Exception:
                pass
        elif fname.endswith(".jsonl"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if first_line:
                    first = json.loads(first_line)
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
            except Exception:
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
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return len(data) if isinstance(data, list) else 1
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
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return len(data) if isinstance(data, list) else 1
                except Exception:
                    pass
    return 0


def prompt_model(console: Console):
    cached = _list_cached_models()
    state = get_state()
    if not cached:
        console.print(f"[yellow]No cached models found.[/]")
        console.print(f"[dim]Use 'Download Model' from the menu to search HuggingFace.[/]")
        path = console.input("[dim]Or enter a HuggingFace model ID (e.g. Qwen/Qwen3-0.6B): [/]").strip()
        if not path:
            return None, None
        detected = detect_template(path)
        return path, detected

    choices = []
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
            default=choices[0] if choices else None,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None, None

    if selected == "__custom__":
        path = console.input("[dim]Enter HuggingFace model path: [/]").strip()
        if not path:
            return None, None
        selected = path
    if not selected:
        return None, None

    detected = detect_template(selected)
    try:
        template = questionary.select(
            f"Select template (auto-detected: {detected}):",
            choices=TEMPLATES,
            default=detected if detected in TEMPLATES else "qwen3",
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(template affects chat format)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return selected, detected

    return selected, template


def prompt_dataset(console: Console):
    from . import DATA_DIR

    datasets = _list_datasets()
    state = get_state()
    if not datasets:
        console.print(f"[yellow]No datasets found.[/]")
        console.print(f"[dim]Drop .json or .jsonl files in {DATA_DIR}[/]")
        console.print("[dim]Format: [{instruction: ..., output: ...}, ...] for alpaca, or [{messages: [...]}] for sharegpt[/]")
        console.print()
        dataset = console.input("[dim]Or enter a dataset name/path manually (Enter to go back): [/]").strip()
        return dataset if dataset else None

    choices = []
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
    if isinstance(selected, list) and len(selected) == 0:
        console.print("[yellow]No dataset selected. Press Space on a dataset to select it, then Enter to confirm.[/]")
        return None
    filtered = [s for s in selected if s != "__custom__"]
    if "__custom__" in selected:
        custom = console.input("[dim]Custom dataset name: [/]").strip()
        if custom:
            filtered.append(custom)
    return ",".join(filtered) if filtered else None


def prompt_stage(console: Console):
    try:
        return questionary.select(
            "Training stage:",
            choices=STAGES,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None


def prompt_finetuning_type(console: Console):
    try:
        return questionary.select(
            "Fine-tuning method:",
            choices=FINETUNING_TYPES,
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None


def prompt_training_params(console: Console, finetuning_type):
    params = {}

    if finetuning_type == "lora":
        try:
            choice_4 = questionary.Choice("4  - Smaller adapter, less VRAM", value=4)
            choice_8 = questionary.Choice("8  - Balanced (recommended)", value=8)
            choice_16 = questionary.Choice("16 - More capacity, more VRAM", value=16)
            rank = questionary.select(
                "LoRA rank:",
                choices=[choice_4, choice_8, choice_16],
                default=choice_8,
                pointer=">",
                use_arrow_keys=True,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None
        params["lora_rank"] = rank if rank else 8
        params["lora_dropout"] = _input_float(console, "LoRA dropout", 0.05)
        params["lora_alpha"] = _input_int(console, "LoRA alpha (0 = auto)", 0)

    try:
        params["epochs"] = _input_float(console, "Number of epochs", 3.0)
        params["learning_rate"] = _input_float(console, "Learning rate", 1e-4)
        params["batch_size"] = _input_int(console, "Batch size per device", 2)
        params["grad_accum"] = _input_int(console, "Gradient accumulation steps", 8)
        params["cutoff_len"] = _input_int(console, "Cutoff length (tokens)", 512)
    except (KeyboardInterrupt, EOFError):
        return None

    params["warmup_ratio"] = _input_float(console, "Warmup ratio", 0.1)
    return params


def _input_int(console, label, default):
    val = console.input(f"[dim]{label} (default {default}): [/]").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        console.print(f"[yellow]Invalid, using {default}[/]")
        return default


def _input_float(console, label, default):
    val = console.input(f"[dim]{label} (default {default}): [/]").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        console.print(f"[yellow]Invalid, using {default}[/]")
        return default


def prompt_chat_model(console: Console):
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
        path = console.input("[dim]Or enter a HuggingFace model ID (e.g. Qwen/Qwen3-0.6B): [/]").strip()
        if not path:
            return None, None, None
        template = detect_template(path)
        return path, None, template

    choices = []
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
            pointer=">",
            use_arrow_keys=True,
            use_jk_keys=True,
            instruction="(j/k to move, Enter to select)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None, None, None

    if not result:
        return None, None, None

    data = json.loads(result)
    model = data["model"]
    adapter = data.get("adapter")
    template = detect_template(model)
    return model, adapter, template
