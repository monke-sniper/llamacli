import json
import os
import yaml

DEFAULT_WORKSPACE = os.path.join(os.path.expanduser("~"), ".llamaworkspace")
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".llamaworkspace", "workspace.yaml")

GUIDE_CONTENT = """# llamacli Guide

## Workspace
Your workspace is the central location for all llamacli files.
Default: ~/.llamaworkspace/

### Structure
    ~/.llamaworkspace/
    ├── workspace.yaml      # workspace config
    ├── .llamacli.yaml      # app state
    ├── data/               # datasets (auto-detected)
    │   ├── dataset_info.json
    │   └── README.txt
    ├── saves/              # LoRA adapters + checkpoints
    ├── models/             # exported / merged models
    └── configs/            # auto-generated YAML from training runs

## How to Change Workspace Location
Edit workspace.yaml:
    workspace_path: /your/custom/path

Or set environment variable:
    LLAMACLII_WORKSPACE=/your/custom/path

## Menu Options

### 1. Quick Train
Fine-tune in 3 prompts. Picks model, dataset, epochs. Uses smart defaults.

### 2. Advanced Training
Full control over all hyperparameters.

### 3. Chat Trained Model
Instantly chat with your last fine-tune.

### 4. Quick Chat
Chat with any cached model.

### 5. Download Model
Search and download models from HuggingFace. Shows file sizes and progress.

### 6. Download Dataset
Search and download datasets from HuggingFace. Auto-detects format.

### 7. Export Adapter
Merge LoRA adapter into a standalone model.

### 8. View Models
Browse your cached models. Set active model.

### 9. View Datasets
Browse available datasets. Set active dataset.

### 10. Add Dataset
Register a dataset manually (local file or HuggingFace URL).

### 11. Workspace Info
Show workspace location, directory sizes, and file counts.

### 12. System Check
Verify Python, LLaMA-Factory, GPU, and workspace setup.

## Dataset Formats

### Alpaca (.json)
    [
      {"instruction": "What is the capital of France?", "output": "Paris"}
    ]

### ShareGPT (.json or .jsonl)
    [
      {
        "messages": [
          {"role": "user", "content": "Hello!"},
          {"role": "assistant", "content": "Hi there!"}
        ]
      }
    ]

### Auto-Detection
Drop .json or .jsonl files in data/ and they appear automatically.
Supported patterns: instruction/output, messages, conversations, prompt/completion, text.

## Training Configs
Each training run produces a YAML config in configs/.
You can manually edit these YAMLs to customize training.
The configs are modular - each run has its own file.

## CLI Commands
    llamacli              # interactive menu
    llamacli --version    # show version
    llamacli --help       # show help
    llamacli train --model X --dataset Y  # direct training
    llamacli setup        # run setup/health check

## Environment Variables
    LLAMACLII_WORKSPACE   # override workspace location
    HF_HOME               # HuggingFace cache location
"""


def get_workspace_path():
    if env_path := os.environ.get("LLAMACLII_WORKSPACE"):
        return os.path.abspath(env_path)

    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if path := cfg.get("workspace_path"):
                return os.path.abspath(path)
        except (yaml.YAMLError, OSError):
            pass

    return DEFAULT_WORKSPACE


def set_workspace_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    config = {"workspace_path": path}
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return path


def ensure_workspace_dirs(workspace_path):
    dirs = {
        "data": os.path.join(workspace_path, "data"),
        "saves": os.path.join(workspace_path, "saves"),
        "models": os.path.join(workspace_path, "models"),
        "configs": os.path.join(workspace_path, "configs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    readme_path = os.path.join(dirs["data"], "README.txt")
    if not os.path.isfile(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("Place your training datasets here.\n")
            f.write("Supported: .json (alpaca/sharegpt), .jsonl\n")

    dataset_info = os.path.join(dirs["data"], "dataset_info.json")
    if not os.path.isfile(dataset_info):
        with open(dataset_info, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    guide_path = os.path.join(workspace_path, "GUIDE.md")
    if not os.path.isfile(guide_path):
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(GUIDE_CONTENT)

    return dirs


def init_workspace():
    workspace_path = get_workspace_path()
    dirs = ensure_workspace_dirs(workspace_path)
    return workspace_path, dirs
