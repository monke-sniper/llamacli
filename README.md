<div align="center">
<pre>
    ____  __  ______  ____  _   ___________
   / __ \/ / / / __ \/ __ \/ | / /  _/ ___/
  / /_/ / /_/ / /_/ / / / /  |/ // / \__ \
 / ____/ __  / _, _/ /_/ / /|  // / ___/ /
/_/   /_/ /_/_/ |_|\____/_/ |_/___//____/
</pre>
</div>

<p align="center">
  <em>Fine-tune LLMs in your terminal.</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
  <a href="https://github.com/monke-sniper/phronis/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/monke-sniper/phronis/actions"><img src="https://img.shields.io/badge/tests-192%2F192-brightgreen" alt="Tests"></a>
</p>

<p align="center">
  <a href="https://github.com/hiyouga/LLaMA-Factory"><img src="https://img.shields.io/badge/LLaMA--Factory-backend-orange" alt="LLaMA-Factory"></a>
  <a href="https://typer.tiangolo.com/"><img src="https://img.shields.io/badge/Typer-CLI-black" alt="Typer"></a>
  <a href="https://rich.readthedocs.io/"><img src="https://img.shields.io/badge/Rich-terminal-cyan" alt="Rich"></a>
  <a href="https://github.com/tmbo/questionary"><img src="https://img.shields.io/badge/Questionary-prompts-purple" alt="Questionary"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow" alt="HuggingFace"></a>
</p>

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/monke-sniper/phronis/main/assets/demo.gif" alt="phronis demo" width="700">
</p>

---

## What is phronis?

An interactive terminal app for fine-tuning large language models. Pick a model, drop in some data, train, and chat — all from the command line. No config files, no manual YAML, no Gradio web UI. Just `phronis`.

## Install

```bash
pip install phronis
```

Requires Python 3.11+ and a GPU - preferably an nvidia gpu (CUDA or MPS). No additional setup needed - phronis auto-detects and installs missing dependencies on first run.

## Quick Start

```bash
phronis
```

<p align="center">
  <img src="https://raw.githubusercontent.com/monke-sniper/phronis/main/assets/menu.png" alt="phronis menu" width="600">
</p>

Then follow the menu:

```
 1.  Quick Train        →  3 prompts, smart defaults, go
 2.  Advanced Training  →  full hyperparameter control
 3.  Chat Trained Model →  instantly chat your last fine-tune
 4.  Quick Chat         →  chat with any cached model
 5.  Download Model     →  search & download from HuggingFace
 6.  Download Dataset   →  search & download from HuggingFace
 7.  Export Adapter     →  merge LoRA into standalone model
 8.  View Models        →  browse your cached models
 9.  View Datasets      →  browse available datasets
10.  Add Dataset        →  register a new dataset
11.  Workspace Info     →  show workspace location & status
12.  System Check       →  verify setup & dependencies
13.  Exit
```

## Features

| Feature | Description |
|---|---|
| **Model Download** | Search and download any HuggingFace text-generation model with progress bar and ETA |
| **Dataset Download** | Search and download datasets from HuggingFace with auto-format-detection |
| **Auto-Discover Datasets** | Drop `.json` files in `data/` — alpaca & sharegpt formats detected automatically |
| **Quick Train** | Fine-tune in 3 prompts: pick model → pick dataset → set epochs → train |
| **Smart Defaults** | LoRA rank=8, LR=1e-4, batch=2, cutoff=512, SFT — tuned for low-VRAM setups |
| **Advanced Training** | Full control: stage (SFT/DPO/PPO), method (LoRA/Full/Freeze), all hyperparams |
| **Chat Trained Model** | Instantly loads your most recent training run — no selection needed |
| **Quick Chat** | Chat with any model ± adapter with streaming token output |
| **Export** | Merge LoRA adapters into standalone models for inference |
| **Centralized Workspace** | All files in `./workspace/` — configurable via `workspace.yaml` or env var |
| **Zero-Setup Bootstrap** | Auto-detects and installs missing dependencies on first run |
| **System Check** | Verify Python, LLaMA-Factory, GPU, and workspace setup |
| **Training Summaries** | Each training run produces a modular YAML summary in `configs/` |

## Project Layout

phronis uses a centralized workspace at `./workspace/` by default. You can change this via `workspace.yaml` or the `PHRONIS_WORKSPACE` environment variable.

```
./workspace/
├── GUIDE.md            # full documentation & configuration guide
├── workspace.yaml      # workspace configuration
├── .phronis.yaml      # per-project state
├── data/               # datasets — auto-discovered
│   ├── dataset_info.json
│   └── README.txt       # format guide
├── saves/              # LoRA adapters + checkpoints
├── models/             # exported / merged models
└── configs/            # auto-generated YAML from training runs
    ├── phronis_run_*.yaml         # training config
    └── phronis_run_*_summary.yaml # training summary
```

## Dataset Formats

Drop files in `data/` and they appear in the dropdown automatically.

**Alpaca** (`.json`):
```json
[
  {"instruction": "What is the capital of France?", "output": "Paris"},
  {"instruction": "Write a haiku.", "output": "Silicon dreams wake..."}
]
```

**ShareGPT** (`.json` or `.jsonl`):
```json
[
  {
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"}
    ]
  }
]
```

## CLI Reference

```bash
phronis              # interactive menu
phronis --version    # show version
phronis --help       # show help
phronis setup        # run system check & install dependencies
phronis train ...     # direct training (see below)
```

### Direct training

```bash
phronis train \
  --model Qwen/Qwen3-0.6B \
  --dataset identity \
  --epochs 3.0 \
  --lr 1e-4 \
  --batch 2 \
  --cutoff 512 \
  --lora-rank 8 \
  --template qwen3
```

| Flag | Default | Description |
|---|---|---|
| `--model` `-m` | *required* | HuggingFace model ID |
| `--dataset` `-d` | *required* | Dataset name |
| `--epochs` `-e` | `3.0` | Training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--batch` `-b` | `2` | Batch size per device |
| `--cutoff` | `512` | Max token length |
| `--lora-rank` | `8` | LoRA adapter rank |
| `--stage` `-s` | `sft` | Stage (sft/dpo/ppo/etc) |
| `--template` `-t` | `qwen3` | Chat template |
| `--output` `-o` | auto | Output run name |

## Detailed Documentation

### Table of Contents
1. [Installation & Prerequisites](#installation--prerequisites)
2. [Workspace Deep Dive](#workspace-deep-dive)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Modes Explained](#training-modes-explained)
5. [Advanced CLI Usage](#advanced-cli-usage)
6. [Export & Inference](#export--inference)
7. [Troubleshooting](#troubleshooting)
8. [Environment Variables](#environment-variables)
9. [Development Setup](#development-setup)
10. [Changelog](#changelog)

---

## Installation & Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.12 |
| GPU | CUDA 11.8+ or Apple MPS | NVIDIA RTX 3060+ (12GB VRAM) |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB free | 50 GB free (models are large) |
| OS | Windows 10+, macOS 13+, Linux | Windows 11, Ubuntu 22.04 |

### Quick Install

```bash
pip install phronis
```

### Development Install

```bash
git clone https://github.com/monke-sniper/phronis.git
cd phronis
pip install -e .
```

### Isolated Environment (Recommended for Python 3.14+)

If your system Python is 3.14 or newer, `phronis` will automatically create an isolated virtual environment with a compatible Python version (3.11–3.13) because CUDA PyTorch wheels are not yet available for Python 3.14. On first launch, the tool will:

1. Detect your Python version
2. Find a compatible interpreter (`py -3.12` on Windows, or system Python 3.12 on Linux/macOS)
3. Create a virtual environment at `./workspace/.venv`
4. Install CUDA PyTorch, LLaMA-Factory, and `phronis` inside the venv
5. Create a wrapper script (`phronis.cmd` on Windows) for easy launching

You can skip this and manage your own environment by using Python 3.11–3.13 directly.

---

## Workspace Deep Dive

### Default Structure

```
./workspace/
├── GUIDE.md                        # auto-generated feature guide
├── workspace.yaml                    # workspace configuration
├── .phronis.yaml                   # application state (active model, history, etc.)
├── .venv/                          # isolated Python environment (auto-created)
├── data/
│   ├── dataset_info.json           # registered datasets registry
│   └── README.txt                  # format guide
├── saves/
│   ├── run_20260110_143022/        # LoRA adapter checkpoints
│   └── run_20260110_143022_latest/
├── models/
│   └── qwen3-0.6b-merged/          # exported / merged standalone models
└── configs/
    ├── phronis_run_20260110_143022.yaml
    └── phronis_run_20260110_143022_summary.yaml
```

### Changing the Workspace Location

**Option 1 — Environment variable (temporary):**
```bash
set PHRONIS_WORKSPACE=D:\phronis-data    # Windows
export PHRONIS_WORKSPACE=/data/phronis   # Linux/macOS
phronis
```

**Option 2 — Config file (persistent):**
```bash
python -c "from phronis.workspace import set_workspace_path; set_workspace_path('D:/phronis-data')"
```

Or manually create `./workspace/workspace.yaml`:
```yaml
workspace_path: D:\phronis-data
```

### The State File (`.phronis.yaml`)

The state file remembers:
- `active_model` — last selected or downloaded model
- `active_adapter` — last training run (for quick export)
- `active_dataset` — default dataset for training
- `active_template` — chat template (qwen3, llama3, mistral, etc.)
- `training_history` — list of all completed training runs
- `theme` — UI color theme

You can edit this file directly or let `phronis` manage it automatically.

---

## Dataset Preparation

### Auto-Discovery

Drop `.json` or `.jsonl` files into `data/` (relative to your working directory). `phronis` scans them on startup and auto-detects the format:

| Keys Detected | Format | Example Use |
|---|---|---|
| `instruction` + `output` | Alpaca | instruction-following tasks |
| `messages` (list of `{role, content}`) | ShareGPT | multi-turn conversations |
| `conversations` | ShareGPT variant | multi-turn with metadata |
| `prompt` + `completion` | Alpaca (legacy) | completion tasks |
| `text` | Raw text | pre-tokenized or plain text |

### Creating a Custom Dataset

**Alpaca format** (`my_data.json`):
```json
[
  {
    "instruction": "Translate the following to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  },
  {
    "instruction": "Summarize this article",
    "input": "The quick brown fox jumps over the lazy dog...",
    "output": "A fox jumps over a dog."
  }
]
```

**ShareGPT format** (`chat_data.jsonl`):
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is quantum computing?"},
  {"role": "assistant", "content": "Quantum computing is a form of computation that harnesses quantum mechanical phenomena..."},
  {"role": "user", "content": "How does a qubit differ from a classical bit?"},
  {"role": "assistant", "content": "A classical bit is either 0 or 1, while a qubit can exist in a superposition of both states simultaneously..."}
]}
```

**Registration via CLI:**
```bash
phronis add dataset --name my_data --file data/my_data.json --format alpaca
phronis add dataset --name chat_data --file data/chat_data.jsonl --format sharegpt
```

### Dataset Tips

- **File encoding**: UTF-8. UTF-8-BOM is supported but not recommended.
- **Minimum size**: 10 examples for sanity, 100+ for usable results, 1,000+ for quality.
- **Naming**: Use lowercase with underscores. The name appears in the interactive menu.
- **README.txt**: `phronis` auto-generates `data/README.txt` with format guidelines.

---

## Training Modes Explained

### Stages

| Stage | Full Name | Use Case |
|---|---|---|
| **SFT** | Supervised Fine-Tuning | Teach the model to follow instructions |
| **DPO** | Direct Preference Optimization | Align model with human preferences (pairwise) |
| **PPO** | Proximal Policy Optimization | Reinforcement learning from rewards |
| **KTO** | Kahneman-Tversky Optimization | Preference learning without pairs |
| **ORPO** | Odds Ratio Preference Optimization | Simplified preference optimization |

### Methods

| Method | VRAM Required | Best For |
|---|---|---|
| **LoRA** | 8–16 GB | Most users. Fast, small adapters, easy to swap |
| **Full** | 40–80 GB | Maximum quality, research, when you have the hardware |
| **Freeze** | 12–24 GB | Fine-tuning only specific layers (e.g., head + last 4 layers) |

### Hyperparameters

| Parameter | Default | Effect | When to Change |
|---|---|---|---|
| `epochs` | 3.0 | How many passes over the dataset | Increase for small datasets, decrease for large |
| `lr` (learning rate) | 1e-4 | Step size for weight updates | Lower (5e-5) for full fine-tune, higher (2e-4) for LoRA |
| `batch` | 2 | Samples per device per step | Increase if you have VRAM headroom |
| `cutoff` | 512 | Max tokens per sequence | Increase for long-context tasks (1024–2048) |
| `lora_rank` | 8 | Adapter matrix rank | 4–8 for fast experiments, 16–32 for quality |
| `grad_accum` | 8 | Steps to accumulate before backprop | Effective batch = batch × grad_accum × GPUs |
| `scheduler` | cosine | Learning rate decay curve | Use `linear` for short runs, `constant` for transfer learning |
| `warmup` | 0.1 | % of steps with rising LR | 0.05 for small datasets, 0.2 for large |

### Target-Loss Training

Set `--target-loss 0.9` and `phronis` will automatically stop training when the eval loss drops near 0.9. This saves time and prevents overfitting. Useful when:
- You know the approximate loss floor for your task
- You want to avoid babysitting the training process
- Your dataset is large and you only need partial convergence

---

## Advanced CLI Usage

### Non-Interactive Training

```bash
# Quick one-liner
phronis train --model Qwen/Qwen3-0.6B --dataset identity --epochs 3.0

# Full control
phronis train \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset my_chat_data \
  --epochs 5.0 \
  --lr 5e-5 \
  --batch 4 \
  --cutoff 1024 \
  --lora-rank 16 \
  --stage sft \
  --template llama3 \
  --method lora \
  --scheduler cosine \
  --warmup 0.1 \
  --grad-accum 4 \
  --output my_first_run

# Dry run — print config without training
phronis train --model Qwen/Qwen3-0.6B --dataset identity --dry-run

# Resume from checkpoint
phronis train --model Qwen/Qwen3-0.6B --dataset identity --resume saves/run_20260110_143022/checkpoint-500

# Push to HuggingFace Hub after training
phronis train --model Qwen/Qwen3-0.6B --dataset identity --push-to-hub
```

### Chat from the Command Line

```bash
# Chat with your most recently trained model
phronis chat

# Chat with a specific model
phronis chat --model Qwen/Qwen3-0.6B

# Chat with a model + LoRA adapter
phronis chat --model Qwen/Qwen3-0.6B --adapter saves/run_20260110_143022

# Single-shot (no interactive loop)
phronis chat --model Qwen/Qwen3-0.6B --message "What is the capital of France?" --max-tokens 256
```

### Model & Dataset Management

```bash
# Search and download models
phronis download model Qwen/Qwen3-0.6B
phronis download model unsloth/Llama-3.2-3B-Instruct --no-confirm

# Search and download datasets
phronis download dataset tatsu-lab/alpaca
phronis download dataset mlabonne/FineTome-100k --no-confirm

# List what's cached
phronis list models
phronis list datasets
phronis list history
phronis list adapters

# JSON output for scripting
phronis list models --json
```

### Maintenance Commands

```bash
# System health check
phronis doctor

# Auto-fix missing dependencies
phronis doctor --fix

# Workspace info and disk usage
phronis info
phronis info --json

# Clean up old configs
phronis clean configs

# Nuke everything (with confirmation)
phronis clean all --force

# Self-update via pip
phronis update
phronis update --check   # check only, don't install

# Re-run bootstrap manually
phronis setup
```

### Global Flags

```bash
phronis --workspace /custom/path      # override workspace for this run
phronis --no-color                    # disable colored output
phronis --quiet                       # suppress non-essential output
phronis --verbose                     # debug-level output
```

---

## Export & Inference

### Merge LoRA into a Standalone Model

```bash
# Export the most recent adapter
phronis export

# Export a specific adapter
phronis export --adapter saves/run_20260110_143022 --output models/my_model

# Export and push to HuggingFace Hub
phronis export --adapter saves/run_20260110_143022 --push-to-hub
```

The merged model is a complete, standalone HuggingFace Transformers model that can be loaded with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./workspace/models/my_model")
tokenizer = AutoTokenizer.from_pretrained("./workspace/models/my_model")
```

### Using Exported Models with vLLM / TGI

Exported models are standard HuggingFace models. Start an inference server:

```bash
# vLLM (recommended for throughput)
python -m vllm.entrypoints.openai.api_server \
  --model ./workspace/models/my_model \
  --tensor-parallel-size 1

# HuggingFace TGI
 docker run --gpus all -p 8080:80 \
  -v ./workspace/models:/models \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /models/my_model
```

---

## Troubleshooting

### "No module named phronis"

Make sure the package is installed:
```bash
pip install phronis
# or, for development:
pip install -e .
```

### CUDA out of memory

Reduce memory usage with these adjustments:
```bash
phronis train --model YOUR_MODEL --dataset YOUR_DATASET \
  --batch 1 \
  --cutoff 256 \
  --lora-rank 4 \
  --grad-accum 16
```

Or use quantization: set `quantization_bit=4` in your training config if using the YAML workflow.

### "LLaMA-Factory CLI not found"

Run the bootstrap manually:
```bash
phronis setup
# or
phronis doctor --fix
```

### Isolated environment keeps recreating

If you see the "Setting up isolated environment" message every time, the venv may be corrupt. Delete it and let `phronis` rebuild:
```bash
rm -rf ./workspace/.venv   # Linux/macOS
rmdir /s /q workspace\.venv   # Windows (run from repo root)
```

### Windows: `phronis` command not recognized

After first-run setup, a wrapper is written to `./workspace/phronis.cmd`. Add this directory to your PATH, or use:
```bash
python -m phronis
```

### Dataset not showing up in menu

1. Check the file is `.json` or `.jsonl`
2. Verify it contains one of the supported key patterns (`instruction`/`output`, `messages`, etc.)
3. Look at `data/README.txt` for the exact format expected
4. Register manually: `phronis add dataset --name my_data --file path/to/data.json --format alpaca`

---

## Environment Variables

| Variable | Purpose | Example |
|---|---|---|
| `PHRONIS_WORKSPACE` | Override default workspace location | `/data/phronis` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |
| `HF_TOKEN` | HuggingFace API token (for gated models) | `hf_xxxxxxxx` |
| `CUDA_VISIBLE_DEVICES` | Limit GPU visibility | `0,1` |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch memory allocator config | `expandable_segments:True` |

---

## Development Setup

### Running Tests

```bash
# Full suite
pytest tests/

# With coverage
pytest tests/ --cov=phronis --cov-report=html

# Specific test file
pytest tests/test_cli_functions.py -v
```

### Project Structure

```
phronis/
├── src/phronis/           # main package
│   ├── cli.py             # Typer CLI entry point & screens
│   ├── prompts.py         # interactive questionary prompts
│   ├── runner.py          # training & export subprocess management
│   ├── hf.py              # HuggingFace download & search
│   ├── workspace.py       # workspace initialization & path management
│   ├── state.py           # app state persistence (.phronis.yaml)
│   ├── env_setup.py       # isolated venv creation & forwarding
│   ├── bootstrap.py       # first-run system check
│   ├── logo.py            # ASCII logo via pyfiglet
│   ├── repro.py           # reproducibility metadata
│   └── data/              # bundled demo datasets
├── tests/                 # test suite (190 tests)
├── data/                  # central data directory
│   ├── dataset_info.json
│   └── demo_chat.json
├── pyproject.toml
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints where practical
- Keep CLI screens in `cli.py`, business logic in respective modules
- Add tests for new features

---

## Changelog

See [CHANGELOG.md](https://github.com/monke-sniper/phronis/blob/main/CHANGELOG.md) for version history.

---

## License

Apache 2.0 — [LICENSE](https://github.com/monke-sniper/phronis/blob/main/LICENSE)
