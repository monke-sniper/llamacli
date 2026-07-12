<div align="center">
<pre>
   ____                           ___
  / / /___ _____ ___  ____ ______/ (_)
 / / / __ `/ __ `__ \/ __ `/ ___/ / /
/ / / /_/ / / / / / / /_/ / /__/ / /
/_/_/\__,_/_/ /_/ /_/\__,_/\___/_/_/
</pre>
</div>

<p align="center">
  <em>Fine-tune LLMs in your terminal. No config files, no boilerplate.</em>
</p>

<p align="center">
  <a href="https://github.com/monke-sniper/llamacli"><img src="https://img.shields.io/github/v/tag/monke-sniper/llamacli?color=%2334D058&label=version" alt="Version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
  <a href="https://github.com/monke-sniper/llamacli/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/monke-sniper/llamacli/actions"><img src="https://img.shields.io/badge/tests-192%2F192-brightgreen" alt="Tests"></a>
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
  <img src="https://raw.githubusercontent.com/monke-sniper/llamacli/main/assets/demo.gif" alt="llamacli demo" width="700">
</p>

---

## What is llamacli?

An interactive terminal app for fine-tuning large language models. Pick a model, drop in some data, train, and chat — all from the command line. No config files, no manual YAML, no Gradio web UI. Just `llamacli`.

## Install

```bash
pip install llamacli
```

Requires Python 3.11+ and a GPU (CUDA or MPS). No additional setup needed - llamacli auto-detects and installs missing dependencies on first run.

## Quick Start

```bash
llamacli
```

<p align="center">
  <img src="https://raw.githubusercontent.com/monke-sniper/llamacli/main/assets/menu.png" alt="llamacli menu" width="600">
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
| **Centralized Workspace** | All files in `~/.llamaworkspace/` — configurable via `workspace.yaml` or env var |
| **Zero-Setup Bootstrap** | Auto-detects and installs missing dependencies on first run |
| **System Check** | Verify Python, LLaMA-Factory, GPU, and workspace setup |
| **Training Summaries** | Each training run produces a modular YAML summary in `configs/` |

## Project Layout

llamacli uses a centralized workspace at `~/.llamaworkspace/` by default. You can change this via `workspace.yaml` or the `LLAMACLII_WORKSPACE` environment variable.

```
~/.llamaworkspace/
├── GUIDE.md            # full documentation & configuration guide
├── workspace.yaml      # workspace configuration
├── .llamacli.yaml      # per-project state
├── data/               # datasets — auto-discovered
│   ├── dataset_info.json
│   └── README.txt       # format guide
├── saves/              # LoRA adapters + checkpoints
├── models/             # exported / merged models
└── configs/            # auto-generated YAML from training runs
    ├── llamacli_run_*.yaml         # training config
    └── llamacli_run_*_summary.yaml # training summary
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
llamacli              # interactive menu
llamacli --version    # show version
llamacli --help       # show help
llamacli setup        # run system check & install dependencies
llamacli train ...     # direct training (see below)
```

### Direct training

```bash
llamacli train \
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

## License

Apache 2.0 — [LICENSE](https://github.com/monke-sniper/llamacli/blob/main/LICENSE)
