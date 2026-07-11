<p align="center">
  <img src="https://raw.githubusercontent.com/llamacli/llamacli/main/assets/demo.gif" alt="llamacli demo" width="600">
</p>

<p align="center">
  <a href="https://github.com/llamacli/llamacli"><img src="https://img.shields.io/badge/llamacli-v0.1.0-blue" alt="version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="python"></a>
  <a href="https://github.com/llamacli/llamacli/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="license"></a>
</p>

<p align="center"><strong>Tech Stack</strong></p>
<p align="center">
  <a href="https://github.com/hiyouga/LLaMA-Factory">LLaMA-Factory</a> ·
  <a href="https://typer.tiangolo.com/">Typer</a> ·
  <a href="https://rich.readthedocs.io/">Rich</a> ·
  <a href="https://github.com/tmbo/questionary">Questionary</a> ·
  <a href="https://huggingface.co/">HuggingFace</a> ·
  <a href="https://pytorch.org/">PyTorch</a>
</p>

---

## What is llamacli?

An interactive terminal app for fine-tuning large language models. No config files, no boilerplate, no manual YAML wrangling.

```
   ____                           ___
  / / /___ _____ ___  ____ ______/ (_)
 / / / __ `/ __ `__ \/ __ `/ ___/ / /
/ / / /_/ / / / / / / /_/ / /__/ / /
/_/_/\__,_/_/ /_/ /_/\__,_/\___/_/_/
```

## Install

```bash
pip install llamacli
```

## Quick Start

```bash
llamacli
```

Then just follow the menu:

1. **Download a model** — search HuggingFace by name, pick one, it downloads automatically
2. **Add datasets** — drop `.json` files in `data/` and they appear in the dropdown
3. **Quick Train** — pick model, pick dataset, set epochs, confirm
4. **Chat Trained Model** — instant chat with your most recent fine-tune

## Features

| | |
|---|---|
| **Download Models** | Search and download any HuggingFace text-generation model from within the CLI |
| **Auto-Discover Datasets** | Drop `.json` or `.jsonl` files in `data/` — alpaca and sharegpt formats detected automatically |
| **Quick Train** | Fine-tune in 3 prompts with smart defaults (LoRA rank=8, SFT, sensible hyperparams) |
| **Advanced Training** | Full control — stage (SFT/DPO/PPO), method (LoRA/Full/Freeze), rank, LR, batch size, and more |
| **Chat Trained Model** | Instantly loads your most recent training run — no model selection needed |
| **Quick Chat** | Chat with any cached model + adapter or merged model |
| **Export Adapters** | Merge LoRA adapters into a standalone model |

## Project Layout

llamacli creates a standard workspace in whichever directory you run it from:

```
my-project/
├── .llamacli.yaml      # per-project state (auto-created)
├── data/               # datasets go here — auto-discovered
│   ├── dataset_info.json
│   └── README.txt       # format guide
├── saves/              # training output (LoRA adapters + checkpoints)
├── models/             # exported / merged models
└── configs/            # auto-generated training YAML configs
```

## CLI Reference

```bash
llamacli                # interactive menu
llamacli train ...      # direct one-shot training (no wizard)
llamacli --version      # show version
llamacli --help         # show help
```

### `train` flags

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | *required* | Model path or HuggingFace ID |
| `--dataset`, `-d` | *required* | Dataset name |
| `--epochs`, `-e` | `3.0` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--batch`, `-b` | `2` | Per-device batch size |
| `--cutoff` | `512` | Max token length |
| `--lora-rank` | `8` | LoRA adapter rank |
| `--stage`, `-s` | `sft` | Training stage (sft/dpo/ppo/etc) |
| `--template`, `-t` | `qwen3` | Chat template |
| `--output`, `-o` | auto | Output run name |

## Dataset Formats

Drop files in `data/` and they appear automatically:

**Alpaca format** (`.json`):
```json
[
  {"instruction": "What is the capital of France?", "output": "Paris"},
  {"instruction": "Write a haiku about AI.", "output": "Silicon dreams wake..."}
]
```

**ShareGPT format** (`.json` or `.jsonl`):
```json
[
  {
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there! How can I help?"}
    ]
  }
]
```

You can also register datasets explicitly by adding entries to `data/dataset_info.json` or using the *Add Dataset* option in the menu.

## Requirements

- Python 3.11+
- GPU with CUDA (NVIDIA) or MPS (Apple Silicon)
- LLaMA-Factory is installed automatically as a dependency

## License

Apache 2.0 — see [LICENSE](LICENSE)
