import os

from .workspace import init_workspace

_workspace_path, _dirs = init_workspace()

PROJECT_ROOT = _workspace_path
STATE_PATH = os.path.join(PROJECT_ROOT, ".llamacli.yaml")
DATA_DIR = _dirs["data"]
SAVES_DIR = _dirs["saves"]
MODELS_DIR = _dirs["models"]
CONFIGS_DIR = _dirs["configs"]
DATASET_INFO = os.path.join(DATA_DIR, "dataset_info.json")
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

DEFAULT_CONFIG = {
    "active_model": "",
    "active_adapter": "",
    "active_template": "qwen3",
    "active_dataset": "",
    "training_history": [],
    "theme": "dark",
}
