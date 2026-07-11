import os


def _get_project_root():
    return os.getcwd()


PROJECT_ROOT = _get_project_root()
STATE_PATH = os.path.join(PROJECT_ROOT, ".llamacli.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVES_DIR = os.path.join(PROJECT_ROOT, "saves")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
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
