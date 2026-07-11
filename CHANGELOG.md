# Changelog

## [Unreleased] - 2026-07-11

### Fixed
- **HuggingFace search error**: Removed invalid `direction` parameter from `list_models()` call that broke search in `huggingface-hub>=1.19.0`
- **Dataset auto-detection**: Fixed datasets dropped in `data/` not being detected. Now supports:
  - UTF-8 and UTF-8-BOM encoded files
  - `instruction`/`output` (alpaca)
  - `messages` (sharegpt)
  - `conversations` (sharegpt)
  - `prompt`/`completion` (alpaca)
  - `text` field (alpaca)
  - Skips `README.txt` in data folder

### Added
- **Download Dataset** menu option: Search and download datasets from HuggingFace with auto-format-detection
- **Progress bar with ETA**: Detailed download progress showing percentage, bytes, file count, speed, and estimated time remaining
- **File size display**: Shows individual file sizes and total download size before proceeding, with confirmation prompt
- **Centralized workspace**: All files stored in `~/.llamaworkspace/` by default
  - Configurable via `workspace.yaml` or `LLAMACLII_WORKSPACE` env var
  - Auto-creates `GUIDE.md` with full documentation
  - Structure: `data/`, `saves/`, `models/`, `configs/`
- **Zero-setup bootstrap**: Auto-checks system on first launch
  - Verifies Python >= 3.11, LLaMA-Factory, PyTorch, GPU
  - Asks before installing missing dependencies
  - Option to skip and run later via `llamacli setup`
- **Workspace Info** menu option: Shows workspace location, directory sizes, file counts
- **System Check** menu option: Verifies Python, LLaMA-Factory, GPU, workspace setup
- **`llamacli setup` command**: Run system check and install missing dependencies manually
- **Training run summaries**: Each training run now saves a YAML summary in `configs/`
- **GUIDE.md**: Auto-generated guide file in workspace explaining all features and configuration

### Optimized
- **Model listing**: Added `lru_cache` to `_list_cached_models` for faster repeated calls
- **Training state**: Now records `active_adapter` for export convenience
- **Import structure**: Migrated from `list_models()` to `HfApi.list_models()` for modern API compatibility

### New Menu Structure
```
 1.  Quick Train
 2.  Advanced Training
 3.  Chat Trained Model
 4.  Quick Chat
 5.  Download Model
 6.  Download Dataset      [NEW]
 7.  Export Adapter
 8.  View Models
 9.  View Datasets
10.  Add Dataset
11.  Workspace Info        [NEW]
12.  System Check          [NEW]
13.  Exit
```
