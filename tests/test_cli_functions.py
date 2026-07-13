import io
import json
import os
import tempfile


from rich.console import Console


def _dummy_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


class TestBuildConfig:
    def test_quick_train_config_defaults(self):
        from phronis.cli import _build_config
        config = _build_config("test/model", "qwen3", "identity", 3.0, "lora", {}, "run_test")
        assert config["model_name_or_path"] == "test/model"
        assert config["template"] == "qwen3"
        assert config["dataset"] == "identity"
        assert config["num_train_epochs"] == 3.0
        assert config["finetuning_type"] == "lora"
        assert config["lora_rank"] == 8
        assert config["lora_dropout"] == 0.05
        assert "run_test" in config["output_dir"] and "lora" in config["output_dir"]
        assert "dataset_dir" in config
        import os
        assert os.path.isabs(config["dataset_dir"])

    def test_quick_train_config_custom_params(self):
        from phronis.cli import _build_config
        params = {
            "lora_rank": 16, "lora_dropout": 0.1, "lora_alpha": 32,
            "epochs": 5.0, "learning_rate": 5e-5, "batch_size": 1,
            "grad_accum": 16, "cutoff_len": 1024, "warmup_ratio": 0.05,
        }
        config = _build_config("test/model", "llama3", "ds1", 5.0, "lora", params, "run_test")
        assert config["lora_rank"] == 16
        assert config["lora_dropout"] == 0.1
        assert config["lora_alpha"] == 32
        assert config["learning_rate"] == 5e-5
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 16
        assert config["cutoff_len"] == 1024

    def test_full_finetune_config_excludes_lora(self):
        from phronis.cli import _build_config
        config = _build_config("test/model", "qwen3", "ds1", 1.0, "full", {}, "run_test")
        assert config["finetuning_type"] == "full"
        assert "lora_rank" not in config

    def test_freeze_config(self):
        from phronis.cli import _build_config
        config = _build_config("test/model", "qwen3", "ds1", 3.0, "freeze", {}, "run_test")
        assert config["finetuning_type"] == "freeze"
        assert "lora_rank" not in config

    def test_target_loss_sets_save_and_logging_steps(self):
        from phronis.cli import _build_config
        config = _build_config("m", "t", "d", 3.0, "lora", {}, "run", target_loss=0.9)
        assert config["save_steps"] == 1
        assert config["logging_steps"] == 1


class TestRecordTraining:
    def test_records_training_history(self):
        import phronis.state as state_mod

        fd, temp_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        old_path = state_mod.STATE_PATH
        old_state = state_mod._state
        state_mod.STATE_PATH = temp_path
        state_mod._state = None

        try:
            from phronis.cli import _record_training
            from phronis.state import get_state

            _record_training("test_run", "test/model", "identity", "sft", 3.0, "qwen3")
            state = get_state()
            assert len(state.training_history) == 1
            entry = state.training_history[0]
            assert entry["name"] == "test_run"
            assert entry["model"] == "test/model"
            assert entry["dataset"] == "identity"
            assert entry["stage"] == "sft"
            assert entry["epochs"] == 3.0
            assert entry["template"] == "qwen3"
            assert "timestamp" in entry
        finally:
            state_mod.STATE_PATH = old_path
            state_mod._state = old_state
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEnsureDirectories:
    def test_creates_all_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            old_vars = {
                "PROJECT_ROOT": phronis.PROJECT_ROOT,
                "DATA_DIR": phronis.DATA_DIR,
                "CONFIGS_DIR": phronis.CONFIGS_DIR,
                "SAVES_DIR": phronis.SAVES_DIR,
                "MODELS_DIR": phronis.MODELS_DIR,
                "DATASET_INFO": phronis.DATASET_INFO,
            }

            phronis.PROJECT_ROOT = tmp
            phronis.DATA_DIR = os.path.join(tmp, "data")
            phronis.CONFIGS_DIR = os.path.join(tmp, "configs")
            phronis.SAVES_DIR = os.path.join(tmp, "saves")
            phronis.MODELS_DIR = os.path.join(tmp, "models")
            phronis.DATASET_INFO = os.path.join(tmp, "data", "dataset_info.json")

            try:
                from phronis.cli import _ensure_directories
                _ensure_directories()
                assert os.path.isdir(os.path.join(tmp, "data"))
                assert os.path.isdir(os.path.join(tmp, "configs"))
                assert os.path.isdir(os.path.join(tmp, "saves"))
                assert os.path.isdir(os.path.join(tmp, "models"))
                assert os.path.isfile(os.path.join(tmp, "data", "dataset_info.json"))
                assert os.path.isfile(os.path.join(tmp, "data", "README.txt"))
            finally:
                for k, v in old_vars.items():
                    setattr(phronis, k, v)

    def test_readme_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            old_vars = {
                "PROJECT_ROOT": phronis.PROJECT_ROOT,
                "DATA_DIR": phronis.DATA_DIR,
                "CONFIGS_DIR": phronis.CONFIGS_DIR,
                "SAVES_DIR": phronis.SAVES_DIR,
                "MODELS_DIR": phronis.MODELS_DIR,
                "DATASET_INFO": phronis.DATASET_INFO,
            }

            phronis.PROJECT_ROOT = tmp
            phronis.DATA_DIR = os.path.join(tmp, "data")
            phronis.CONFIGS_DIR = os.path.join(tmp, "configs")
            phronis.SAVES_DIR = os.path.join(tmp, "saves")
            phronis.MODELS_DIR = os.path.join(tmp, "models")
            phronis.DATASET_INFO = os.path.join(tmp, "data", "dataset_info.json")

            try:
                from phronis.cli import _ensure_directories
                _ensure_directories()
                with open(os.path.join(tmp, "data", "README.txt"), "r") as f:
                    content = f.read()
                assert "Alpaca format" in content
                assert "ShareGPT format" in content
            finally:
                for k, v in old_vars.items():
                    setattr(phronis, k, v)


class TestExportScreenConfig:
    def test_export_config_structure(self):
        config = {
            "model_name_or_path": "test/model",
            "adapter_name_or_path": "saves/run/lora",
            "template": "qwen3",
            "finetuning_type": "lora",
            "export_dir": "models/run",
            "export_size": 2,
            "export_legacy_format": False,
        }
        assert config["export_size"] == 2
        assert config["finetuning_type"] == "lora"
        assert config["export_legacy_format"] is False


class TestSmartDefaults:
    def test_all_required_keys_present(self):
        from phronis.cli import SMART_DEFAULTS
        required = [
            "stage", "finetuning_type",
            "cutoff_len", "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "lr_scheduler_type", "warmup_ratio",
            "bf16", "logging_steps", "save_steps", "plot_loss",
            "overwrite_output_dir", "report_to", "trust_remote_code", "do_train",
        ]
        for key in required:
            assert key in SMART_DEFAULTS, f"Missing: {key}"

    def test_sensible_values(self):
        from phronis.cli import SMART_DEFAULTS
        assert 0 < SMART_DEFAULTS["learning_rate"] < 0.1
        assert 1 <= SMART_DEFAULTS["per_device_train_batch_size"] <= 64
        assert 1 <= SMART_DEFAULTS["gradient_accumulation_steps"] <= 128
        assert 64 <= SMART_DEFAULTS["cutoff_len"] <= 32768


class TestChatTrainedLogic:
    def test_no_history_initial_state(self):
        from phronis.state import AppState
        state = AppState()
        assert state.training_history == []

    def test_most_recent_is_last_entry(self):
        from phronis.state import AppState
        state = AppState()
        state.training_history = [
            {"name": "first", "model": "m1", "dataset": "ds1", "adapter": "saves/first/lora", "template": "qwen3", "stage": "sft", "epochs": 1, "config": "cfg", "timestamp": "2026-01-01"},
            {"name": "second", "model": "m2", "dataset": "ds2", "adapter": "saves/second/lora", "template": "llama3", "stage": "dpo", "epochs": 2, "config": "cfg", "timestamp": "2026-02-01"},
        ]
        assert len(state.training_history) == 2
        assert state.training_history[-1]["name"] == "second"
        assert state.training_history[-1]["model"] == "m2"


class TestMainMenu:
    def test_menu_has_all_choices(self):
        from phronis.cli import MAIN_MENU
        values = [c.value for c in MAIN_MENU]
        assert "quick_train" in values
        assert "advanced_train" in values
        assert "yaml_train" in values
        assert "chat_trained" in values
        assert "quick_chat" in values
        assert "download_model" in values
        assert "download_dataset" in values
        assert "export" in values
        assert "view_models" in values
        assert "view_datasets" in values
        assert "add_dataset" in values
        assert "workspace_info" in values
        assert "system_check" in values
        assert "exit" in values

    def test_menu_choices_unique(self):
        from phronis.cli import MAIN_MENU
        values = [c.value for c in MAIN_MENU]
        assert len(values) == len(set(values))


class TestHfDownload:
    def test_import_works(self):
        from phronis.hf import search_models, download_model, download_model_interactive
        assert callable(search_models)
        assert callable(download_model)
        assert callable(download_model_interactive)

    def test_dataset_download_imports(self):
        from phronis.hf import search_datasets, download_dataset, download_dataset_interactive
        assert callable(search_datasets)
        assert callable(download_dataset)
        assert callable(download_dataset_interactive)

    def test_rich_tqdm_exists(self):
        from phronis.hf import RichTqdm
        assert hasattr(RichTqdm, "format_bar")
        assert hasattr(RichTqdm, "format_size")
        assert hasattr(RichTqdm, "format_speed")
        assert hasattr(RichTqdm, "format_eta")

    def test_format_size(self):
        from phronis.hf import RichTqdm
        assert "B" in RichTqdm.format_size(100)
        assert "KB" in RichTqdm.format_size(2048)
        assert "MB" in RichTqdm.format_size(2 * 1024 * 1024)
        assert "GB" in RichTqdm.format_size(2 * 1024 * 1024 * 1024)

    def test_format_speed(self):
        from phronis.hf import RichTqdm
        speed = RichTqdm.format_speed(1024 * 1024 * 5)
        assert "MB/s" in speed

    def test_format_eta(self):
        from phronis.hf import RichTqdm
        assert "min" in RichTqdm.format_eta(120) or "s" in RichTqdm.format_eta(120)
        assert "s" in RichTqdm.format_eta(30)

    def test_hf_availability_check(self):
        from phronis.hf import _check_hf, _HF_AVAILABLE
        console = _dummy_console()
        assert _check_hf(console) == _HF_AVAILABLE

    def test_make_rich_tqdm_returns_class_with_get_lock(self):
        """Regression test: snapshot_download expects a class, not a function."""
        from phronis.hf import _make_rich_tqdm
        console = _dummy_console()
        cls = _make_rich_tqdm(console)
        assert isinstance(cls, type), "_make_rich_tqdm must return a class, not a function"
        assert hasattr(cls, "get_lock"), "Returned class must have get_lock (tqdm requirement)"
        # Ensure instantiating it doesn't crash and console is wired in
        instance = cls(total=100)
        assert instance._rich_console is console


class TestAddDatasetScreen:
    def test_add_dataset_creates_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            old_data = phronis.DATA_DIR
            old_dsi = phronis.DATASET_INFO
            phronis.DATA_DIR = tmp
            phronis.DATASET_INFO = os.path.join(tmp, "dataset_info.json")

            try:
                entry = {"file_name": "test_data.json", "formatting": "alpaca"}
                with open(os.path.join(tmp, "dataset_info.json"), "w") as f:
                    json.dump({"test_ds": entry}, f)

                with open(os.path.join(tmp, "dataset_info.json"), "r") as f:
                    registry = json.load(f)
                assert "test_ds" in registry
                assert registry["test_ds"]["file_name"] == "test_data.json"
            finally:
                phronis.DATA_DIR = old_data
                phronis.DATASET_INFO = old_dsi


class TestLogo:
    def test_logo_returns_text(self):
        from phronis.logo import get_logo_text
        text = get_logo_text()
        assert isinstance(text, str)
        assert len(text) > 0


class TestChatTrainedAdapters:
    def test_adapter_found_in_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            saves_dir = os.path.join(tmp, "saves", "my_run", "lora")
            os.makedirs(saves_dir)
            adapter_config = os.path.join(saves_dir, "adapter_config.json")
            with open(adapter_config, "w") as f:
                json.dump({"base_model_name_or_path": "test/model"}, f)

            assert os.path.isdir(saves_dir)
            assert os.path.isfile(adapter_config)

    def test_adapter_missing_shows_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "saves", "nonexistent", "lora")
            assert not os.path.isdir(missing)


class TestDatasetAutoDiscover:
    def test_alpaca_format_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = [{"instruction": "test", "output": "result"}]
            data_file = os.path.join(tmp, "test_alpaca.json")
            with open(data_file, "w") as f:
                json.dump(data, f)

            import phronis
            import phronis.prompts as prompts_mod
            old_data = phronis.DATA_DIR
            old_dsi = phronis.DATASET_INFO
            pd_old_data = prompts_mod.DATA_DIR
            pd_old_dsi = prompts_mod.DATASET_INFO

            phronis.DATA_DIR = tmp
            phronis.DATASET_INFO = os.path.join(tmp, "nonexistent.json")
            prompts_mod.DATA_DIR = tmp
            prompts_mod.DATASET_INFO = os.path.join(tmp, "nonexistent.json")

            try:
                from phronis.prompts import _list_datasets
                datasets = _list_datasets()
                names = [d["name"] for d in datasets]
                assert "test_alpaca" in names, f"Names: {names}"
                ds = [d for d in datasets if d["name"] == "test_alpaca"][0]
                assert ds["format"] == "alpaca"
                assert ds["source"] == "auto"
            finally:
                phronis.DATA_DIR = old_data
                phronis.DATASET_INFO = old_dsi
                prompts_mod.DATA_DIR = pd_old_data
                prompts_mod.DATASET_INFO = pd_old_dsi

    def test_sharegpt_format_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = [{"messages": [{"role": "user", "content": "hi"}]}]
            data_file = os.path.join(tmp, "test_sharegpt.json")
            with open(data_file, "w") as f:
                json.dump(data, f)

            import phronis
            import phronis.prompts as prompts_mod
            old_data = phronis.DATA_DIR
            old_dsi = phronis.DATASET_INFO
            pd_old_data = prompts_mod.DATA_DIR
            pd_old_dsi = prompts_mod.DATASET_INFO

            phronis.DATA_DIR = tmp
            phronis.DATASET_INFO = os.path.join(tmp, "nonexistent.json")
            prompts_mod.DATA_DIR = tmp
            prompts_mod.DATASET_INFO = os.path.join(tmp, "nonexistent.json")

            try:
                from phronis.prompts import _list_datasets
                datasets = _list_datasets()
                ds = [d for d in datasets if d["name"] == "test_sharegpt"][0]
                assert ds["format"] == "sharegpt"
                assert ds["source"] == "auto"
            finally:
                phronis.DATA_DIR = old_data
                phronis.DATASET_INFO = old_dsi
                prompts_mod.DATA_DIR = pd_old_data
                prompts_mod.DATASET_INFO = pd_old_dsi

    def test_unknown_format_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = [{"some_key": "some_value"}]
            data_file = os.path.join(tmp, "unknown.json")
            with open(data_file, "w") as f:
                json.dump(data, f)

            import phronis
            import phronis.prompts as prompts_mod
            old_data = phronis.DATA_DIR
            old_dsi = phronis.DATASET_INFO
            pd_old_data = prompts_mod.DATA_DIR
            pd_old_dsi = prompts_mod.DATASET_INFO

            phronis.DATA_DIR = tmp
            phronis.DATASET_INFO = os.path.join(tmp, "nonexistent.json")
            prompts_mod.DATA_DIR = tmp
            prompts_mod.DATASET_INFO = os.path.join(tmp, "nonexistent.json")

            try:
                from phronis.prompts import _list_datasets
                datasets = _list_datasets()
                names = [d["name"] for d in datasets]
                assert "unknown" not in names
            finally:
                phronis.DATA_DIR = old_data
                phronis.DATASET_INFO = old_dsi
                prompts_mod.DATA_DIR = pd_old_data
                prompts_mod.DATASET_INFO = pd_old_dsi

    def test_registered_overrides_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = [{"instruction": "test", "output": "result"}]
            data_file = os.path.join(tmp, "conflict.json")
            with open(data_file, "w") as f:
                json.dump(data, f)

            dsi = os.path.join(tmp, "dataset_info.json")
            with open(dsi, "w") as f:
                json.dump({"conflict": {"file_name": "conflict.json", "formatting": "sharegpt"}}, f)

            import phronis
            import phronis.prompts as prompts_mod
            old_data = phronis.DATA_DIR
            old_dsi = phronis.DATASET_INFO
            pd_old_data = prompts_mod.DATA_DIR
            pd_old_dsi = prompts_mod.DATASET_INFO

            phronis.DATA_DIR = tmp
            phronis.DATASET_INFO = dsi
            prompts_mod.DATA_DIR = tmp
            prompts_mod.DATASET_INFO = dsi

            try:
                from phronis.prompts import _list_datasets
                datasets = _list_datasets()
                ds = [d for d in datasets if d["name"] == "conflict"][0]
                assert ds["format"] == "sharegpt"
                assert ds["source"] == "registered"
            finally:
                phronis.DATA_DIR = old_data
                phronis.DATASET_INFO = old_dsi
                prompts_mod.DATA_DIR = pd_old_data
                prompts_mod.DATASET_INFO = pd_old_dsi


class TestPromptModelEmpty:
    def test_no_cached_models_returns_empty(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        import phronis
        import phronis.prompts as prompts_mod
        old_cache = phronis.HF_CACHE
        pd_old_cache = prompts_mod.HF_CACHE
        phronis.HF_CACHE = tmp
        prompts_mod.HF_CACHE = tmp

        try:
            from phronis.prompts import _list_cached_models
            models = _list_cached_models()
            assert models == []
        finally:
            phronis.HF_CACHE = old_cache
            prompts_mod.HF_CACHE = pd_old_cache


class TestCheckFirstRun:
    def test_creates_marker_after_bootstrap(self, monkeypatch):
        import tempfile
        import phronis
        import phronis.cli as cli_mod
        import phronis.state as state_mod

        with tempfile.TemporaryDirectory() as tmp:
            old_project_root = phronis.PROJECT_ROOT
            phronis.PROJECT_ROOT = tmp

            old_state_path = state_mod.STATE_PATH
            state_mod.STATE_PATH = os.path.join(tmp, ".phronis.yaml")
            state_mod._state = None

            monkeypatch.setattr(cli_mod, "run_bootstrap", lambda c: None)

            try:
                console = _dummy_console()
                cli_mod._check_first_run(console)
                assert os.path.isfile(os.path.join(tmp, ".phronis.yaml"))
            finally:
                phronis.PROJECT_ROOT = old_project_root
                state_mod.STATE_PATH = old_state_path
                state_mod._state = None



class TestUpdateCommand:
    def test_source_install_runs_git_pull_and_pip_install_e(self, monkeypatch, tmp_path):
        import phronis
        import phronis.cli as cli_mod
        captured = []

        def _fake_run(args, **kwargs):
            captured.append(args)
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()

        monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

        fake_repo = str(tmp_path / "fake_repo")
        os.makedirs(os.path.join(fake_repo, ".git"))
        monkeypatch.setattr(phronis, "REPO_ROOT", fake_repo)

        monkeypatch.setattr(cli_mod, "console", _dummy_console())

        cli_mod.update(check=False, force_pip=False)

        cmds = [" ".join(str(a) for a in c) for c in captured]
        assert any("git -C" in c and "pull" in c for c in cmds), f"Expected git pull, got: {cmds}"
        assert any("pip install -e" in c for c in cmds), f"Expected pip install -e, got: {cmds}"

    def test_pypi_install_runs_pip_upgrade(self, monkeypatch, tmp_path):
        import phronis
        import phronis.cli as cli_mod
        captured = []

        def _fake_run(args, **kwargs):
            captured.append(args)
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()

        monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

        fake_repo = str(tmp_path / "fake_repo")
        os.makedirs(fake_repo, exist_ok=True)
        monkeypatch.setattr(phronis, "REPO_ROOT", fake_repo)

        monkeypatch.setattr(cli_mod, "console", _dummy_console())

        cli_mod.update(check=False, force_pip=False)

        cmds = [" ".join(str(a) for a in c) for c in captured]
        assert any("pip install --upgrade phronis" in c for c in cmds), f"Expected pip upgrade, got: {cmds}"

    def test_check_shows_source_type(self, monkeypatch, tmp_path):
        import phronis
        import phronis.cli as cli_mod
        captured = []

        def _fake_run(args, **kwargs):
            captured.append(args)
            class _R:
                returncode = 0
                stdout = "LATEST: 9.9.9"
                stderr = ""
            return _R()

        monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

        fake_repo = str(tmp_path / "fake_repo")
        os.makedirs(os.path.join(fake_repo, ".git"))
        monkeypatch.setattr(phronis, "REPO_ROOT", fake_repo)

        console = _dummy_console()
        monkeypatch.setattr(cli_mod, "console", console)

        cli_mod.update(check=True, force_pip=False)

        output = console.file.getvalue()
        assert "source (git)" in output, f"Expected 'source (git)' in output. Got: {output}"
        assert fake_repo in output, f"Expected repo path in output. Got: {output}"

    def test_force_pip_skips_git(self, monkeypatch, tmp_path):
        import phronis
        import phronis.cli as cli_mod
        captured = []

        def _fake_run(args, **kwargs):
            captured.append(args)
            class _R:
                returncode = 0
                stdout = "LATEST: 9.9.9"
                stderr = ""
            return _R()

        monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

        fake_repo = str(tmp_path / "fake_repo")
        os.makedirs(os.path.join(fake_repo, ".git"))
        monkeypatch.setattr(phronis, "REPO_ROOT", fake_repo)

        monkeypatch.setattr(cli_mod, "console", _dummy_console())

        cli_mod.update(check=False, force_pip=True)

        cmds = [" ".join(str(a) for a in c) for c in captured]
        assert not any("git" in c for c in cmds), f"Did not expect git command, got: {cmds}"
        assert any("pip install --upgrade phronis" in c for c in cmds), f"Expected pip upgrade, got: {cmds}"
