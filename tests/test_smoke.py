import os
import subprocess
import sys

import pytest

from llamacli import PROJECT_ROOT

VENV_CLI = os.path.join(os.path.dirname(sys.executable), "Scripts", "llamacli.exe")


class TestImports:
    def test_cli_imports(self):
        import llamacli.cli
        assert hasattr(llamacli.cli, "app")
        assert hasattr(llamacli.cli, "entry")
        assert hasattr(llamacli.cli, "interactive_loop")
        assert hasattr(llamacli.cli, "quick_train")
        assert hasattr(llamacli.cli, "advanced_train")
        assert hasattr(llamacli.cli, "chat_trained")

    def test_prompts_imports(self):
        import llamacli.prompts
        assert hasattr(llamacli.prompts, "prompt_model")
        assert hasattr(llamacli.prompts, "prompt_dataset")
        assert hasattr(llamacli.prompts, "prompt_stage")
        assert hasattr(llamacli.prompts, "prompt_finetuning_type")
        assert hasattr(llamacli.prompts, "prompt_training_params")
        assert hasattr(llamacli.prompts, "detect_template")
        assert hasattr(llamacli.prompts, "_list_cached_models")
        assert hasattr(llamacli.prompts, "_list_datasets")

    def test_runner_imports(self):
        import llamacli.runner
        assert hasattr(llamacli.runner, "run_training")
        assert hasattr(llamacli.runner, "run_export")

    def test_hf_imports(self):
        import llamacli.hf
        assert hasattr(llamacli.hf, "search_models")
        assert hasattr(llamacli.hf, "download_model")
        assert hasattr(llamacli.hf, "download_model_interactive")

    def test_logo_imports(self):
        import llamacli.logo
        assert hasattr(llamacli.logo, "print_logo")
        assert hasattr(llamacli.logo, "get_logo_text")

    def test_state_imports(self):
        import llamacli.state
        assert hasattr(llamacli.state, "get_state")
        assert hasattr(llamacli.state, "AppState")
        assert hasattr(llamacli.state, "reload_state")

    def test_all_screen_functions_load(self):
        from llamacli.cli import (
            interactive_loop,
            quick_train,
            advanced_train,
            chat_trained,
            quick_chat,
            view_models_screen,
            view_datasets_screen,
            view_demo_datasets_screen,
            add_dataset_screen,
            export_screen,
            show_main_menu,
        )

    def test_new_commands_exist(self):
        """Verify all new subcommands are registered in the Typer app."""
        from llamacli.cli import app
        registered = set()
        for cmd in app.registered_commands:
            name = cmd.name or cmd.callback.__name__
            # Map ad dataset back to its CLI name 'add'
            if name == "add_dataset":
                name = "add"
            registered.add(name)
        expected = {"setup", "train", "chat", "export", "download", "list", "add", "info", "doctor", "update", "clean"}
        for cmd in expected:
            assert cmd in registered, f"'{cmd}' not registered in Typer app"


class TestSmoke:
    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "llamacli" in result.stdout

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "LLaMA-Factory Interactive CLI" in result.stdout
        assert "train" in result.stdout

    def test_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "train", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--dataset" in result.stdout
        assert "--epochs" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--resume" in result.stdout
        assert "--method" in result.stdout
        assert "--grad-accum" in result.stdout
        assert "--warmup" in result.stdout
        assert "--scheduler" in result.stdout
        assert "--force" in result.stdout
        assert "--push-to-hub" in result.stdout

    def test_chat_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "chat", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--message" in result.stdout
        assert "--max-tokens" in result.stdout

    def test_export_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "export", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--adapter" in result.stdout
        assert "--model" in result.stdout
        assert "--output" in result.stdout

    def test_download_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "download", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "model" in result.stdout
        assert "dataset" in result.stdout
        assert "--no-confirm" in result.stdout

    def test_list_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "list", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--json" in result.stdout

    def test_info_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "info", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--json" in result.stdout

    def test_doctor_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "doctor", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--fix" in result.stdout

    def test_update_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "update", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--check" in result.stdout

    def test_clean_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "clean", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "configs" in result.stdout
        assert "--force" in result.stdout

    def test_main_help_has_all_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "llamacli", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        commands = ["train", "chat", "export", "download", "list", "info", "doctor", "update", "clean", "setup"]
        for cmd in commands:
            assert cmd in result.stdout, f"'{cmd}' not found in main --help"
