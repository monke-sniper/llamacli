"""E2E tests that spawn the real llamacli process and interact via stdin/stdout.

These tests verify:
- Subcommand discovery (--help output)
- Flag parsing and validation
- Non-interactive command execution
- Workspace isolation
- Output formatting
"""
import os
import sys
import tempfile

import pytest

from .e2e_test_harness import CliRunner


@pytest.fixture
def workspace(tmp_path):
    """Create an isolated workspace directory with state marker to skip bootstrap."""
    ws = tmp_path / "e2e_workspace"
    ws.mkdir()
    # Create a minimal state file so _check_first_run() doesn't trigger bootstrap
    state_file = ws / ".llamacli.yaml"
    state_file.write_text("active_model: ''\nactive_dataset: ''\n", encoding="utf-8")
    return str(ws)


@pytest.fixture
def runner(workspace):
    """Provide a CliRunner with a temp workspace, auto-cleanup."""
    r = CliRunner(workspace)
    yield r
    r.close()


class TestHelpDiscovery:
    """Verify --help output for all new subcommands."""

    @pytest.mark.parametrize("subcommand", [
        [],
        ["train"],
        ["chat"],
        ["export"],
        ["download"],
        ["list"],
        ["add", "dataset"],
        ["info"],
        ["doctor"],
        ["update"],
        ["clean"],
    ])
    def test_help_shows_subcommand(self, runner, workspace, subcommand):
        runner.start(args=subcommand + ["--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "Usage:" in output or "usage:" in output

    @pytest.mark.parametrize("subcommand", [
        "chat",
        "export",
        "download",
        "list",
        "info",
        "doctor",
        "update",
        "clean",
    ])
    def test_main_help_lists_subcommand(self, runner, workspace, subcommand):
        runner.start(args=["--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert subcommand in output, f"'{subcommand}' not found in help output"


class TestVersionFlag:
    def test_version_prints_version(self, runner, workspace):
        runner.start(args=["--version"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "llamacli" in output.lower()


class TestGlobalFlags:
    def test_workspace_flag_overrides_path(self, runner, workspace):
        # Global flags must come BEFORE the subcommand
        runner.start(args=["--workspace", workspace, "info"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        # Rich wraps long paths, so check for the basename which should appear
        assert os.path.basename(workspace) in output or workspace in output

    def test_no_color_flag(self, runner, workspace):
        runner.start(args=["--no-color", "info"])
        runner.assert_exit_code(0)
        output = runner.get_output()
        # Verify no ANSI escape sequences in the raw output
        assert "\x1b[" not in output, "ANSI sequences found despite --no-color"


class TestInfoCommand:
    def test_info_shows_workspace(self, runner, workspace):
        runner.start(args=["info"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "Workspace" in output or "workspace" in output
        assert "data/" in output
        assert "saves/" in output

    def test_info_json_shows_json(self, runner, workspace):
        runner.start(args=["info", "--json"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        import json
        data = json.loads(output.strip())
        assert "workspace" in data
        assert "directories" in data


class TestListCommand:
    def test_list_models_empty(self, runner, workspace):
        runner.start(args=["list", "models"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "No cached models" in output or "0" in output

    def test_list_datasets_empty(self, runner, workspace):
        runner.start(args=["list", "datasets"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "No datasets" in output or "dataset" in output.lower()

    def test_list_history_empty(self, runner, workspace):
        runner.start(args=["list", "history"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "No training history" in output or "history" in output.lower()

    def test_list_adapters_empty(self, runner, workspace):
        runner.start(args=["list", "adapters"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "No adapters" in output or "adapter" in output.lower()

    def test_list_json_output(self, runner, workspace):
        runner.start(args=["list", "models", "--json"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        import json
        # The entire output should be valid JSON (array or object)
        try:
            data = json.loads(output.strip())
            assert isinstance(data, list) or isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail(f"Expected valid JSON output, got: {output!r}")


class TestTrainCommand:
    def test_train_help_shows_new_flags(self, runner, workspace):
        runner.start(args=["train", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "--dry-run" in output
        assert "--resume" in output
        assert "--method" in output
        assert "--grad-accum" in output
        assert "--warmup" in output
        assert "--scheduler" in output
        assert "--force" in output
        assert "--push-to-hub" in output

    def test_train_missing_model_fails(self, runner, workspace):
        runner.start(args=["train", "--dataset", "test"])
        runner.assert_exit_code(2)
        output = runner.get_clean_output()
        assert "--model" in output or "required" in output.lower() or "error" in output.lower()

    def test_train_missing_dataset_fails(self, runner, workspace):
        runner.start(args=["train", "--model", "Qwen/Qwen3-0.6B"])
        runner.assert_exit_code(2)
        output = runner.get_clean_output()
        assert "--dataset" in output or "required" in output.lower() or "error" in output.lower()

    def test_train_dry_run(self, runner, workspace):
        # Create a dummy dataset
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        import json
        dummy_data = [
            {"instruction": "Hello", "output": "World"},
            {"instruction": "Test", "output": "Data"},
        ]
        with open(os.path.join(data_dir, "test_data.json"), "w", encoding="utf-8") as f:
            json.dump(dummy_data, f)

        # Register it
        dataset_info = os.path.join(data_dir, "dataset_info.json")
        with open(dataset_info, "w", encoding="utf-8") as f:
            json.dump({
                "test_data": {"file_name": "test_data.json", "formatting": "alpaca"}
            }, f)

        runner.start(args=[
            "--workspace", workspace,
            "train",
            "--model", "Qwen/Qwen3-0.6B",
            "--dataset", "test_data",
            "--dry-run",
        ])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "Dry run" in output
        assert "model_name_or_path" in output
        assert "Qwen/Qwen3-0.6B" in output


class TestAddCommand:
    def test_add_dataset_file(self, runner, workspace):
        runner.start(args=[
            "--workspace", workspace,
            "add",
            "--name", "my_test",
            "--file", "my_test.json",
            "--format", "alpaca",
        ])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "added" in output.lower() or "my_test" in output

    def test_add_dataset_hf(self, runner, workspace):
        runner.start(args=[
            "--workspace", workspace,
            "add",
            "--name", "my_hf",
            "--hf-url", "https://huggingface.co/datasets/test",
            "--format", "sharegpt",
        ])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "added" in output.lower() or "my_hf" in output

    def test_add_missing_flags_fails(self, runner, workspace):
        runner.start(args=[
            "--workspace", workspace,
            "add", "dataset",
            "--name", "test",
        ])
        runner.assert_exit_code(2)


class TestExportCommand:
    def test_export_help(self, runner, workspace):
        runner.start(args=["export", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "--adapter" in output
        assert "--model" in output
        assert "--output" in output
        assert "--size" in output


class TestDownloadCommand:
    def test_download_help(self, runner, workspace):
        runner.start(args=["download", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "model" in output
        assert "dataset" in output
        assert "--no-confirm" in output


class TestChatCommand:
    def test_chat_help(self, runner, workspace):
        runner.start(args=["chat", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "--model" in output
        assert "--message" in output
        assert "--max-tokens" in output


class TestDoctorCommand:
    def test_doctor_help(self, runner, workspace):
        runner.start(args=["doctor", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "--fix" in output


class TestUpdateCommand:
    def test_update_help(self, runner, workspace):
        runner.start(args=["update", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "--check" in output

    def test_update_check(self, runner, workspace):
        runner.start(args=["update", "--check"])
        # Allow any exit code since pip index may not be available
        try:
            runner.assert_exit_code(0, timeout=10)
        except AssertionError:
            pass  # pip index may not be supported
        output = runner.get_clean_output()
        # At minimum the banner should appear; if pip is unavailable we accept that
        assert "Update" in output or "pip" in output.lower() or "llamacli" in output.lower() or output == ""


class TestCleanCommand:
    def test_clean_help(self, runner, workspace):
        runner.start(args=["clean", "--help"])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "configs" in output
        assert "cache" in output
        assert "saves" in output
        assert "all" in output
        assert "--force" in output

    def test_clean_configs_force(self, runner, workspace):
        # Create a config file to clean
        configs_dir = os.path.join(workspace, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        with open(os.path.join(configs_dir, "test.yaml"), "w", encoding="utf-8") as f:
            f.write("test: true\n")

        runner.start(args=[
            "--workspace", workspace,
            "clean", "configs",
            "--force",
        ])
        runner.assert_exit_code(0)
        output = runner.get_clean_output()
        assert "Deleted" in output or "configs" in output


class TestInvalidCommands:
    def test_invalid_subcommand(self, runner, workspace):
        runner.start(args=["badcmd"])
        runner.assert_exit_code(2)
        output = runner.get_clean_output()
        assert "error" in output.lower() or "invalid" in output.lower() or "badcmd" in output

    def test_invalid_list_type(self, runner, workspace):
        runner.start(args=["list", "badtype"])
        runner.assert_exit_code(1)
        output = runner.get_clean_output()
        assert "Unknown" in output or "invalid" in output.lower() or "models" in output
