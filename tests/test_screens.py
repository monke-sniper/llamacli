import os

import pytest

import llamacli.cli as cli_mod


def _fake_confirm_yes(*a, **k):
    class _Q:
        @staticmethod
        def ask():
            return True
    return _Q()


def _fake_confirm_no(*a, **k):
    class _Q:
        @staticmethod
        def ask():
            return False
    return _Q()


class TestQuickTrainScreen:
    def test_runs_and_creates_yaml(self, temp_workspace, mock_console, monkeypatch):
        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: ("Qwen/Qwen3-0.6B", "qwen3"))
        monkeypatch.setattr(cli_mod, "prompt_dataset", lambda c, **kw: "identity")
        monkeypatch.setattr(cli_mod, "prompt_target_loss", lambda c, **kw: None)
        monkeypatch.setattr(mock_console, "input", lambda prompt: "3")
        monkeypatch.setattr(cli_mod.questionary, "confirm", _fake_confirm_yes)
        monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, target_loss=None: True)

        cli_mod.quick_train(mock_console)

        configs_dir = os.path.join(temp_workspace, "configs")
        files = os.listdir(configs_dir)
        assert any(f.startswith("llamacli_run_") and not f.endswith("_summary.yaml") for f in files)

        import llamacli.state as state_mod
        state = state_mod.get_state()
        assert len(state.training_history) >= 1
        assert state.training_history[-1]["model"] == "Qwen/Qwen3-0.6B"
        assert state.training_history[-1]["dataset"] == "identity"

    def test_shows_config_table(self, temp_workspace, mock_console, monkeypatch):
        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: ("Qwen/Qwen3-0.6B", "qwen3"))
        monkeypatch.setattr(cli_mod, "prompt_dataset", lambda c, **kw: "identity")
        monkeypatch.setattr(cli_mod, "prompt_target_loss", lambda c, **kw: None)
        monkeypatch.setattr(mock_console, "input", lambda prompt: "3")
        monkeypatch.setattr(cli_mod.questionary, "confirm", _fake_confirm_no)

        cli_mod.quick_train(mock_console)
        text = mock_console.file.getvalue()
        assert "Quick Train Configuration" in text
        assert "model_name_or_path" in text
        assert "Qwen/Qwen3-0.6B" in text


class TestAdvancedTrainScreen:
    def test_runs_and_creates_yaml(self, temp_workspace, mock_console, monkeypatch):
        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: ("Qwen/Qwen3-0.6B", "qwen3"))
        monkeypatch.setattr(cli_mod, "prompt_dataset", lambda c, **kw: "identity")
        monkeypatch.setattr(cli_mod, "prompt_stage", lambda c, **kw: "sft")
        monkeypatch.setattr(cli_mod, "prompt_finetuning_type", lambda c, **kw: "lora")
        monkeypatch.setattr(
            cli_mod, "prompt_training_params",
            lambda c, ft, **kw: {
                "epochs": 2,
                "learning_rate": 5e-5,
                "batch_size": 1,
                "grad_accum": 8,
                "cutoff_len": 256,
                "warmup_ratio": 0.1,
            },
        )
        monkeypatch.setattr(cli_mod, "prompt_target_loss", lambda c, **kw: None)
        monkeypatch.setattr(mock_console, "input", lambda prompt: "adv_run")
        monkeypatch.setattr(cli_mod.questionary, "confirm", _fake_confirm_yes)
        monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, target_loss=None: True)

        cli_mod.advanced_train(mock_console)

        configs_dir = os.path.join(temp_workspace, "configs")
        files = os.listdir(configs_dir)
        assert "llamacli_adv_run.yaml" in files

        import llamacli.state as state_mod
        state = state_mod.get_state()
        assert len(state.training_history) >= 1
        assert state.training_history[-1]["stage"] == "sft"


class TestViewDatasetsScreen:
    def test_shows_personal_datasets(self, temp_workspace, mock_console, monkeypatch):
        data_dir = os.path.join(temp_workspace, "data")
        ds_path = os.path.join(data_dir, "my_data.json")
        import json
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump([{"instruction": "hi", "output": "hello"}], f)

        monkeypatch.setattr(mock_console, "input", lambda prompt: "")
        cli_mod.view_datasets_screen(mock_console)
        text = mock_console.file.getvalue()
        assert "my_data" in text
        assert "1" in text

    def test_can_set_active_dataset(self, temp_workspace, mock_console, monkeypatch):
        data_dir = os.path.join(temp_workspace, "data")
        ds_path = os.path.join(data_dir, "active_ds.json")
        import json
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump([{"instruction": "a", "output": "b"}], f)

        monkeypatch.setattr(mock_console, "input", lambda prompt: "1")
        cli_mod.view_datasets_screen(mock_console)

        import llamacli.state as state_mod
        state = state_mod.get_state()
        assert state.active_dataset == "active_ds"
