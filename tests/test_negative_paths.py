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


class TestNegativePaths:
    def test_cancel_at_confirm_skips_training_and_yaml(self, temp_workspace, mock_console, monkeypatch):
        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: ("m", "t"))
        monkeypatch.setattr(cli_mod, "prompt_dataset", lambda c, **kw: "ds")
        monkeypatch.setattr(cli_mod, "prompt_target_loss", lambda c, **kw: None)
        monkeypatch.setattr(mock_console, "input", lambda prompt: "3")
        monkeypatch.setattr(cli_mod.questionary, "confirm", _fake_confirm_no)

        run_training_called = [False]

        def _track(c, p, o, **kw):
            run_training_called[0] = True
            return True

        monkeypatch.setattr(cli_mod, "run_training", _track)

        cli_mod.quick_train(mock_console)
        assert not run_training_called[0]
        configs_dir = os.path.join(temp_workspace, "configs")
        assert not os.listdir(configs_dir)

    def test_failed_run_does_not_record_history(self, temp_workspace, mock_console, monkeypatch):
        import llamacli.state as state_mod

        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: ("m", "t"))
        monkeypatch.setattr(cli_mod, "prompt_dataset", lambda c, **kw: "ds")
        monkeypatch.setattr(cli_mod, "prompt_target_loss", lambda c, **kw: None)
        monkeypatch.setattr(mock_console, "input", lambda prompt: "3")
        monkeypatch.setattr(cli_mod.questionary, "confirm", _fake_confirm_yes)
        monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: False)

        # Ensure a clean slate in case a previous test left history in the real state file
        state = state_mod.get_state()
        state.training_history = []
        state.save()
        assert len(state_mod.get_state().training_history) == 0

        cli_mod.quick_train(mock_console)
        state2 = state_mod.get_state()
        assert len(state2.training_history) == 0

        # YAML is still saved because _write_config_and_train runs before training
        configs_dir = os.path.join(temp_workspace, "configs")
        assert os.listdir(configs_dir)

    def test_cancel_at_model_prompt_aborts(self, temp_workspace, mock_console, monkeypatch):
        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: (None, None))
        run_training_called = [False]
        monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: run_training_called.__setitem__(0, True) or True)
        cli_mod.quick_train(mock_console)
        assert not run_training_called[0]
        text = mock_console.file.getvalue()
        assert "Cancelled" in text

    def test_cancel_at_dataset_prompt_aborts(self, temp_workspace, mock_console, monkeypatch):
        monkeypatch.setattr(cli_mod, "prompt_model", lambda c, **kw: ("m", "t"))
        monkeypatch.setattr(cli_mod, "prompt_dataset", lambda c, **kw: None)
        run_training_called = [False]
        monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: run_training_called.__setitem__(0, True) or True)
        cli_mod.quick_train(mock_console)
        assert not run_training_called[0]
        text = mock_console.file.getvalue()
        assert "Cancelled" in text
