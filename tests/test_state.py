import json
import os
import shutil
import tempfile

import yaml

import pytest

from llamacli import DEFAULT_CONFIG, STATE_PATH
from llamacli.state import AppState, get_state


class TestAppState:
    def test_default_values(self):
        state = AppState()
        assert state.active_model == ""
        assert state.active_dataset == ""
        assert state.active_adapter == ""
        assert state.active_template == "qwen3"
        assert state.theme == "dark"
        assert state.training_history == []

    def test_load_defaults_when_no_config(self):
        import llamacli.state as state_mod
        old_path = state_mod.STATE_PATH
        old_state = state_mod._state

        temp_dir = tempfile.mkdtemp()
        state_mod.STATE_PATH = os.path.join(temp_dir, "nonexistent.yaml")
        state_mod._state = None

        try:
            state = AppState.load()
            assert isinstance(state, AppState)
            assert state.active_model == ""
            assert state.active_dataset == ""
            assert state.active_template == "qwen3"
        finally:
            state_mod.STATE_PATH = old_path
            state_mod._state = old_state
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_save_and_load(self):
        import llamacli.state as state_mod
        old_path = state_mod.STATE_PATH
        old_state = state_mod._state

        fd, temp_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        state_mod.STATE_PATH = temp_path
        state_mod._state = None

        try:
            state = AppState()
            state.active_model = "test/model"
            state.active_dataset = "test_dataset"
            state.active_template = "llama3"
            state.save()

            with open(temp_path, "r") as f:
                data = yaml.safe_load(f)
            assert data["active_model"] == "test/model"
            assert data["active_dataset"] == "test_dataset"
            assert data["active_template"] == "llama3"
        finally:
            state_mod.STATE_PATH = old_path
            state_mod._state = old_state
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_training_history_append(self):
        state = AppState()
        assert len(state.training_history) == 0
        state.training_history.append({"name": "run1", "loss": 2.5, "epochs": 3, "timestamp": "2026-01-01"})
        state.training_history.append({"name": "run2", "loss": 1.8, "epochs": 5, "timestamp": "2026-02-01"})
        assert len(state.training_history) == 2
        assert state.training_history[0]["name"] == "run1"

    def test_merge_config_on_load(self):
        fd, temp_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(temp_path, "w") as f:
            yaml.dump({"active_model": "override/model"}, f)

        import llamacli.state as state_mod
        old_path = state_mod.STATE_PATH
        old_state = state_mod._state
        state_mod.STATE_PATH = temp_path
        state_mod._state = None

        try:
            state = AppState.load()
            assert state.active_model == "override/model"
            assert state.active_template == "qwen3"
            assert state.active_dataset == ""
        finally:
            state_mod.STATE_PATH = old_path
            state_mod._state = old_state
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_reload_state(self):
        import llamacli.state as state_mod
        old_path = state_mod.STATE_PATH
        old_state = state_mod._state

        fd, temp_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        state_mod.STATE_PATH = temp_path
        state_mod._state = None

        try:
            s1 = state_mod.get_state()
            s1.active_model = "reload/test"
            s1.save()

            s2 = state_mod.reload_state()
            assert s2.active_model == "reload/test"
        finally:
            state_mod.STATE_PATH = old_path
            state_mod._state = old_state
            if os.path.exists(temp_path):
                os.unlink(temp_path)
