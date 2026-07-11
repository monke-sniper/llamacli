import os
import tempfile
import yaml

import pytest


class TestTrainingWizardConfig:
    def test_full_config_roundtrip(self):
        config = {
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "trust_remote_code": True,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": 8,
            "lora_dropout": 0.05,
            "lora_target": "all",
            "dataset": "identity",
            "template": "qwen3",
            "cutoff_len": 512,
            "max_samples": 10000,
            "preprocessing_num_workers": 8,
            "output_dir": "saves/test_run/lora",
            "logging_steps": 5,
            "save_steps": 100,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "report_to": "none",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "num_train_epochs": 3.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            config_path = f.name

        try:
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["model_name_or_path"] == "Qwen/Qwen3-0.6B"
            assert loaded["stage"] == "sft"
            assert loaded["finetuning_type"] == "lora"
            assert loaded["lora_rank"] == 8
            assert loaded["lora_dropout"] == 0.05
            assert loaded["dataset"] == "identity"
            assert loaded["template"] == "qwen3"
            assert loaded["num_train_epochs"] == 3.0
            assert loaded["learning_rate"] == 1e-4
            assert loaded["per_device_train_batch_size"] == 2
            assert loaded["gradient_accumulation_steps"] == 8
            assert loaded["cutoff_len"] == 512
            assert loaded["bf16"] is True
            assert loaded["output_dir"] == "saves/test_run/lora"
        finally:
            os.unlink(config_path)

    def test_dpo_config_roundtrip(self):
        config = {
            "model_name_or_path": "test/model",
            "trust_remote_code": True,
            "stage": "dpo",
            "do_train": True,
            "finetuning_type": "full",
            "dataset": "dpo_en_demo",
            "template": "qwen3",
            "cutoff_len": 1024,
            "max_samples": 500,
            "preprocessing_num_workers": 4,
            "output_dir": "saves/dpo_run/lora",
            "logging_steps": 10,
            "save_steps": 200,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "report_to": "none",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 5e-5,
            "num_train_epochs": 1.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": True,
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            config_path = f.name

        try:
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["stage"] == "dpo"
            assert loaded["finetuning_type"] == "full"
            assert loaded["dataset"] == "dpo_en_demo"
            assert loaded["num_train_epochs"] == 1.0
            assert loaded["learning_rate"] == 5e-5
        finally:
            os.unlink(config_path)

    def test_freeze_config(self):
        config = {
            "model_name_or_path": "test/model",
            "stage": "sft",
            "finetuning_type": "freeze",
            "dataset": "identity",
            "template": "qwen3",
            "cutoff_len": 512,
            "output_dir": "saves/freeze_run/lora",
            "num_train_epochs": 3.0,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "bf16": True,
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            config_path = f.name
        try:
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["finetuning_type"] == "freeze"
            assert "lora_rank" not in loaded
        finally:
            os.unlink(config_path)


class TestStatePersistence:
    def test_state_survives_roundtrip(self):
        import tempfile
        import llamacli.state as state_mod

        old_path = state_mod.STATE_PATH
        old_state = state_mod._state

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            state_mod.STATE_PATH = f.name

        try:
            state_mod._state = None
            s1 = state_mod.get_state()
            s1.active_model = "test/model"
            s1.active_dataset = "ds1"
            s1.save()

            state_mod._state = None
            s2 = state_mod.get_state()
            assert s2.active_model == "test/model"
            assert s2.active_dataset == "ds1"
        finally:
            state_mod.STATE_PATH = old_path
            state_mod._state = None
            if os.path.exists(f.name):
                os.unlink(f.name)


class TestDirectoryIntegrity:
    def test_project_root_is_cwd(self):
        from llamacli import PROJECT_ROOT
        assert os.path.isdir(PROJECT_ROOT)

    def test_data_dir_exists_or_creatable(self):
        from llamacli import DATA_DIR
        os.makedirs(DATA_DIR, exist_ok=True)
        assert os.path.isdir(DATA_DIR)

    def test_saves_dir_exists_or_creatable(self):
        from llamacli import SAVES_DIR
        os.makedirs(SAVES_DIR, exist_ok=True)
        assert os.path.isdir(SAVES_DIR)
