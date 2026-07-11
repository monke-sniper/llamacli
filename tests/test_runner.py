import os
import yaml

import pytest

from llamacli import CONFIGS_DIR, PROJECT_ROOT


class TestRunner:
    def test_find_cli_works(self):
        from llamacli.runner import _find_cli
        cli = _find_cli()
        assert os.path.isfile(cli) or cli is not None

    def test_can_generate_valid_yaml_config(self):
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
            "max_samples": 1000,
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

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["model_name_or_path"] == "Qwen/Qwen3-0.6B"
            assert loaded["stage"] == "sft"
            assert loaded["finetuning_type"] == "lora"
            assert loaded["lora_rank"] == 8
            assert loaded["num_train_epochs"] == 3.0
            assert loaded["learning_rate"] == 1e-4
        finally:
            os.unlink(temp_path)


class TestExportConfig:
    def test_export_config_valid(self):
        config = {
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "adapter_name_or_path": "saves/qwen3-0.6b-identity/lora",
            "template": "qwen3",
            "finetuning_type": "lora",
            "export_dir": "models/qwen3-0.6b-identity",
            "export_size": 2,
            "export_legacy_format": False,
        }
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            temp_path = f.name
        try:
            with open(temp_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["export_dir"] == "models/qwen3-0.6b-identity"
            assert loaded["export_size"] == 2
        finally:
            os.unlink(temp_path)


class TestMetricsParsing:
    def test_parse_valid_metrics_line(self):
        from llamacli.runner import _parse_metrics
        line = "{'loss': '3.35', 'grad_norm': '4.318', 'learning_rate': '9.619e-05', 'epoch': '0.87'}"
        result = _parse_metrics(line)
        assert result is not None
        assert result["loss"] == "3.35"
        assert result["grad_norm"] == "4.318"
        assert result["learning_rate"] == "9.619e-05"
        assert result["epoch"] == "0.87"

    def test_parse_metrics_with_extra_keys(self):
        from llamacli.runner import _parse_metrics
        line = "{'loss': '2.41', 'grad_norm': '2.25', 'learning_rate': '5.975e-05', 'epoch': '1.70', 'total_flos': '9999GF'}"
        result = _parse_metrics(line)
        assert result is not None
        assert result["loss"] == "2.41"
        assert "total_flos" not in result

    def test_parse_non_metrics_line(self):
        from llamacli.runner import _parse_metrics
        line = "[INFO] Loading dataset identity.json..."
        result = _parse_metrics(line)
        assert result is None

    def test_parse_line_with_metrics_embedded(self):
        from llamacli.runner import _parse_metrics
        line = "INFO {'loss': '1.50', 'grad_norm': '1.0', 'learning_rate': '1e-06', 'epoch': '3.0'} done"
        result = _parse_metrics(line)
        assert result is not None
        assert result["loss"] == "1.50"

    def test_format_metrics_bar_empty(self):
        from llamacli.runner import _format_metrics
        bar = _format_metrics({}, (0, 0), 10)
        assert "10s" in bar

    def test_format_metrics_bar_full(self):
        from llamacli.runner import _format_metrics
        metrics = {"loss": "3.35", "grad_norm": "4.32", "learning_rate": "9.6e-05", "epoch": "0.87"}
        bar = _format_metrics(metrics, (5, 18), 30)
        assert "Step [5/18]" in bar
        assert "Loss 3.35" in bar
        assert "Epoch 0.87" in bar
        assert "GNorm 4.32" in bar
        assert "LR 9.6e-05" in bar
        assert "30s" in bar

    def test_format_metrics_no_steps(self):
        from llamacli.runner import _format_metrics
        metrics = {"loss": "2.41"}
        bar = _format_metrics(metrics, (3, 0), 15)
        assert "Step 3" in bar
        assert "Loss 2.41" in bar
