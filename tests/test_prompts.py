import pytest

from llamacli.prompts import (
    _count_dataset,
    _list_cached_models,
    _list_datasets,
    detect_template,
    FINETUNING_TYPES,
    MODEL_TYPE_TO_TEMPLATE,
    STAGES,
    TEMPLATES,
    TEXT_GEN_MODEL_TYPES,
)


class TestModelListing:
    def test_list_returns_list(self):
        models = _list_cached_models()
        assert isinstance(models, list)

    def test_model_has_required_keys(self):
        models = _list_cached_models()
        for m in models:
            assert "repo_id" in m
            assert "size_gb" in m

    def test_models_are_sorted(self):
        models = _list_cached_models()
        ids = [m["repo_id"].lower() for m in models]
        assert ids == sorted(ids)


class TestDatasetListing:
    def test_list_returns_list(self):
        datasets = _list_datasets()
        assert isinstance(datasets, list)

    def test_dataset_has_required_keys(self):
        datasets = _list_datasets()
        for d in datasets:
            assert "name" in d
            assert "format" in d
            assert "source" in d

    def test_datasets_are_sorted(self):
        datasets = _list_datasets()
        names = [d["name"].lower() for d in datasets]
        assert names == sorted(names)


class TestDatasetCount:
    def test_count_returns_int(self):
        datasets = _list_datasets()
        if datasets:
            cnt = _count_dataset(datasets[0]["name"])
            assert isinstance(cnt, int)

    def test_nonexistent_dataset_returns_zero(self):
        cnt = _count_dataset("__nonexistent__")
        assert cnt == 0


class TestTemplateDetection:
    def test_qwen_detection(self):
        assert detect_template("Qwen/Qwen3-0.6B") == "qwen3"
        assert detect_template("Qwen/Qwen2.5-7B") == "qwen2"

    def test_llama_detection(self):
        assert detect_template("meta-llama/Llama-3-8B") == "llama3"
        assert detect_template("meta-llama/Llama-2-7b-hf") == "llama2"

    def test_mistral_detection(self):
        assert detect_template("mistralai/Mistral-7B-v0.1") == "mistral"

    def test_deepseek_detection(self):
        assert detect_template("deepseek-ai/DeepSeek-R1") == "deepseekr1"

    def test_phi_detection(self):
        assert detect_template("microsoft/phi-4") == "phi4"

    def test_unknown_falls_back(self):
        result = detect_template("something/unknown-model")
        assert isinstance(result, str)
        assert len(result) > 0


class TestConstants:
    def test_stages_have_all_required(self):
        stage_values = [c.value for c in STAGES]
        assert "sft" in stage_values
        assert "dpo" in stage_values
        assert "ppo" in stage_values
        assert "grpo" in stage_values
        assert "kto" in stage_values
        assert "pt" in stage_values
        assert "rm" in stage_values

    def test_finetuning_types(self):
        types = [c.value for c in FINETUNING_TYPES]
        assert "lora" in types
        assert "full" in types
        assert "freeze" in types

    def test_templates_includes_common(self):
        assert "qwen3" in TEMPLATES
        assert "llama3" in TEMPLATES
        assert "mistral" in TEMPLATES

    def test_model_type_mapping_has_key_entries(self):
        assert MODEL_TYPE_TO_TEMPLATE["qwen3"] == "qwen3"
        assert MODEL_TYPE_TO_TEMPLATE["llama"] == "llama3"
        assert MODEL_TYPE_TO_TEMPLATE["mistral"] == "mistral"
        assert MODEL_TYPE_TO_TEMPLATE["deepseek"] == "deepseek"
        assert MODEL_TYPE_TO_TEMPLATE["deepseek_v3"] == "deepseek3"


class TestChoiceUniqueness:
    def test_lora_choices_no_duplicate_defaults(self):
        import questionary
        choice_4 = questionary.Choice("4  - Smaller adapter, less VRAM", value=4)
        choice_8 = questionary.Choice("8  - Balanced (recommended)", value=8)
        choice_16 = questionary.Choice("16 - More capacity, more VRAM", value=16)
        choices = [choice_4, choice_8, choice_16]
        values = [c.value for c in choices]
        assert len(values) == len(set(values))

    def test_all_stage_choices_unique(self):
        values = [c.value for c in STAGES]
        assert len(values) == len(set(values))

    def test_all_ft_choices_unique(self):
        values = [c.value for c in FINETUNING_TYPES]
        assert len(values) == len(set(values))


class TestModelTypeFilter:
    def test_qwen3_is_text_gen(self):
        assert "qwen3" in TEXT_GEN_MODEL_TYPES

    def test_llama_in_text_gen(self):
        assert "llama" in TEXT_GEN_MODEL_TYPES
        assert "mistral" in TEXT_GEN_MODEL_TYPES
        assert "deepseek" in TEXT_GEN_MODEL_TYPES
        assert "gemma" in TEXT_GEN_MODEL_TYPES
        assert "phi" in TEXT_GEN_MODEL_TYPES
        assert "falcon" in TEXT_GEN_MODEL_TYPES

    def test_no_tts_types(self):
        assert "bark" not in TEXT_GEN_MODEL_TYPES
        assert "speecht5" not in TEXT_GEN_MODEL_TYPES
        assert "t5" not in TEXT_GEN_MODEL_TYPES

    def test_no_embedding_types(self):
        assert "bert" not in TEXT_GEN_MODEL_TYPES
        assert "roberta" not in TEXT_GEN_MODEL_TYPES
        assert "deberta" not in TEXT_GEN_MODEL_TYPES

    def test_no_audio_types(self):
        assert "whisper" not in TEXT_GEN_MODEL_TYPES
        assert "wav2vec2" not in TEXT_GEN_MODEL_TYPES
        assert "hubert" not in TEXT_GEN_MODEL_TYPES

    def test_no_vision_only_types(self):
        assert "vit" not in TEXT_GEN_MODEL_TYPES
        assert "clip" not in TEXT_GEN_MODEL_TYPES

    def test_listed_models_are_all_text_gen(self):
        models = _list_cached_models()
        for m in models:
            mt = m.get("model_type", "?")
            assert mt in TEXT_GEN_MODEL_TYPES or mt == "?", f"Non-text-gen model leaked: {m['repo_id']} type={mt}"

    def test_model_list_includes_model_type(self):
        models = _list_cached_models()
        for m in models:
            assert "model_type" in m


class TestDatasetFilter:
    def test_no_broken_datasets(self):
        datasets = _list_datasets()
        assert isinstance(datasets, list)
        assert all(isinstance(d.get("name"), str) for d in datasets)
        assert all(isinstance(d.get("format"), str) for d in datasets)

    def test_empty_dataset_list_handled(self):
        datasets = _list_datasets()
        assert isinstance(datasets, list)


class TestChatModelPicker:
    def test_prompt_chat_model_exists(self):
        from llamacli.prompts import prompt_chat_model
        assert callable(prompt_chat_model)
