import io
import os
import tempfile

from rich.console import Console


def _dummy_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


class TestDemoDatasetHelpers:
    def test_list_demo_datasets_returns_list(self):
        from phronis.prompts import _list_demo_datasets
        datasets = _list_demo_datasets()
        assert isinstance(datasets, list)
        names = [d["name"] for d in datasets]
        assert "identity" in names
        assert "demo_chat" in names

    def test_demo_datasets_have_required_keys(self):
        from phronis.prompts import _list_demo_datasets
        datasets = _list_demo_datasets()
        assert all("name" in d for d in datasets)
        assert all("format" in d for d in datasets)
        assert all("source" in d for d in datasets)
        assert all(d["source"] == "demo" for d in datasets)

    def test_count_demo_dataset_returns_int(self):
        from phronis.prompts import _count_demo_dataset, _list_demo_datasets
        datasets = _list_demo_datasets()
        assert datasets
        cnt = _count_demo_dataset(datasets[0]["name"])
        assert isinstance(cnt, int)
        assert cnt > 0

    def test_count_nonexistent_demo_dataset(self):
        from phronis.prompts import _count_demo_dataset
        assert _count_demo_dataset("__missing__") == 0


class TestPromptDatasetFallback:
    def test_falls_back_to_demo_when_workspace_empty(self, monkeypatch):
        import phronis
        import phronis.prompts as prompts_mod

        with tempfile.TemporaryDirectory() as tmp:
            old_data = phronis.DATA_DIR
            old_dsi = phronis.DATASET_INFO
            pd_old_data = prompts_mod.DATA_DIR
            pd_old_dsi = prompts_mod.DATASET_INFO

            phronis.DATA_DIR = tmp
            phronis.DATASET_INFO = os.path.join(tmp, "nonexistent.json")
            prompts_mod.DATA_DIR = tmp
            prompts_mod.DATASET_INFO = os.path.join(tmp, "nonexistent.json")

            demo_sets = prompts_mod._list_demo_datasets()
            assert demo_sets, "Bundled demo datasets must exist"

            class _FakeCheckbox:
                def __init__(self, *a, **k):
                    pass

                @staticmethod
                def ask():
                    return [demo_sets[0]["name"]]

            monkeypatch.setattr(prompts_mod.questionary, "checkbox", lambda *a, **k: _FakeCheckbox(*a, **k))

            try:
                console = _dummy_console()
                result = prompts_mod.prompt_dataset(console)
                assert result == demo_sets[0]["name"]
                text = console.file.getvalue()
                assert "No personal datasets found" in text
            finally:
                phronis.DATA_DIR = old_data
                phronis.DATASET_INFO = old_dsi
                prompts_mod.DATA_DIR = pd_old_data
                prompts_mod.DATASET_INFO = pd_old_dsi


class TestDemoDatasetJsonlCount:
    def test_count_jsonl_demo_dataset(self):
        import tempfile
        import phronis.prompts as prompts_mod
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = prompts_mod.BUNDLED_DATA_DIR
            old_dsi = prompts_mod.BUNDLED_DATASET_INFO
            try:
                prompts_mod.BUNDLED_DATA_DIR = tmp
                prompts_mod.BUNDLED_DATASET_INFO = os.path.join(tmp, "dataset_info.json")
                with open(os.path.join(tmp, "chat.jsonl"), "w", encoding="utf-8") as f:
                    f.write('{"messages":[{"role":"user","content":"hi"}]}\n')
                    f.write('{"messages":[{"role":"user","content":"hello"}]}\n')
                    f.write('\n')
                cnt = prompts_mod._count_demo_dataset("chat")
                assert cnt == 2
            finally:
                prompts_mod.BUNDLED_DATA_DIR = old_dir
                prompts_mod.BUNDLED_DATASET_INFO = old_dsi
