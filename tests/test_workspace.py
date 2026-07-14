import json
import os
import tempfile



class TestSyncDemoDatasets:
    def test_is_noop(self):
        """sync_demo_datasets copies bundled demo datasets into workspace data dir on first run."""
        from phronis.workspace import sync_demo_datasets

        with tempfile.TemporaryDirectory() as tmp:
            bundled_dir = os.path.join(tmp, "bundled")
            os.makedirs(bundled_dir)
            data_dir = os.path.join(tmp, "workspace_data")
            os.makedirs(data_dir)
            bundled_info = os.path.join(bundled_dir, "dataset_info.json")
            workspace_info = os.path.join(data_dir, "dataset_info.json")

            with open(bundled_info, "w", encoding="utf-8") as f:
                json.dump(
                    {"identity": {"file_name": "identity.json", "formatting": "alpaca"}},
                    f,
                )
            with open(
                os.path.join(bundled_dir, "identity.json"), "w", encoding="utf-8"
            ) as f:
                json.dump([{"instruction": "hi", "output": "hello"}], f)

            sync_demo_datasets(bundled_dir, bundled_info, data_dir, workspace_info)

            # Copies bundled datasets into workspace on first run so they appear in UI
            assert os.path.isfile(os.path.join(data_dir, "identity.json"))

    def test_skips_when_bundled_dir_missing(self):
        from phronis.workspace import sync_demo_datasets

        with tempfile.TemporaryDirectory() as tmp:
            sync_demo_datasets(
                os.path.join(tmp, "nonexistent"),
                os.path.join(tmp, "nonexistent.json"),
                tmp,
                os.path.join(tmp, "info.json"),
            )
            assert not os.path.isfile(os.path.join(tmp, "info.json"))
