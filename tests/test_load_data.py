import sys
from pathlib import Path

from omegaconf import OmegaConf
from datasets import DatasetDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from dialogue_summarization import load_data


def test_load_dataset_from_config_defaults_keep_in_memory(monkeypatch):
    captured = {}

    def fake_load_from_disk(path, keep_in_memory):
        captured["path"] = path
        captured["keep_in_memory"] = keep_in_memory
        return "dataset"

    monkeypatch.setattr(DatasetDict, "load_from_disk", staticmethod(fake_load_from_disk))

    cfg = OmegaConf.create({"dataset": {"save_path": "data/path"}})

    result = load_data.load_dataset_from_config(cfg)

    assert result == "dataset"
    assert captured == {"path": "data/path", "keep_in_memory": False}


def test_load_dataset_from_config_respects_keep_in_memory(monkeypatch):
    captured = {}

    def fake_load_from_disk(path, keep_in_memory):
        captured["path"] = path
        captured["keep_in_memory"] = keep_in_memory
        return "dataset"

    monkeypatch.setattr(DatasetDict, "load_from_disk", staticmethod(fake_load_from_disk))

    cfg = OmegaConf.create({"dataset": {"save_path": "data/path", "dataset_in_memory": True}})

    result = load_data.load_dataset_from_config(cfg)

    assert result == "dataset"
    assert captured == {"path": "data/path", "keep_in_memory": True}
