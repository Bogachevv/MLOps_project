from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import sys

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import dialogue_summarization.process_dataset as process_dataset
import dialogue_summarization.train as train


def _compose_config(config_name: str):
    config_dir = PROJECT_ROOT / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    return cfg


def test_train_main_smoke_calls_heavy_dependencies(monkeypatch):
    cfg = _compose_config("train")

    model = MagicMock()
    tokenizer = MagicMock()
    trainer = MagicMock()

    load_from_config = MagicMock(return_value=(model, tokenizer))
    load_dataset_from_config = MagicMock(
        return_value={"train": "train_ds", "validation": "val_ds"}
    )
    get_trainer = MagicMock(return_value=trainer)

    monkeypatch.setattr(train.load_model, "load_from_config", load_from_config)
    monkeypatch.setattr(train.load_data, "load_dataset_from_config", load_dataset_from_config)
    monkeypatch.setattr(train, "get_trainer", get_trainer)

    train.main.__wrapped__(cfg)

    load_from_config.assert_called_once_with(cfg)
    load_dataset_from_config.assert_called_once_with(cfg)
    get_trainer.assert_called_once_with(cfg, model, tokenizer, "train_ds", "val_ds")
    trainer.train.assert_called_once_with()
    model.save_pretrained.assert_called_once_with(
        cfg.peft_config.pretrained_path, max_shard_size=cfg.get("save_shard_size", "5GB")
    )
    tokenizer.save_pretrained.assert_called_once_with(cfg.peft_config.pretrained_path)


def test_process_dataset_main_smoke_calls(monkeypatch):
    cfg = _compose_config("process_data")

    tokenizer = MagicMock()
    dataset = {"train": [{"text": "a"}], "validation": [{"text": "b"}]}

    load_tokenizer = MagicMock(return_value=tokenizer)
    process_by_config = MagicMock(return_value=dataset)

    monkeypatch.setattr(process_dataset.load_model, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(process_dataset, "process_by_config", process_by_config)

    process_dataset.main.__wrapped__(cfg)

    load_tokenizer.assert_called_once_with(cfg)
    process_by_config.assert_called_once_with(cfg, tokenizer)
