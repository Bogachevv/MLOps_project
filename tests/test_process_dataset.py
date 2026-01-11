import sys
import types

import pytest
from omegaconf import OmegaConf

from dialogue_summarization import process_dataset


def _install_data_generation_module() -> None:
    data_generation = types.ModuleType("data_generation")
    samsum = types.ModuleType("data_generation.samsum")
    data_generation.samsum = samsum
    sys.modules["data_generation"] = data_generation
    sys.modules["data_generation.samsum"] = samsum


def test_process_by_config_uses_samsum(mocker):
    _install_data_generation_module()
    mock_process = mocker.patch(
        "data_generation.samsum.process_by_config",
        return_value={"train": "ok"},
        create=True,
    )
    cfg = OmegaConf.create({"dataset": {"ds_name": "knkarthick/samsum"}})
    tokenizer = mocker.Mock()

    result = process_dataset.process_by_config(cfg, tokenizer)

    mock_process.assert_called_once_with(cfg, tokenizer)
    assert result == {"train": "ok"}


def test_process_by_config_unknown_dataset():
    cfg = OmegaConf.create({"dataset": {"ds_name": "unknown/ds"}})

    with pytest.raises(AttributeError, match=r"Unknown dataset name: unknown/ds"):
        process_dataset.process_by_config(cfg, tokenizer=object())