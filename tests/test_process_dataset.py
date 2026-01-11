import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from dialogue_summarization import process_dataset


def test_process_by_config_uses_samsum(mocker):
    dataset = DatasetDict(
        {"train": Dataset.from_dict({"dialogue": ["Hi"], "summary": ["Bye"]})}
    )

    tokenizer = mocker.Mock()
    tokenizer.apply_chat_template.side_effect = ["PROMPT", "FULL"]
    cfg = OmegaConf.create(
        {
            "dataset": {
                "ds_name": "knkarthick/samsum",
                "sys_prompt_pth": "sys_prompt.txt",
                "usr_prompt_pth": "usr_prompt.txt",
            }
        }
    )

    load_mock = mocker.patch(
        "dialogue_summarization.data_generation.samsum._load_dataset",
        return_value=dataset,
    )
    
    get_prompt_mock = mocker.patch(
        "dialogue_summarization.data_generation.samsum._get_prompt",
        side_effect=["SYS", "USR {dialogue}"],
    )
    save_mock = mocker.patch("dialogue_summarization.data_generation.samsum._save")

    result = process_dataset.process_by_config(cfg, tokenizer)

    load_mock.assert_called_once_with(cfg)
    get_prompt_mock.assert_has_calls(
        [mocker.call("sys_prompt.txt"), mocker.call("usr_prompt.txt")]
    )
    save_mock.assert_called_once_with(cfg, result)
    assert result["train"][0]["prompt_text"] == "PROMPT"
    assert result["train"][0]["text"] == "FULL"


def test_process_by_config_unknown_dataset():
    cfg = OmegaConf.create({"dataset": {"ds_name": "unknown/ds"}})

    with pytest.raises(AttributeError, match=r"Unknown dataset name: unknown/ds"):
        process_dataset.process_by_config(cfg, tokenizer=object())
