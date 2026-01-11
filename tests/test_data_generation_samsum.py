import datasets
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from dialogue_summarization.data_generation import samsum


def _make_dataset_dict() -> DatasetDict:
    data = {
        "dialogue": ["Hi there.", "How are you?"],
        "summary": ["Greeting.", "Asking wellbeing."],
        "extra": ["x", "y"],
    }
    return DatasetDict({"train": Dataset.from_dict(data)})


def test_load_dataset_uses_datasets_load_dataset(mocker):
    dataset = _make_dataset_dict()
    load_mock = mocker.patch.object(datasets, "load_dataset", return_value=dataset)

    cfg = OmegaConf.create({"dataset": {"ds_name": "samsum"}})

    result = samsum._load_dataset(cfg)

    assert result is dataset
    load_mock.assert_called_once_with("samsum")


def test_get_prompt_reads_and_strips(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  Hello world\n")

    result = samsum._get_prompt(str(prompt_file))

    assert result == "Hello world"


def test_build_texts_returns_expected_fields_and_texts(mocker):
    tokenizer = mocker.Mock()
    tokenizer.apply_chat_template.side_effect = ["PROMPT", "FULL"]

    example = {"dialogue": "  Hi ", "summary": " Bye \n"}
    result = samsum._build_texts(example, tokenizer, "SYS", "USR {dialogue}")

    assert result == {"text": "FULL", "prompt_text": "PROMPT"}
    tokenizer.apply_chat_template.assert_has_calls(
        [
            mocker.call(
                [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "USR Hi"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
            mocker.call(
                [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "USR Hi"},
                    {"role": "assistant", "content": "Bye"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            ),
        ]
    )


def test_process_removes_redundant_columns_and_adds_texts(tmp_path, mocker):
    dataset = _make_dataset_dict()
    sys_prompt = tmp_path / "sys.txt"
    usr_prompt = tmp_path / "usr.txt"
    sys_prompt.write_text("SYS")
    usr_prompt.write_text("USR {dialogue}")

    tokenizer = mocker.Mock()
    tokenizer.apply_chat_template.side_effect = ["PROMPT", "FULL"] * 2

    cfg = OmegaConf.create(
        {
            "dataset": {
                "sys_prompt_pth": str(sys_prompt),
                "usr_prompt_pth": str(usr_prompt),
            }
        }
    )

    result = samsum._process(cfg, dataset, tokenizer)

    assert set(result["train"].column_names) == {"dialogue", "summary", "text", "prompt_text"}, f"{result['train'].column_names=}"
    assert result["train"][0]["text"] == "FULL"
    assert result["train"][0]["prompt_text"] == "PROMPT"


def test_save_calls_save_to_disk(mocker):
    dataset = mocker.Mock()
    cfg = OmegaConf.create({"dataset": {"save_path": "output/path"}})

    samsum._save(cfg, dataset)

    dataset.save_to_disk.assert_called_once_with("output/path")


def test_process_by_config_calls_load_process_save_in_order(mocker):
    dataset = mocker.Mock()
    processed = mocker.Mock()
    tokenizer = mocker.Mock()
    cfg = OmegaConf.create({"dataset": {"ds_name": "samsum"}})

    load_mock = mocker.patch.object(samsum, "_load_dataset", return_value=dataset)
    process_mock = mocker.patch.object(samsum, "_process", return_value=processed)
    save_mock = mocker.patch.object(samsum, "_save")
    result = samsum.process_by_config(cfg, tokenizer)

    assert result is processed
    assert load_mock.call_count == 1
    assert process_mock.call_count == 1
    assert save_mock.call_count == 1
    assert load_mock.mock_calls[0] == mocker.call(cfg)
    assert process_mock.mock_calls[0] == mocker.call(cfg, dataset, tokenizer)
    assert save_mock.mock_calls[0] == mocker.call(cfg, processed)
