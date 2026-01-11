import re

import datasets
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

import transformers

import dialogue_summarization.data_generation.samsum as samsum
import dialogue_summarization.inference as inference
import dialogue_summarization.train as train


class FakeBatch(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


def test_pipeline_integration(tmp_path, mocker):
    sys_prompt_path = tmp_path / "sys_prompt.txt"
    usr_prompt_path = tmp_path / "usr_prompt.txt"
    sys_prompt_path.write_text("SYSTEM")
    usr_prompt_path.write_text("Summarize: {dialogue}")

    save_path = tmp_path / "dataset"
    cfg = OmegaConf.create(
        {
            "dataset": {
                "ds_name": "dummy",
                "sys_prompt_pth": str(sys_prompt_path),
                "usr_prompt_pth": str(usr_prompt_path),
                "save_path": str(save_path),
            },
            "trainer_config": {"output_dir": str(tmp_path / "out")},
            "model": {"response_template": "assistant"},
        }
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"dialogue": ["Hello"], "summary": ["Hi there"]}
            )
        }
    )

    load_dataset = mocker.patch("datasets.load_dataset", return_value=dataset)
    loaded_dataset = samsum._load_dataset(cfg)
    load_dataset.assert_called_once_with("dummy")

    tokenizer = mocker.Mock(spec=transformers.AutoTokenizer)

    def fake_apply_chat_template(messages, tokenize, add_generation_prompt):
        parts = [f"{msg['role']}: {msg['content']}" for msg in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return " | ".join(parts)

    # tokenizer.apply_chat_template.side_effect = fake_apply_chat_template
    tokenizer.apply_chat_template = mocker.Mock(side_effect=fake_apply_chat_template)

    processed = samsum._process(cfg, loaded_dataset, tokenizer)
    assert processed["train"][0]["text"] == (
        "system: SYSTEM | user: Summarize: Hello | assistant: Hi there"
    )
    assert processed["train"][0]["prompt_text"] == (
        "system: SYSTEM | user: Summarize: Hello | assistant:"
    )

    save_to_disk = mocker.patch.object(processed, "save_to_disk")
    samsum._save(cfg, processed)
    save_to_disk.assert_called_once_with(str(save_path))
    assert not re.match(r"^[A-Za-z]:\\", str(save_path))

    fake_batch = FakeBatch(torch.tensor([[1, 2]]))

    def fake_tokenizer_call(texts, return_tensors):
        assert texts == ["system: SYSTEM | user: Summarize: Hello | assistant:"]
        assert return_tensors == "pt"
        return fake_batch

    # tokenizer.__call__.side_effect = fake_tokenizer_call
    # tokenizer.__call__ = mocker.Mock(side_effect=fake_tokenizer_call)
    tokenizer.side_effect = fake_tokenizer_call

    model_inputs = inference.prepare_prompt(
        "Hello", tokenizer, "SYSTEM", "Summarize: {dialogue}"
    )
    assert model_inputs is fake_batch

    model = mocker.Mock(spec=transformers.AutoModelForCausalLM)

    collator = mocker.patch.object(
        train, "DataCollatorForCompletionOnlyLM", return_value="collator"
    )
    sft_trainer = mocker.patch.object(train, "SFTTrainer", return_value="trainer")
    trainer = train.get_trainer(
        cfg,
        model,
        tokenizer,
        processed["train"],
        processed["train"],
    )

    collator.assert_called_once_with("assistant", tokenizer=tokenizer)
    sft_trainer.assert_called_once()
    assert trainer == "trainer"

    model.device = "cpu"
    model.generate = mocker.Mock(return_value=torch.tensor([[1, 2, 3, 4]]))
    tokenizer.decode = mocker.Mock(return_value="decoded")

    output = inference.generate(fake_batch, model, tokenizer)

    model.generate.assert_called_once_with(**fake_batch, max_new_tokens=128)
    tokenizer.decode.assert_called_once_with([3, 4], skip_special_tokens=True)
    assert output == "decoded"
