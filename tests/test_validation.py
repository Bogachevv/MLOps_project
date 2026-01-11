from omegaconf import OmegaConf

import dialogue_summarization.validation as validation


class FakeDataset:
    def __init__(self, prompts, summaries):
        self._prompts = list(prompts)
        self._summaries = list(summaries)

    def __len__(self):
        return len(self._prompts)

    def select(self, indices):
        return FakeDataset(
            [self._prompts[i] for i in indices],
            [self._summaries[i] for i in indices],
        )

    def __getitem__(self, key):
        if key == "prompt_text":
            return self._prompts
        if key == "summary":
            return self._summaries
        raise KeyError(key)


def _make_model(mocker):
    model = mocker.Mock()
    model.generation_config = mocker.Mock()
    model.generate = mocker.Mock()
    return model


def test_validate_limits_samples_and_computes_rouge(mocker):
    cfg = OmegaConf.create(
        {
            "validation": {
                "validation_samples": 1,
                "batch_size": 2,
                "max_new_tokens": 5,
            },
            "model": {}
        }
    )
    dataset = FakeDataset(["p1", "p2"], ["s1", "s2"])
    model = _make_model(mocker)
    tokenizer = mocker.Mock()

    called = {}

    def fake_pipeline(task, model, tokenizer):
        called["task"] = task
        called["model"] = model
        called["tokenizer"] = tokenizer

        def _call(prompts, return_full_text, max_new_tokens, batch_size):
            called["prompts"] = prompts
            called["return_full_text"] = return_full_text
            called["max_new_tokens"] = max_new_tokens
            called["batch_size"] = batch_size
            return [[{"generated_text": "assistant\n\npred"}] for _ in prompts]

        return _call

    mocker.patch.object(validation, "pipeline", fake_pipeline)
    rouge_mock = mocker.patch.object(
        validation.metrics,
        "compute_rouge",
        return_value={"rouge1": 0.1},
    )

    validation.validate(cfg, model, tokenizer, dataset)

    assert called["task"] == "text-generation"
    assert called["model"] is model
    assert called["tokenizer"] is tokenizer
    assert called["prompts"] == ["p1"]
    assert called["return_full_text"] is False
    assert called["max_new_tokens"] == 5
    assert called["batch_size"] == 2
    rouge_mock.assert_called_once_with(["pred"], ["s1"])


def test_validate_defaults_use_full_dataset(mocker):
    cfg = OmegaConf.create({
        "validation": {"validation_samples": None},
        "model": {}
    })


    dataset = FakeDataset(["p1", "p2"], ["s1", "s2"])
    model = _make_model(mocker)
    tokenizer = mocker.Mock()

    called = {}

    def fake_pipeline(task, model, tokenizer):
        def _call(prompts, return_full_text, max_new_tokens, batch_size):
            called["prompts"] = prompts
            called["return_full_text"] = return_full_text
            called["max_new_tokens"] = max_new_tokens
            called["batch_size"] = batch_size
            return [[{"generated_text": "pred"}] for _ in prompts]

        return _call

    mocker.patch.object(validation, "pipeline", fake_pipeline)
    rouge_mock = mocker.patch.object(validation.metrics, "compute_rouge", return_value={})

    validation.validate(cfg, model, tokenizer, dataset)

    assert called["prompts"] == ["p1", "p2"]
    assert called["return_full_text"] is False
    assert called["max_new_tokens"] == 128
    assert called["batch_size"] == 1
    rouge_mock.assert_called_once_with(["pred", "pred"], ["s1", "s2"])


def test_main_calls_loaders_and_validate(mocker):
    cfg = OmegaConf.create({"validation": {"validation_samples": 1}})
    model = _make_model(mocker)
    tokenizer = mocker.Mock()
    dataset = {"validation": "val_ds"}

    load_from_config = mocker.patch.object(
        validation.load_model,
        "load_from_config",
        return_value=(model, tokenizer),
    )
    load_dataset = mocker.patch.object(
        validation.load_data,
        "load_dataset_from_config",
        return_value=dataset,
    )
    validate = mocker.patch.object(validation, "validate")

    validation.main.__wrapped__(cfg)

    load_from_config.assert_called_once_with(cfg)
    load_dataset.assert_called_once_with(cfg)
    validate.assert_called_once_with(cfg, model, tokenizer, "val_ds")
