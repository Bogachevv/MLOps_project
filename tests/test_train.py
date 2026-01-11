from omegaconf import OmegaConf

import dialogue_summarization.train as train


def _make_cfg(extra_peft=None):
    peft_config = {"pretrained_path": "./artifacts"}
    if extra_peft:
        peft_config.update(extra_peft)
    return OmegaConf.create(
        {
            "trainer_config": {"learning_rate": 0.0001},
            "model": {"response_template": "### Response"},
            "peft_config": peft_config,
        }
    )


def test_get_trainer_passes_expected_args(mocker):
    cfg = _make_cfg()
    model = mocker.Mock()
    tokenizer = mocker.Mock()
    train_dataset = object()
    val_dataset = object()

    training_args = mocker.Mock(name="training_args")
    collator = mocker.Mock(name="collator")
    trainer_instance = mocker.Mock(name="trainer_instance")

    sft_config_mock = mocker.Mock(return_value=training_args)
    collator_mock = mocker.Mock(return_value=collator)
    trainer_mock = mocker.Mock(return_value=trainer_instance)

    mocker.patch.object(train, "SFTConfig", sft_config_mock)
    mocker.patch.object(train, "DataCollatorForCompletionOnlyLM", collator_mock)
    mocker.patch.object(train, "SFTTrainer", trainer_mock)

    result = train.get_trainer(cfg, model, tokenizer, train_dataset, val_dataset)

    assert result is trainer_instance
    sft_config_mock.assert_called_once_with(learning_rate=0.0001)
    collator_mock.assert_called_once_with(
        "### Response",
        tokenizer=tokenizer,
    )
    trainer_mock.assert_called_once_with(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )


def test_train_merge_adapters_true(mocker):
    cfg = _make_cfg({"merge_tuned": True})
    model = mocker.Mock()
    tokenizer = mocker.Mock()
    train_dataset = object()
    val_dataset = object()

    trainer = mocker.Mock()
    merged_model = mocker.Mock()

    mocker.patch.object(train, "get_trainer", return_value=trainer)
    model.merge_and_unload.return_value = merged_model

    train.train(cfg, model, tokenizer, train_dataset, val_dataset)

    trainer.train.assert_called_once_with()
    model.merge_and_unload.assert_called_once_with(progressbar=True, safe_merge=True)
    merged_model.save_pretrained.assert_called_once_with(
        "./artifacts",
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained.assert_called_once_with("./artifacts")


def test_train_merge_adapters_false(mocker):
    cfg = _make_cfg({"merge_tuned": False})
    model = mocker.Mock()
    tokenizer = mocker.Mock()
    train_dataset = object()
    val_dataset = object()

    trainer = mocker.Mock()
    mocker.patch.object(train, "get_trainer", return_value=trainer)

    train.train(cfg, model, tokenizer, train_dataset, val_dataset)

    trainer.train.assert_called_once_with()
    model.merge_and_unload.assert_not_called()
    model.save_pretrained.assert_called_once_with(
        "./artifacts",
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained.assert_called_once_with("./artifacts")
