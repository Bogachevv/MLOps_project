import types

import pytest
from omegaconf import OmegaConf

import dialogue_summarization.load_model as load_model


def test_get_dtype_valid_values():
    assert load_model._get_dtype("float16") is load_model.torch.float16
    assert load_model._get_dtype("fp16") is load_model.torch.float16
    assert load_model._get_dtype("bfloat16") is load_model.torch.bfloat16
    assert load_model._get_dtype("bf16") is load_model.torch.bfloat16
    assert load_model._get_dtype("float32") is load_model.torch.float32
    assert load_model._get_dtype("fp32") is load_model.torch.float32


def test_get_dtype_invalid_value():
    with pytest.raises(KeyError):
        load_model._get_dtype("int8")


def test_load_tokenizer_sets_pad_token(monkeypatch):
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>")
    called = {}

    def fake_from_pretrained(model_name):
        called["model_name"] = model_name
        return tokenizer

    monkeypatch.setattr(load_model.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    cfg = OmegaConf.create({"model": {"model": "fake-model"}})
    result = load_model.load_tokenizer(cfg)

    assert called["model_name"] == "fake-model"
    assert result.pad_token == result.eos_token


def test_get_peft_new_builds_lora_config(monkeypatch):
    recorded = {}

    def fake_get_peft_model(model, adapter_config):
        recorded["model"] = model
        recorded["adapter_config"] = adapter_config
        return types.SimpleNamespace(print_trainable_parameters=lambda: None)

    monkeypatch.setattr(load_model, "get_peft_model", fake_get_peft_model)

    peft_config = OmegaConf.create(
        {
            "ft_strategy": "LoRA",
            "peft_is_trainable": True,
            "LoRA_config": {
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": ["q_proj"],
            },
        }
    )

    model = object()
    load_model._get_peft_new(peft_config, model)

    adapter_config = recorded["adapter_config"]
    assert recorded["model"] is model
    assert isinstance(adapter_config, load_model.LoraConfig)
    assert adapter_config.task_type == load_model.TaskType.CAUSAL_LM
    assert adapter_config.inference_mode is False
    assert adapter_config.r == 4
    assert adapter_config.lora_alpha == 8
    assert adapter_config.lora_dropout == 0.1
    assert adapter_config.bias == "none"
    assert adapter_config.target_modules == {"q_proj", }


def test_get_peft_pretrained_calls_from_pretrained(monkeypatch):
    recorded = {}

    def fake_from_pretrained(*, model, model_id, is_trainable):
        recorded["model"] = model
        recorded["model_id"] = model_id
        recorded["is_trainable"] = is_trainable
        return "peft-model"

    monkeypatch.setattr(load_model.PeftModel, "from_pretrained", fake_from_pretrained)

    peft_config = OmegaConf.create(
        {
            "pretrained_path": "adapter-path",
            "peft_is_trainable": True,
        }
    )
    model = object()

    result = load_model._get_peft_pretrained(peft_config, model)

    assert result == "peft-model"
    assert recorded == {
        "model": model,
        "model_id": "adapter-path",
        "is_trainable": True,
    }


def test_load_model_uses_device_map_and_dtype(monkeypatch):
    recorded = {}

    def fake_from_pretrained(model_name, torch_dtype, device_map):
        recorded["model_name"] = model_name
        recorded["torch_dtype"] = torch_dtype
        recorded["device_map"] = device_map
        return "model"

    monkeypatch.setattr(load_model.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    cfg = OmegaConf.create(
        {
            "model": {"model": "fake-model"},
            "dtype": "fp16",
            "peft_config": None,
        }
    )

    result = load_model.load_model(cfg)

    assert result == "model"
    assert recorded == {
        "model_name": "fake-model",
        "torch_dtype": load_model.torch.float16,
        "device_map": "auto",
    }
