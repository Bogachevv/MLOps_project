import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import peft
from peft import get_peft_model, PeftConfig, LoraConfig, PeftModel, TaskType

from omegaconf import DictConfig, OmegaConf


def _get_dtype(dtype: str) -> torch.dtype:
    _DTYPE_MAP = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    return _DTYPE_MAP[dtype]


def _load_tokenizer(cfg: DictConfig):
    model_name = cfg.model.model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def _get_peft_new(peft_config: DictConfig, model: AutoModelForCausalLM) -> PeftModel:
    print("New LoRA")

    ft_strategy = peft_config.ft_strategy
    is_trainable = peft_config.get('peft_is_trainable', True)

    if ft_strategy == 'LoRA':
        adapter_config = OmegaConf.to_container(peft_config.LoRA_config, resolve=True)

        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not is_trainable, 
            **adapter_config,
        )

        print(f"{adapter_config=}")
    else:
        raise ValueError('Incorrect FT type')

    model_adapter = get_peft_model(model, adapter_config)
    model_adapter.print_trainable_parameters()

    print(f"{model_adapter=}")

    return model_adapter


def _get_peft_pretrained(peft_config: DictConfig, model: AutoModelForCausalLM) -> PeftModel:
    print("Pretrained LoRA")

    adapter_pth = peft_config.pretrained_path
    is_trainable = peft_config.get('peft_is_trainable', False)

    model_adapter = PeftModel.from_pretrained(
        model=model,
        model_id=adapter_pth,
        is_trainable=is_trainable,
    )

    return model_adapter


def _get_peft(config: DictConfig, model: AutoModelForCausalLM) -> PeftModel:
    peft_config = config.peft_config

    if peft_config.get('pretrained_path'):
        return _get_peft_pretrained(peft_config, model)
    else:
        return _get_peft_new(peft_config, model)


def _load_model(cfg: DictConfig):
    model_name = str(cfg.model.model)

    torch_dtype = _get_dtype(cfg.dtype)
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if cfg.get("peft_config"):
        print("Loading PEFT")
        model = _get_peft(cfg, model)
    else:
        print("Running without PEFT")


    return model


def load_from_config(cfg: DictConfig) -> tuple[AutoModelForCausalLM | PeftModel, AutoTokenizer]:
    model = _load_model(cfg)
    tokenizer = _load_tokenizer(cfg)

    return model, tokenizer
