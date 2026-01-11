import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import evaluate

import numpy as np
import numexpr

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('eval', lambda expr: numexpr.evaluate(expr).item(), replace=True)

import peft
from peft import PeftModel

from dialogue_summarization import load_model, load_data
import metrics

from typing import Union, Optional

from dotenv import load_dotenv
load_dotenv()


@torch.inference_mode()
def validate(cfg: DictConfig, model: Union[AutoModelForCausalLM, PeftModel], tokenizer: AutoTokenizer, val_dataset):
    model.eval()

    validation_cfg = cfg.validation

    max_samples = validation_cfg.validation_samples 
    batch_size = int(validation_cfg.get("batch_size", 1)) 
    max_new_tokens = int(validation_cfg.get("max_new_tokens", 128))

    if max_samples: 
        max_samples = min(int(max_samples), len(val_dataset)) 
        val_dataset = val_dataset.select(range(max_samples))

    pl = pipeline("text-generation", model=model, tokenizer=tokenizer)

    generations = pl(
        list(val_dataset['prompt_text']),
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size
    )
    predictions = [
        gen[0]['generated_text'].replace('assistant\n\n', '')
        for gen in generations
    ]

    rouge = metrics.compute_rouge(predictions, list(val_dataset['summary']))
    print(f"\n{rouge=}")


@hydra.main(version_base=None, config_path="../../configs", config_name="validation")
def main(cfg: DictConfig):
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model.load_from_config(cfg)

    dataset = load_data.load_dataset_from_config(cfg)
    val_dataset = dataset['validation']

    validate(cfg, model, tokenizer, val_dataset)


if __name__ == '__main__':
    main()
