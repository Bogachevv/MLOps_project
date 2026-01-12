from dotenv import load_dotenv
import os
load_dotenv()

import comet_ml

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import numpy as np
import numexpr

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('eval', lambda expr: numexpr.evaluate(expr).item(), replace=True)

import peft
from peft import PeftModel

from dialogue_summarization import load_model, load_data
from dialogue_summarization.loggers import log_hydra

from typing import Union, Optional


def get_trainer(cfg: DictConfig, model: Union[AutoModelForCausalLM, PeftModel], tokenizer: AutoTokenizer, train_dataset, val_dataset):
    model.train()

    training_args = SFTConfig(
        **OmegaConf.to_container(cfg.trainer_config, resolve=True),
    )
    
    response_template = cfg.model.response_template

    print(f"WARNING: Collator response_template is fixed to {response_template}")
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # report_to = cfg.get('report_to', 'wandb')

    return trainer


def push_to_hub(cfg: DictConfig, repo_id: str, model: Union[AutoModelForCausalLM, PeftModel], tokenizer: AutoTokenizer) -> None:
    print(f"Uploading to {repo_id}")

    token = os.environ.get('HF_SAVE_TOKEN')
    token = os.environ.get('HF_TOKEN') if token is None else token

    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)


def train(cfg: DictConfig, model: Union[AutoModelForCausalLM, PeftModel], tokenizer: AutoTokenizer, train_dataset, val_dataset):
    max_shard_size = cfg.get('save_shard_size', '5GB')
    print(f"Max shard size: {max_shard_size}")
    
    trainer = get_trainer(cfg, model, tokenizer, train_dataset, val_dataset)
    
    trainer.add_callback(
        log_hydra.HydraOutputsToCometCallback(
            asset_name="hydra_outputs",
            base_dir=".hydra",
            cfg=cfg
        )
    )
    
    trainer.train()

    merge_adapters = cfg.peft_config.get('merge_tuned', False)
    if merge_adapters:
        model = model.merge_and_unload(
            progressbar=True,
            safe_merge=True
        )

    model.save_pretrained(cfg.peft_config.pretrained_path, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(cfg.peft_config.pretrained_path)

    repo_id = cfg.get("hf_repo_id")
    if repo_id:
        push_to_hub(cfg, repo_id, model, tokenizer)


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model.load_from_config(cfg)

    dataset = load_data.load_dataset_from_config(cfg)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    train(cfg, model, tokenizer, train_dataset, val_dataset)


if __name__ == '__main__':
    main()
