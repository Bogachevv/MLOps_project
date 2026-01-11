import transformers
import datasets
from datasets import DatasetDict

import dialogue_summarization.load_model as load_model

import numpy as np
import numexpr

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('eval', lambda expr: numexpr.evaluate(expr).item(), replace=True)


def process_by_config(cfg: DictConfig, tokenizer: transformers.AutoTokenizer) -> DatasetDict:
    ds_name = cfg.dataset.ds_name

    if ds_name == 'knkarthick/samsum':
        from dialogue_summarization.data_generation import samsum
        return samsum.process_by_config(cfg, tokenizer)
    

    raise AttributeError(f"Unknown dataset name: {ds_name}")


@hydra.main(version_base=None, config_path="../../configs", config_name="process_data")
def main(cfg: DictConfig):
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    tokenizer = load_model.load_tokenizer(cfg)
    dataset = process_by_config(cfg, tokenizer)

    print(dataset)

    for split in dataset.keys():
        print(f"{split=}\texample={dataset[split][0]}")


if __name__ == '__main__':
    main()
