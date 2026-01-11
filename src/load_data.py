from datasets import DatasetDict
from omegaconf import OmegaConf, DictConfig


__all__ = ['load_dataset_from_config']


def load_dataset_from_config(cfg: DictConfig) -> DatasetDict:
    dataset_path = cfg.dataset.save_path
    dataset_in_memory = cfg.dataset.get('dataset_in_memory', False)

    dataset = DatasetDict.load_from_disk(dataset_path, keep_in_memory=dataset_in_memory)

    return dataset
