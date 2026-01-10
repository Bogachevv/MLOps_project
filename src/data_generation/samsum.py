import transformers
import datasets
from datasets import DatasetDict

from omegaconf import DictConfig


def _load_dataset(cfg: DictConfig) -> DatasetDict:
    ds_name = cfg.dataset.ds_name

    ds = datasets.load_dataset(ds_name)
    return ds


def _get_prompt(path: str) -> str:
    with open(path, 'r') as f:
        prompt = f.read().strip()
    
    return prompt


def _build_texts(example, tokenizer: transformers.AutoTokenizer, sys_prompt: str, usr_prompt: str) -> dict[str, str]:
    dialogue = example['dialogue'].strip()
    summary = example['summary'].strip()
    
    user_prompt = usr_prompt.format(dialogue=dialogue)

    prompt_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    full_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": summary},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {
        "text": text,
        "prompt_text": prompt_text,
    }


def _process(cfg: DictConfig, dataset: DatasetDict, tokenizer: transformers.AutoTokenizer) -> DatasetDict:
    sys_prompt = _get_prompt(cfg.dataset.sys_prompt_pth)
    usr_prompt = _get_prompt(cfg.dataset.usr_prompt_pth)

    batch_size = cfg.dataset.get('preprocessing_bs')
    num_proc = cfg.dataset.get('preprocessing_nproc')

    remove_cols = dataset["train"].column_names
    dataset = dataset.map(
        _build_texts,
        fn_kwargs={'tokenizer': tokenizer, 'sys_prompt': sys_prompt, 'usr_prompt': usr_prompt},
        remove_columns=remove_cols
    )

    return dataset


def _save(cfg: DictConfig, dataset: DatasetDict) -> None:
    save_path = cfg.dataset.save_path

    dataset.save_to_disk(save_path)


def process_by_config(cfg: DictConfig, tokenizer: transformers.AutoTokenizer) -> DatasetDict:
    dataset = _load_dataset(cfg)
    dataset = _process(cfg, dataset, tokenizer)
    _save(cfg, dataset)

    return dataset
