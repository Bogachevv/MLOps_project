import dialogue_summarization.load_model as load_model

import transformers
from transformers import pipeline

import numpy as np
import numexpr

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('eval', lambda expr: numexpr.evaluate(expr).item(), replace=True)

import sys

from dotenv import load_dotenv
load_dotenv()


def _get_prompt(path: str) -> str:
    with open(path, 'r') as f:
        prompt = f.read().strip()
    
    return prompt


def prepare_prompt(text: str, tokenizer: transformers.AutoTokenizer, sys_prompt: str, usr_prompt: str):
    user_prompt = usr_prompt.format(dialogue=text)
    prompt_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([prompt_text], return_tensors="pt")

    return model_inputs


def generate(model_inputs, model, tokenizer) -> str:
    model_inputs = model_inputs.to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    return content


@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def main(cfg: DictConfig):
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model.load_from_config(cfg)

    sys_prompt = _get_prompt(cfg.dataset.sys_prompt_pth)
    usr_prompt = _get_prompt(cfg.dataset.usr_prompt_pth)

    print(f"{sys_prompt=}\n\n{usr_prompt=}")

    print(f">> ")
    text = sys.stdin.read().strip()

    model_inputs = prepare_prompt(text, tokenizer, sys_prompt, usr_prompt)
    content = generate(model_inputs, model, tokenizer)

    print("Model answer:", content)


if __name__ == '__main__':
    main()
