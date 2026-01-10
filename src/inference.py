import load_model
from transformers import pipeline

import numpy as np
import numexpr

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('eval', lambda expr: numexpr.evaluate(expr).item(), replace=True)


def prepare_prompt(text: str, tokenizer):
    messages = [
        {"role": "user", "content": text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt")

    return model_inputs


def generate(model_inputs, model, tokenizer) -> str:
    model_inputs = model_inputs.to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    return content


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model.load_from_config(cfg)

    line = input(">> ")
    while line:
        model_inputs = prepare_prompt(line, tokenizer)
        content = generate(model_inputs, model, tokenizer)

        print("Model answer:", content)
        line = input(">> ")

    print(model)


if __name__ == '__main__':
    main()
