import time
from typing import Iterable, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
import numexpr

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('eval', lambda expr: numexpr.evaluate(expr).item(), replace=True)

from peft import PeftModel

from dialogue_summarization import load_model, load_data

from dotenv import load_dotenv
load_dotenv()

def _extract_reference(example: dict) -> Optional[str]:
    if "summary" in example:
        return str(example["summary"]).strip()

    text = example.get("text")
    prompt_text = example.get("prompt_text")
    if isinstance(text, str) and isinstance(prompt_text, str) and text.startswith(prompt_text):
        return text[len(prompt_text):].strip()

    return None


def _extract_prompt(example: dict) -> str:
    if "prompt_text" in example:
        return str(example["prompt_text"])
    if "text" in example:
        return str(example["text"])
    raise KeyError("Expected 'prompt_text' or 'text' in validation dataset.")


def _chunked_indices(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def validate(cfg: DictConfig, model: Union[AutoModelForCausalLM, PeftModel], tokenizer: AutoTokenizer, val_dataset):
    validation_cfg = cfg.validation
    
    max_samples = validation_cfg.validation_samples
    batch_size = int(validation_cfg.get("batch_size", 1))
    max_new_tokens = int(validation_cfg.get("max_new_tokens", 128))

    if max_samples:
        max_samples = min(int(max_samples), len(val_dataset))
        val_dataset = val_dataset.select(range(max_samples))

    prompts = []
    refs = []
    for example in val_dataset:
        prompts.append(_extract_prompt(example))
        refs.append(_extract_reference(example))

    valid_pairs = [(p, r) for p, r in zip(prompts, refs, strict=False) if r]
    if not valid_pairs:
        print("WARNING: No reference summaries found; metrics will be skipped.")

    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    total_time = 0.0
    total_generated_tokens = 0
    predictions = []

    for start, end in _chunked_indices(len(prompts), batch_size):
        batch_prompts = prompts[start:end]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_time = time.perf_counter() - start_time
        total_time += batch_time

        for i in range(generated_ids.size(0)):
            prompt_len = int(batch_inputs["attention_mask"][i].sum().item())
            output_ids = generated_ids[i][prompt_len:]
            total_generated_tokens += int(output_ids.numel())
            predictions.append(tokenizer.decode(output_ids, skip_special_tokens=True))

    if valid_pairs:
        refs_filtered = [r for _, r in valid_pairs]
        preds_filtered = [predictions[i] for i, r in enumerate(refs) if r]
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(
            predictions=preds_filtered,
            references=refs_filtered,
            use_aggregator=True,
            use_stemmer=True,
        )
        print("ROUGE scores:")
        for key, value in rouge_scores.items():
            print(f"  {key}: {value:.4f}")

    if predictions:
        avg_time = total_time / len(predictions)
        print(f"Avg generation time per sample: {avg_time:.4f}s")
        if total_generated_tokens > 0:
            print(f"Generated tokens/sec: {total_generated_tokens / total_time:.2f}")


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
