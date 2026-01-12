import shutil
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dialogue_summarization import load_model


def export_for_torchserve(cfg: DictConfig) -> Path:
    output_dir = Path(cfg.torchserve.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model.load_from_config(cfg)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(model.state_dict(), output_dir / "model.pt")

    sys_prompt = Path(cfg.dataset.sys_prompt_pth)
    usr_prompt = Path(cfg.dataset.usr_prompt_pth)
    if sys_prompt.exists():
        shutil.copy2(sys_prompt, output_dir / sys_prompt.name)
    if usr_prompt.exists():
        shutil.copy2(usr_prompt, output_dir / usr_prompt.name)

    model_config = Path(cfg.torchserve.model_config_path)
    if model_config.exists():
        shutil.copy2(model_config, output_dir / model_config.name)

    return output_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="torchserve_export")
def main(cfg: DictConfig):
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    output_dir = export_for_torchserve(cfg)
    print(f"TorchServe artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
