import os
import shutil
import tempfile
from pathlib import Path

from transformers import TrainerCallback
from hydra.core.hydra_config import HydraConfig

from omegaconf import OmegaConf, DictConfig
from typing import Optional

class HydraOutputsToCometCallback(TrainerCallback):
    """
    Logs Hydra outputs (.hydra/{config,overrides,hydra}.yaml) to Comet as a single zip asset.
    """

    def __init__(self, asset_name: str = "hydra_outputs", base_dir: str = ".hydra", cfg: Optional[DictConfig] = None):
        self.asset_name = asset_name
        self.base_dir = base_dir
        self._logged = False
        self.cfg = cfg

    def on_train_begin(self, args, state, control, **kwargs):
        if self._logged:
            return control
        if not state.is_world_process_zero:
            return control

        try:
            import comet_ml
            from comet_ml import config as comet_config
        except Exception:
            # comet_ml not installed or not available; do nothing
            return control

        exp = comet_config.get_global_experiment()
        if exp is None:
            # CometCallback not active (e.g., report_to doesn't include comet_ml), or not initialized
            return control

        # Locate Hydra run dir and .hydra
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        hydra_dir = run_dir / self.base_dir

        if not hydra_dir.exists():
            # If you changed hydra.output_subdir or disabled it, adjust base_dir accordingly.
            return control

        # Zip .hydra and log as asset
        with tempfile.TemporaryDirectory() as tmp:
            archive_base = Path(tmp) / f"{self.asset_name}"
            zip_path = shutil.make_archive(
                base_name=str(archive_base),
                format="zip",
                root_dir=str(run_dir),
                base_dir=self.base_dir,
            )

            # Helpful naming: include hydra run folder name
            run_id = run_dir.name
            exp.log_asset(zip_path, file_name=f"{self.asset_name}__{run_id}.zip")

            # Also store the run directory path for quick navigation
            exp.log_parameter("hydra/output_dir", str(run_dir))

        # Log effective config as yaml
        if self.cfg:
            exp.log_text(OmegaConf.to_yaml(self.cfg), metadata={"kind": "hydra_effective_cfg"})

        self._logged = True
        return control
