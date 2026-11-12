"""Train a ReCap estimator or ACP policy with PyTorch."""

import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.training import config as train_config
from openpi.training.recap_pytorch import init_wandb, run


def main(config, *, output_dir=None, worker_setup=None):
    run(config, output_dir=output_dir, worker_setup=worker_setup)


if __name__ == "__main__":
    config = train_config.cli()
    stage = config.recap_training_stage
    if stage is None:
        stage = {"Pi0ValueConfig": "value", "Pi0StageProgressConfig": "stage_progress"}.get(type(config.model).__name__)
        if stage is None and config.name.startswith("pi05_acp_"):
            stage = "offline_rl"
    main(dataclasses.replace(config, recap_training_stage=stage, ema_decay=None), output_dir=config.checkpoint_dir)
