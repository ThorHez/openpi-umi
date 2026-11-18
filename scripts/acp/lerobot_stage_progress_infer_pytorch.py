"""Use a trained Kai0 stage-progress checkpoint to annotate one LeRobot dataset."""

import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
import runpy

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from openpi.training import recap_pytorch


init_wandb = recap_pytorch.init_wandb


def _stage_progress_checkpoint(path):
    checkpoint = Path(path).resolve(strict=True)
    if checkpoint.name == "model.safetensors" and checkpoint.is_file():
        checkpoint = checkpoint.parent
    marker = checkpoint / "recap_checkpoint.json"
    if not (checkpoint / "model.safetensors").is_file() or not marker.is_file():
        raise ValueError("Kai0 标注需要已训练的进度 checkpoint，目录须包含 model.safetensors 和 recap_checkpoint.json")
    try:
        with marker.open("rb") as stream:
            contents = stream.read(64 * 1024 + 1)
        if len(contents) > 64 * 1024:
            raise ValueError("checkpoint receipt is too large")
        receipt = json.loads(contents)
    except (OSError, ValueError) as exc:
        raise ValueError("无法读取 Kai0 阶段进度 checkpoint 的 recap_checkpoint.json") from exc
    if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1 or receipt.get("backend") != "pytorch"
            or receipt.get("stage") != "stage_progress" or type(receipt.get("step")) is not int
            or receipt["step"] < 0):
        raise ValueError("所选 checkpoint 不是有效的 PyTorch Kai0 阶段进度模型")
    return checkpoint


def main(config, *, output_dir=None, worker_setup=None, device=None):
    if config.recap_training_stage != "stage_annotation" or config.pytorch_weight_path is None:
        raise ValueError("Kai0 标注需要 stage_annotation 配置和已训练的进度模型权重")
    if hasattr(config.data, "datasets") or config.batch_size < 1:
        raise ValueError("Kai0 标注需要单个 LeRobot 数据集和正数 batch")
    checkpoint = _stage_progress_checkpoint(config.pytorch_weight_path)

    from openpi.training import data_loader, stage_progress, stage_progress_labels

    if worker_setup is not None:
        config = worker_setup(config)
    device = recap_pytorch._device(0) if device is None else torch.device(device)
    torch.manual_seed(config.seed)
    source = stage_progress.StageProgressSource(config, skip_norm_stats=True)
    model = recap_pytorch.load_scoring_model(config, checkpoint, "stage_progress", device)
    collate = functools.partial(data_loader._collate_fn, image_keys=config.model.image_keys,
                                depth_keys=config.model.depth_keys)

    @torch.inference_mode()
    def predict(reference, current):
        samples = {int(row): source.observation(row) for row in np.union1d(reference, current)}
        ref = recap_pytorch.torch_observation(collate([samples[int(row)] for row in reference]), device)
        cur = recap_pytorch.torch_observation(collate([samples[int(row)] for row in current]), device)
        return model.predict_progress(ref, cur).detach().float().cpu().reshape(-1).numpy()

    output = Path(config.checkpoint_base_dir).parent if output_dir is None else Path(output_dir)
    return stage_progress_labels.annotate_dataset(config, source, predict, output)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config-name")
    source.add_argument("--config-file", type=Path, help="可信的导出训练配置 training_config.py")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="在该目录下生成 stage_advantage_dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-step", type=int)
    parser.add_argument("--positive-ratio", type=float)
    parser.add_argument("--stage-count", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.config_file:
        path = args.config_file.resolve(strict=True)
        config = runpy.run_path(str(path))["build_config"](path.parent)
    else:
        from openpi.training import config as train_config

        config = train_config.get_config(args.config_name)
    if hasattr(config.data, "datasets"):
        parser.error("Kai0 标注需要单数据集进度配置")
    settings = {key: getattr(args, key) for key in ("n_step", "positive_ratio", "stage_count")
                if getattr(args, key) is not None}
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=str(args.dataset_root.resolve(strict=True))),
                                 model=dataclasses.replace(config.model, **settings), batch_size=args.batch_size,
                                 pytorch_weight_path=str(args.checkpoint_dir), recap_training_stage="stage_annotation")
    print(main(config, output_dir=args.output_dir, device=args.device))


if __name__ == "__main__":
    cli()
