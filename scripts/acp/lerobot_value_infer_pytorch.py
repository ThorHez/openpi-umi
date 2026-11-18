"""Use a PyTorch ReCap value checkpoint to publish a labeled LeRobot copy."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import runpy

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from openpi.training import recap_pytorch
from openpi.training.recap_advantages import label_value_advantages, load_existing_value_targets
from openpi.training.stage_progress_labels import publish_labeled_copy


init_wandb = recap_pytorch.init_wandb
DATASET_NAME = "value_advantage_dataset"


def _data_config(config, dataset_root, checkpoint, dataset_index, norm_asset_id):
    from openpi.training import config as train_config, data_loader

    factory = config.data
    multi = hasattr(factory, "datasets")
    children = factory.datasets if multi else [factory]
    if not 0 <= dataset_index < len(children):
        raise ValueError("数据处理配置索引超出范围")
    child = children[dataset_index]
    assets = checkpoint / "assets"
    if norm_asset_id is None:
        candidates = ("shared", f"dataset_{dataset_index}", child.assets.asset_id, ".")
        norm_asset_id = next((name for name in candidates if name and (assets / name / "norm_stats.json").is_file()), None)
    if norm_asset_id is None:
        raise ValueError("价值 checkpoint 未包含匹配的归一化统计，请指定 --norm-asset-id")
    stats = (assets / norm_asset_id / "norm_stats.json").resolve()
    if not stats.is_relative_to(assets.resolve()) or not stats.is_file():
        raise ValueError("归一化统计必须位于 checkpoint 的 assets 目录")
    bound = dataclasses.replace(child, repo_id=str(dataset_root), assets=train_config.AssetsConfig(
        assets_dir=str(assets), asset_id=norm_asset_id))
    data = bound.create(config.assets_dirs, config.model)
    return data_loader._apply_global_normalization(factory._apply_state_pad(data) if multi else data, config)


@torch.inference_mode()
def _predict_values(config, dataset, checkpoint, device):
    scorer = recap_pytorch.load_scoring_model(config, checkpoint, "value", device)
    predictions = []
    for batch in recap_pytorch.make_loader(dataset, config, batch_size=config.batch_size):
        observation = recap_pytorch.torch_observation(batch, device)
        predictions.append(scorer.predict_value(observation).detach().float().cpu().reshape(-1).numpy())
    if not predictions:
        raise ValueError("价值标注数据集为空")
    return np.concatenate(predictions)


def annotate(config, checkpoint, dataset_root, output, *, dataset_index=0, norm_asset_id=None,
             n_step=50, positive_ratio=0.3, device=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    dataset_root = Path(dataset_root).resolve(strict=True)
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("价值 checkpoint 目录必须包含 model.safetensors")
    if config.batch_size < 1 or n_step < 1 or not 0 < positive_ratio <= 1:
        raise ValueError("batch 和优势窗口必须为正数，正例比例必须在 (0, 1] 内")
    data = _data_config(config, dataset_root, checkpoint, dataset_index, norm_asset_id)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    return _annotate_bound(config, data, checkpoint, dataset_root, output,
                           n_step=n_step, positive_ratio=positive_ratio, device=device)


def _annotate_bound(config, data, checkpoint, dataset_root, output, *, n_step, positive_ratio, device):
    from openpi.training import data_loader

    dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    source = dataset._dataset if isinstance(dataset, data_loader.TransformedDataset) else dataset
    raw = source.hf_dataset.with_format(None)
    indices = np.asarray(raw["index"], dtype=np.int64).reshape(-1)
    episodes = np.asarray(raw["episode_index"], dtype=np.int64).reshape(-1)
    frames = np.asarray(raw["frame_index"], dtype=np.int64).reshape(-1)
    tasks = np.asarray(raw["task_index"], dtype=np.int64).reshape(-1)
    targets = load_existing_value_targets(raw, "value_target")
    transformed = data_loader.transform_dataset(dataset, data)
    values = _predict_values(config, transformed, checkpoint, device)
    if len(values) != len(indices):
        raise ValueError("价值预测没有覆盖每个源帧")
    advantages, positive, thresholds = label_value_advantages(
        values, targets, episodes, frames, tasks, n_step=n_step, positive_ratio=positive_ratio
    )
    return publish_labeled_copy(dataset_root, output, indices, {
        "predicted_value": values, "advantage": advantages, "is_positive": positive,
    }, receipt={"schema_version": 1, "backend": "pytorch", "method": "n_step_value",
                "checkpoint": str(checkpoint), "n_step": n_step, "positive_ratio": positive_ratio,
                "frame_count": len(indices), "positive_count": int(positive.sum()), "thresholds": thresholds},
        receipt_name="value_advantage.json")


def _trained_value_checkpoint(path):
    checkpoint = Path(path).resolve(strict=True)
    if checkpoint.name == "model.safetensors" and checkpoint.is_file():
        checkpoint = checkpoint.parent
    marker = checkpoint / "recap_checkpoint.json"
    if not (checkpoint / "model.safetensors").is_file() or not marker.is_file():
        raise ValueError("价值标注需要包含 model.safetensors 和 recap_checkpoint.json 的已训练价值 checkpoint")
    try:
        with marker.open("rb") as stream:
            contents = stream.read(64 * 1024 + 1)
        if len(contents) > 64 * 1024:
            raise ValueError("checkpoint receipt is too large")
        receipt = json.loads(contents)
    except (OSError, ValueError) as exc:
        raise ValueError("无法读取价值 checkpoint 的 recap_checkpoint.json") from exc
    if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1 or receipt.get("backend") != "pytorch"
            or receipt.get("stage") != "value" or type(receipt.get("step")) is not int or receipt["step"] < 0):
        raise ValueError("所选 checkpoint 不是有效的 PyTorch ReCap 价值模型")
    return checkpoint


def main(config, *, output_dir=None, n_step=50, positive_ratio=0.3):
    if config.recap_training_stage != "value_annotation" or config.pytorch_weight_path is None:
        raise ValueError("价值标注需要 value_annotation 配置和已训练的价值模型权重")
    factory = config.data
    datasets = getattr(factory, "datasets", None)
    if (datasets is not None and len(datasets) != 1) or config.batch_size < 1:
        raise ValueError("价值标注需要单个 LeRobot 数据集和正数 batch")
    if (type(n_step) is not int or not 1 <= n_step <= 1000
            or type(positive_ratio) not in (int, float) or not 0.001 <= positive_ratio <= 1):
        raise ValueError("价值标注的优势窗口须为 1–1000 帧，正例比例须为 0.001–1")
    checkpoint = _trained_value_checkpoint(config.pytorch_weight_path)

    from openpi.training import data_loader

    if datasets is None:
        data = factory.create(config.assets_dirs, config.model)
    else:
        (data,) = factory.create_all(config.assets_dirs, config.model)
    data = data_loader._apply_global_normalization(data, config)
    dataset_root = Path(data.repo_id).resolve(strict=True)
    output_root = Path(config.checkpoint_base_dir).parent if output_dir is None else Path(output_dir)
    device = recap_pytorch._device(0)
    return _annotate_bound(config, data, checkpoint, dataset_root, output_root / DATASET_NAME,
                           n_step=n_step, positive_ratio=positive_ratio, device=device)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config-name")
    source.add_argument("--config-file", type=Path, help="本次训练导出的可信 training_config.py")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--norm-asset-id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-step", type=int, default=50)
    parser.add_argument("--positive-ratio", type=float, default=0.3)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.config_file:
        builder = runpy.run_path(str(args.config_file))["build_config"]
        config = builder(args.config_file.resolve().parent)
    else:
        from openpi.training import config as train_config

        config = train_config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=args.num_workers)
    print(annotate(config, args.checkpoint_dir, args.dataset_root, args.output_dir, dataset_index=args.dataset_index,
                   norm_asset_id=args.norm_asset_id, n_step=args.n_step, positive_ratio=args.positive_ratio,
                   device=args.device))


if __name__ == "__main__":
    cli()
