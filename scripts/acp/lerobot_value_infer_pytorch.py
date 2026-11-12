"""Use a PyTorch ReCap value checkpoint to publish a labeled LeRobot copy."""

import argparse
import dataclasses
import os
from pathlib import Path
import runpy

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from openpi.training import config as train_config, data_loader, recap_pytorch
from openpi.training.recap_advantages import label_value_advantages, load_existing_value_targets
from openpi.training.stage_progress_labels import publish_labeled_copy


def _data_config(config, dataset_root, checkpoint, dataset_index, norm_asset_id):
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
    dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    source = dataset._dataset if isinstance(dataset, data_loader.TransformedDataset) else dataset
    raw = source.hf_dataset.with_format(None)
    indices = np.asarray(raw["index"], dtype=np.int64).reshape(-1)
    episodes = np.asarray(raw["episode_index"], dtype=np.int64).reshape(-1)
    frames = np.asarray(raw["frame_index"], dtype=np.int64).reshape(-1)
    tasks = np.asarray(raw["task_index"], dtype=np.int64).reshape(-1)
    targets = load_existing_value_targets(raw, "value_target")
    transformed = data_loader.transform_dataset(dataset, data)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
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


def main():
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
        config = train_config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=args.num_workers)
    print(annotate(config, args.checkpoint_dir, args.dataset_root, args.output_dir, dataset_index=args.dataset_index,
                   norm_asset_id=args.norm_asset_id, n_step=args.n_step, positive_ratio=args.positive_ratio,
                   device=args.device))


if __name__ == "__main__":
    main()
