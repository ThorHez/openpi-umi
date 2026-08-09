"""Probe ShellGame semantics in frozen compressed-memory representations.

The script extracts two representations at the first grasp observation:

* ``history_mem``: the compressed historical tokens before memory fusion.
* ``fused_prefix``: the projected image tokens after history cross-attention;
  these are the exact visual tokens consumed by the PaliGemma/action stack.

Linear ridge classifiers are fit on training-split episodes and evaluated on
held-out validation episodes under paired history interventions.  No policy
or probe parameter is used while extracting validation features.
"""

# This diagnostic intentionally uses evaluator / Policy internals so its
# preprocessing and temporal window exactly match online inference.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import json
import logging
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import joint_fk_memory_ablation as memory_ablation
import joint_fk_selection_eval as fk_eval
import numpy as np

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.training import config as training_config

CLASSES = ("left", "middle", "right")
MODES = (
    "normal",
    "memory_off",
    "shuffle_history",
    "wrong_history",
    "reveal_only",
    "current_only",
)
REPRESENTATIONS = ("history_mem", "fused_prefix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_mem_compress_evan_shellgame_openpi_joint_260727")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pi0_mem_compress_evan_shellgame_openpi_joint_260727/my_experiment/23000"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    # Defaults are divisible by the nine (target identity, final slot)
    # combinations, enabling an exactly joint-balanced probe split.
    parser.add_argument("--num-train", type=int, default=594)
    parser.add_argument("--num-val", type=int, default=297)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260806)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=10_000)
    parser.add_argument(
        "--ridge-lambdas",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/shellgame/memory_linear_probe/23000_train594_val297.json"),
    )
    return parser.parse_args()


def _metadata_label_pair(episode_dir: Path) -> tuple[str, str]:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    return str(metadata["target_cup_identity"]), str(metadata["final_ball_cup"])


def _joint_balanced_sample(
    pool: np.ndarray,
    episode_dirs: list[Path],
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    num_groups = len(CLASSES) ** 2
    if count % num_groups != 0:
        raise ValueError(f"Balanced probe counts must be divisible by {num_groups}; got {count}.")
    per_group = count // num_groups
    grouped: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for episode_id in pool:
        grouped[_metadata_label_pair(episode_dirs[int(episode_id)])].append(int(episode_id))

    expected = [(target, slot) for target in CLASSES for slot in CLASSES]
    missing = {
        f"{target}->{slot}": per_group - len(grouped[(target, slot)])
        for target, slot in expected
        if len(grouped[(target, slot)]) < per_group
    }
    if missing:
        raise ValueError(f"Not enough episodes for a joint-balanced probe split: {missing}")

    rng = np.random.default_rng(seed)
    selected = []
    for pair in expected:
        candidates = np.asarray(grouped[pair], dtype=np.int64)
        selected.extend(rng.choice(candidates, size=per_group, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _select_split_ids(
    args: argparse.Namespace,
    episode_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    num_episodes = len(episode_dirs)
    val_pool = fk_eval._validation_episode_ids(num_episodes, args.val_ratio, args.split_seed)
    train_pool = np.setdiff1d(np.arange(num_episodes, dtype=np.int64), val_pool)
    if not 0 < args.num_train <= len(train_pool):
        raise ValueError(f"--num-train must be in [1, {len(train_pool)}]")
    if not 0 < args.num_val <= len(val_pool):
        raise ValueError(f"--num-val must be in [1, {len(val_pool)}]")
    train_ids = _joint_balanced_sample(
        train_pool, episode_dirs, args.num_train, seed=args.sample_seed
    )
    val_ids = _joint_balanced_sample(
        val_pool, episode_dirs, args.num_val, seed=args.sample_seed + 1
    )
    return train_ids, val_ids


def _choose_probe_donors(records: list[dict]) -> dict[str, dict]:
    """Choose donors whose two semantic labels both differ from the target."""
    donors = {}
    for index, record in enumerate(records):
        for offset in range(1, len(records) + 1):
            candidate = records[(index + offset) % len(records)]
            if (
                candidate["target_cup"] != record["target_cup"]
                and candidate["final_ball_slot"] != record["final_ball_slot"]
            ):
                donors[record["episode"]] = candidate
                break
        if record["episode"] not in donors:
            raise RuntimeError(f"No two-label donor found for {record['episode']}")
    return donors


def _current_only_observation(record: dict, num_frames: int) -> dict:
    obs = dict(record["obs"])
    current_index = num_frames - 1
    for stream in ("left_wrist_0_rgb_0", "left_wrist_0_rgb_1"):
        current = record["obs"][f"{stream}_{current_index}"]
        for index in range(current_index):
            obs[f"{stream}_{index}"] = current
    return obs


def _mode_observation(
    record: dict,
    mode: str,
    donor: dict,
    args: argparse.Namespace,
) -> dict:
    if mode == "current_only":
        return _current_only_observation(record, args.num_frames)
    return memory_ablation._mode_observation(
        record,
        mode,
        donor=donor,
        seed=args.sample_seed,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
    )


def _set_memory_gate(model, value: float) -> None:
    module = model.PaliGemma.img.module
    model.PaliGemma.img.module = dataclasses.replace(module, history_gate_fixed=float(value))


def _make_feature_extractor(model):
    """Freeze an NNX model and return a lightweight JIT feature extractor."""
    graphdef, state = nnx.split(model)

    def extract(model_state, observation):
        frozen_model = nnx.merge(graphdef, model_state)
        observation = model_api.preprocess_observation(None, observation, train=False)
        prefix_tokens, _, _, _, encoder_auxes = frozen_model._embed_prefix_with_history_mem(observation)

        history = jnp.stack([aux["history_mem"] for aux in encoder_auxes], axis=1)
        fused_streams = []
        offset = 0
        for aux in encoder_auxes:
            stream_length = aux["pre_ln"].shape[1]
            fused_streams.append(prefix_tokens[:, offset : offset + stream_length])
            offset += stream_length
        fused = jnp.stack(fused_streams, axis=1)
        stream_mask = jnp.stack([observation.image_masks[name] for name in observation.images], axis=1)
        return history, fused, stream_mask

    compiled = jax.jit(extract)
    return lambda observation: compiled(state, observation)


def _extract_features(policy, extractor, observations: list[dict], batch_size: int) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    valid_stream_indices = None
    for start in range(0, len(observations), batch_size):
        raw_batch = observations[start : start + batch_size]
        transformed = [policy._input_transform(jax.tree.map(lambda x: x, obs)) for obs in raw_batch]
        valid_size = len(transformed)
        while len(transformed) < batch_size:
            transformed.append(transformed[-1])
        stacked = jax.tree.map(jnp.asarray, fk_eval._stack_dicts(transformed))
        observation = model_api.Observation.from_dict(stacked)
        history, fused, stream_mask = extractor(observation)
        history = np.asarray(history[:valid_size], dtype=np.float16)
        fused = np.asarray(fused[:valid_size], dtype=np.float16)
        mask = np.asarray(stream_mask[:valid_size], dtype=bool)

        current_indices = np.flatnonzero(mask[0])
        if valid_stream_indices is None:
            valid_stream_indices = current_indices
            if len(valid_stream_indices) != 2:
                raise RuntimeError(f"Expected two valid image streams, got mask={mask[0].tolist()}")
            history_dim = int(np.prod(history[:, valid_stream_indices].shape[1:]))
            fused_dim = int(np.prod(fused[:, valid_stream_indices].shape[1:]))
            features["history_mem"] = np.empty((len(observations), history_dim), dtype=np.float16)
            features["fused_prefix"] = np.empty((len(observations), fused_dim), dtype=np.float16)
        elif not np.array_equal(current_indices, valid_stream_indices) or not np.all(
            mask[:, valid_stream_indices]
        ):
            raise RuntimeError("Valid image-stream mask changed between batches")

        end = start + valid_size
        features["history_mem"][start:end] = history[:, valid_stream_indices].reshape(valid_size, -1)
        features["fused_prefix"][start:end] = fused[:, valid_stream_indices].reshape(valid_size, -1)
        logging.info("feature extraction %d/%d", end, len(observations))
    return features


def _normalize_rows_in_place(features: np.ndarray, batch_size: int = 8) -> None:
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        block = features[start:end].astype(np.float32)
        norm = np.sqrt(np.einsum("ij,ij->i", block, block, optimize=True))
        features[start:end] = (block / np.maximum(norm[:, None], 1e-12)).astype(np.float16)


def _joint_stratified_folds(
    target_labels: np.ndarray,
    slot_labels: np.ndarray,
    num_folds: int,
    seed: int,
) -> np.ndarray:
    folds = np.empty(len(target_labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for index, pair in enumerate(zip(target_labels, slot_labels, strict=True)):
        groups[(int(pair[0]), int(pair[1]))].append(index)
    for indices in groups.values():
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            folds[index] = offset % num_folds
    return folds


def _one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(CLASSES), dtype=np.float32)[labels]


def _ridge_cv_scores(
    gram: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    lambdas: list[float],
) -> tuple[float, dict[str, float]]:
    targets = _one_hot(labels)
    fold_scores = {ridge_lambda: [] for ridge_lambda in lambdas}
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        val = np.flatnonzero(folds == fold)
        train_gram = gram[np.ix_(train, train)].astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(train_gram)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected_targets = eigenvectors.T @ targets[train]
        cross_gram = gram[np.ix_(val, train)]
        for ridge_lambda in lambdas:
            alpha = eigenvectors @ (projected_targets / (eigenvalues[:, None] + ridge_lambda))
            prediction = np.argmax(cross_gram @ alpha, axis=1)
            fold_scores[ridge_lambda].append(float(np.mean(prediction == labels[val])))
    scores = {
        f"{ridge_lambda:g}": float(np.mean(fold_scores[ridge_lambda]))
        for ridge_lambda in lambdas
    }
    best = min(
        lambdas,
        key=lambda value: (-scores[f"{value:g}"], value),
    )
    return float(best), scores


def _fit_probe(
    features: np.ndarray,
    target_labels: np.ndarray,
    slot_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    _normalize_rows_in_place(features)
    device_features = jnp.asarray(features)
    gram = np.asarray(
        jnp.matmul(device_features, device_features.T, preferred_element_type=jnp.float32) + 1.0,
        dtype=np.float32,
    )
    folds = _joint_stratified_folds(target_labels, slot_labels, args.cv_folds, args.sample_seed)
    lambdas = [float(value) for value in args.ridge_lambdas]

    outputs = {}
    for label_name, labels in (("target_identity", target_labels), ("final_slot", slot_labels)):
        ridge_lambda, cv_scores = _ridge_cv_scores(gram, labels, folds, lambdas)
        alpha = np.linalg.solve(
            gram + ridge_lambda * np.eye(len(gram), dtype=np.float32),
            _one_hot(labels),
        )
        weights = np.asarray(
            jnp.matmul(device_features.T, jnp.asarray(alpha), preferred_element_type=jnp.float32),
            dtype=np.float32,
        )
        bias = np.sum(alpha, axis=0, dtype=np.float32)
        fitted = np.argmax(gram @ alpha, axis=1)
        outputs[label_name] = {
            "weights": weights,
            "bias": bias,
            "ridge_lambda": ridge_lambda,
            "cv_accuracy_by_lambda": cv_scores,
            "cv_best_accuracy": cv_scores[f"{ridge_lambda:g}"],
            "training_fit_accuracy": float(np.mean(fitted == labels)),
        }
    del device_features
    gc.collect()
    return outputs


def _predict(
    features: np.ndarray,
    probe: dict,
    reference: np.ndarray | None,
    batch_size: int = 8,
) -> tuple[np.ndarray, float | None]:
    _normalize_rows_in_place(features, batch_size=batch_size)
    predictions = np.empty(len(features), dtype=np.int64)
    cosine_parts = []
    weights = jnp.asarray(probe["weights"])
    bias = jnp.asarray(probe["bias"])
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        device_block = jnp.asarray(features[start:end])
        scores = jnp.matmul(device_block, weights, preferred_element_type=jnp.float32) + bias
        predictions[start:end] = np.asarray(jnp.argmax(scores, axis=1), dtype=np.int64)
        if reference is not None:
            reference_block = jnp.asarray(reference[start:end])
            cosine_parts.append(
                np.asarray(
                    jnp.sum(device_block * reference_block, axis=1, dtype=jnp.float32),
                    dtype=np.float32,
                )
            )
    mean_cosine = float(np.mean(np.concatenate(cosine_parts))) if cosine_parts else None
    return predictions, mean_cosine


def _labels(records: list[dict], key: str) -> np.ndarray:
    class_index = {name: index for index, name in enumerate(CLASSES)}
    return np.asarray([class_index[record[key]] for record in records], dtype=np.int64)


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {CLASSES[index]: int(np.sum(values == index)) for index in range(len(CLASSES))}


def _summarize_predictions(
    prediction: np.ndarray,
    labels: np.ndarray,
    *,
    normal_prediction: np.ndarray | None,
    donor_labels: np.ndarray | None,
    mean_cosine: float | None,
    permutation_repeats: int,
    permutation_seed: int,
) -> dict:
    observed_accuracy = float(np.mean(prediction == labels))
    rng = np.random.default_rng(permutation_seed)
    null_accuracies = np.empty(permutation_repeats, dtype=np.float32)
    for index in range(permutation_repeats):
        null_accuracies[index] = np.mean(prediction == rng.permutation(labels))
    summary = {
        "accuracy": observed_accuracy,
        "correct": int(np.sum(prediction == labels)),
        "num_episodes": len(labels),
        "predicted_distribution": _distribution(prediction),
        "feature_cosine_vs_normal_mean": mean_cosine,
        "label_permutation_null_mean": float(np.mean(null_accuracies)),
        "label_permutation_null_std": float(np.std(null_accuracies)),
        "label_permutation_null_p_ge_observed": float(
            (1 + np.sum(null_accuracies >= observed_accuracy)) / (permutation_repeats + 1)
        ),
        "label_permutation_repeats": int(permutation_repeats),
    }
    if normal_prediction is not None:
        summary["paired_prediction_change_vs_normal_rate"] = float(np.mean(prediction != normal_prediction))
    if donor_labels is not None:
        summary["donor_label_accuracy"] = float(np.mean(prediction == donor_labels))
        summary["donor_label_correct"] = int(np.sum(prediction == donor_labels))
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs = sorted(path for path in args.dataset_root.expanduser().resolve().glob("episode_*") if path.is_dir())
    train_ids, val_ids = _select_split_ids(args, episode_dirs)

    logging.info("loading %d training episodes", len(train_ids))
    train_records = [fk_eval._load_episode(episode_dirs[int(index)], args) for index in train_ids]
    train_target = _labels(train_records, "target_cup")
    train_slot = _labels(train_records, "final_ball_slot")

    config = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=fk_eval.GRASP_PROMPT,
    )
    original_gate = float(policy._model.PaliGemma.img.module.history_gate_fixed)
    _set_memory_gate(policy._model, original_gate)
    extractor_on = _make_feature_extractor(policy._model)
    _set_memory_gate(policy._model, 0.0)
    extractor_off = _make_feature_extractor(policy._model)
    _set_memory_gate(policy._model, original_gate)

    logging.info("extracting normal training representations")
    train_features = _extract_features(
        policy,
        extractor_on,
        [record["obs"] for record in train_records],
        args.batch_size,
    )
    feature_dimensions = {
        representation: int(values.shape[1]) for representation, values in train_features.items()
    }
    probes = {}
    probe_metadata = {}
    for representation in REPRESENTATIONS:
        logging.info("fitting %s linear probes", representation)
        fitted = _fit_probe(train_features[representation], train_target, train_slot, args)
        probes[representation] = fitted
        probe_metadata[representation] = {
            label_name: {
                key: value
                for key, value in label_probe.items()
                if key not in ("weights", "bias")
            }
            for label_name, label_probe in fitted.items()
        }
    del train_features, train_records
    gc.collect()

    logging.info("loading %d validation episodes", len(val_ids))
    val_records = [fk_eval._load_episode(episode_dirs[int(index)], args) for index in val_ids]
    donors = _choose_probe_donors(val_records)
    val_target = _labels(val_records, "target_cup")
    val_slot = _labels(val_records, "final_ball_slot")
    donor_target = _labels([donors[record["episode"]] for record in val_records], "target_cup")
    donor_slot = _labels([donors[record["episode"]] for record in val_records], "final_ball_slot")

    summaries = {}
    sample_predictions = {}
    normal_features: dict[str, np.ndarray] = {}
    normal_predictions: dict[str, dict[str, np.ndarray]] = {}

    if "normal" not in args.modes:
        raise ValueError("The paired probe requires normal to be included in --modes")
    ordered_modes = ["normal", *[mode for mode in args.modes if mode != "normal"]]
    for mode_index, mode in enumerate(ordered_modes):
        logging.info("=== validation mode=%s ===", mode)
        observations = [
            _mode_observation(record, mode, donors[record["episode"]], args)
            for record in val_records
        ]
        extractor = extractor_off if mode == "memory_off" else extractor_on
        mode_features = _extract_features(policy, extractor, observations, args.batch_size)
        mode_summary = {}
        mode_samples = []
        for index, record in enumerate(val_records):
            mode_samples.append(
                {
                    "episode": record["episode"],
                    "target_identity": record["target_cup"],
                    "final_slot": record["final_ball_slot"],
                    "donor_episode": donors[record["episode"]]["episode"] if mode == "wrong_history" else None,
                    "donor_target_identity": CLASSES[donor_target[index]] if mode == "wrong_history" else None,
                    "donor_final_slot": CLASSES[donor_slot[index]] if mode == "wrong_history" else None,
                    "predictions": {},
                }
            )

        for representation_index, representation in enumerate(REPRESENTATIONS):
            mode_summary[representation] = {}
            reference = normal_features.get(representation)
            if mode == "normal":
                normal_features[representation] = mode_features[representation]
                reference = None
                normal_predictions[representation] = {}
            for label_index, (label_name, labels, wrong_labels) in enumerate((
                ("target_identity", val_target, donor_target),
                ("final_slot", val_slot, donor_slot),
            )):
                prediction, mean_cosine = _predict(
                    mode_features[representation],
                    probes[representation][label_name],
                    reference,
                )
                if mode == "normal":
                    normal_predictions[representation][label_name] = prediction
                summary = _summarize_predictions(
                    prediction,
                    labels,
                    normal_prediction=normal_predictions[representation][label_name] if mode != "normal" else None,
                    donor_labels=wrong_labels if mode == "wrong_history" else None,
                    mean_cosine=mean_cosine,
                    permutation_repeats=args.permutation_repeats,
                    permutation_seed=(
                        args.sample_seed
                        + 10_000 * mode_index
                        + 1_000 * representation_index
                        + 100 * label_index
                    ),
                )
                mode_summary[representation][label_name] = summary
                for index, value in enumerate(prediction):
                    mode_samples[index]["predictions"].setdefault(representation, {})[label_name] = CLASSES[value]
        summaries[mode] = mode_summary
        sample_predictions[mode] = mode_samples
        if mode != "normal":
            del mode_features
            gc.collect()

    compact = {
        mode: {
            representation: {
                label: {
                    "accuracy": values["accuracy"],
                    "paired_prediction_change_vs_normal_rate": values.get(
                        "paired_prediction_change_vs_normal_rate"
                    ),
                    "donor_label_accuracy": values.get("donor_label_accuracy"),
                    "feature_cosine_vs_normal_mean": values["feature_cosine_vs_normal_mean"],
                    "label_permutation_null_p_ge_observed": values[
                        "label_permutation_null_p_ge_observed"
                    ],
                }
                for label, values in representation_summary.items()
            }
            for representation, representation_summary in mode_summary.items()
        }
        for mode, mode_summary in summaries.items()
    }
    output = {
        "config": args.config,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "classes": CLASSES,
        "representations": REPRESENTATIONS,
        "feature_dimensions": feature_dimensions,
        "modes": ordered_modes,
        "split": {
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "sample_seed": args.sample_seed,
            "train_episode_ids": train_ids.tolist(),
            "val_episode_ids": val_ids.tolist(),
        },
        "label_distribution": {
            "train_target_identity": _distribution(train_target),
            "train_final_slot": _distribution(train_slot),
            "val_target_identity": _distribution(val_target),
            "val_final_slot": _distribution(val_slot),
            "val_donor_target_identity": _distribution(donor_target),
            "val_donor_final_slot": _distribution(donor_slot),
            "train_joint": dict(
                collections.Counter(
                    f"{CLASSES[target]}->{CLASSES[slot]}"
                    for target, slot in zip(train_target, train_slot, strict=True)
                )
            ),
            "val_joint": dict(
                collections.Counter(
                    f"{CLASSES[target]}->{CLASSES[slot]}"
                    for target, slot in zip(val_target, val_slot, strict=True)
                )
            ),
        },
        "probe_fit": probe_metadata,
        "summaries": summaries,
        "samples": sample_predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
