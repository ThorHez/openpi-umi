"""Locate where ShellGame condition information is lost in the action path.

All model weights are frozen.  Linear ridge probes are trained on an
episode-held-out split for three representations from the same forward pass:

1. per-camera ``history_mem`` tokens;
2. mask-weighted camera-aggregated memory;
3. the 16 action-query tokens consumed by action cross-attention.

The split is jointly balanced by target cup identity and final ball slot so a
probe cannot exploit correlation between those labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

import joint_fk_selection_eval as fk_eval
import memory_linear_probe as linear_probe

from openpi.models import model as model_api
from openpi.models import pi0_mem_fixed_grid_query_action as query_action
from openpi.policies import policy_config
from openpi.training import config as training_config


REPRESENTATIONS = (
    "raw_history_mem",
    "aggregated_memory",
    "action_query_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--num-train", type=int, default=270)
    parser.add_argument("--num-val", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=260810)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=10_000)
    parser.add_argument(
        "--ridge-lambdas",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _make_extractor(model):
    graphdef, state = nnx.split(model)

    def extract(model_state, observation):
        frozen = nnx.merge(graphdef, model_state)
        observation = model_api.preprocess_observation(None, observation, train=False)
        _, _, _, _, encoder_auxes = frozen._embed_prefix_with_history_mem(observation)
        per_stream = jnp.stack(
            [aux["history_mem"] for aux in encoder_auxes], axis=1
        )
        aggregated = frozen._aggregate_stream_memories(observation, encoder_auxes)
        action_queries = frozen.HistoryActionQueryResampler(aggregated)
        stream_mask = jnp.stack(
            [observation.image_masks[name] for name in observation.images], axis=1
        )
        return per_stream, aggregated, action_queries, stream_mask

    compiled = jax.jit(extract)
    return lambda observation: compiled(state, observation)


def _extract_features(policy, extractor, observations, batch_size):
    features: dict[str, np.ndarray] = {}
    valid_stream_indices = None
    for start in range(0, len(observations), batch_size):
        raw_batch = observations[start : start + batch_size]
        transformed = [
            policy._input_transform(jax.tree.map(lambda value: value, obs))  # noqa: SLF001
            for obs in raw_batch
        ]
        valid_size = len(transformed)
        while len(transformed) < batch_size:
            transformed.append(transformed[-1])
        stacked = jax.tree.map(jnp.asarray, fk_eval._stack_dicts(transformed))
        observation = model_api.Observation.from_dict(stacked)
        per_stream, aggregated, action_queries, stream_mask = extractor(observation)
        per_stream = np.asarray(per_stream[:valid_size], dtype=np.float16)
        aggregated = np.asarray(aggregated[:valid_size], dtype=np.float16)
        action_queries = np.asarray(action_queries[:valid_size], dtype=np.float16)
        mask = np.asarray(stream_mask[:valid_size], dtype=bool)

        current_indices = np.flatnonzero(mask[0])
        if valid_stream_indices is None:
            valid_stream_indices = current_indices
            if len(valid_stream_indices) != 2:
                raise RuntimeError(
                    f"Expected two valid image streams, got {mask[0].tolist()}"
                )
            dimensions = {
                "raw_history_mem": int(
                    np.prod(per_stream[:, valid_stream_indices].shape[1:])
                ),
                "aggregated_memory": int(np.prod(aggregated.shape[1:])),
                "action_query_tokens": int(np.prod(action_queries.shape[1:])),
            }
            for name, dimension in dimensions.items():
                features[name] = np.empty(
                    (len(observations), dimension), dtype=np.float16
                )
        elif not np.array_equal(current_indices, valid_stream_indices) or not np.all(
            mask[:, valid_stream_indices]
        ):
            raise RuntimeError("Valid image-stream mask changed between batches")

        end = start + valid_size
        features["raw_history_mem"][start:end] = per_stream[
            :, valid_stream_indices
        ].reshape(valid_size, -1)
        features["aggregated_memory"][start:end] = aggregated.reshape(valid_size, -1)
        features["action_query_tokens"][start:end] = action_queries.reshape(valid_size, -1)
        logging.info("feature extraction %d/%d", end, len(observations))
    return features


def _probe_metadata(fitted):
    return {
        label_name: {
            key: value
            for key, value in label_probe.items()
            if key not in ("weights", "bias")
        }
        for label_name, label_probe in fitted.items()
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs = sorted(
        path
        for path in args.dataset_root.expanduser().resolve().glob("episode_*")
        if path.is_dir()
    )
    train_ids, val_ids = linear_probe._select_split_ids(args, episode_dirs)
    train_records = [fk_eval._load_episode(episode_dirs[int(index)], args) for index in train_ids]
    val_records = [fk_eval._load_episode(episode_dirs[int(index)], args) for index in val_ids]
    train_target = linear_probe._labels(train_records, "target_cup")
    train_slot = linear_probe._labels(train_records, "final_ball_slot")
    val_target = linear_probe._labels(val_records, "target_cup")
    val_slot = linear_probe._labels(val_records, "final_ball_slot")

    config = dataclasses.replace(
        training_config.get_config(args.config), exp_name="condition_stage_probe"
    )
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=fk_eval.GRASP_PROMPT,
    )
    if not isinstance(policy._model, query_action.Pi0MemFixedGridQueryAction):  # noqa: SLF001
        raise TypeError(f"Unexpected model type: {type(policy._model).__name__}")  # noqa: SLF001
    extractor = _make_extractor(policy._model)  # noqa: SLF001

    logging.info("extracting %d frozen training representations", len(train_records))
    train_features = _extract_features(
        policy, extractor, [record["obs"] for record in train_records], args.batch_size
    )
    dimensions = {name: int(values.shape[1]) for name, values in train_features.items()}
    probes = {}
    training_results = {}
    for representation in REPRESENTATIONS:
        logging.info("fitting linear probe: %s", representation)
        probes[representation] = linear_probe._fit_probe(
            train_features.pop(representation), train_target, train_slot, args
        )
        training_results[representation] = _probe_metadata(probes[representation])
        gc.collect()

    logging.info("extracting %d held-out validation representations", len(val_records))
    val_features = _extract_features(
        policy, extractor, [record["obs"] for record in val_records], args.batch_size
    )
    validation_results = {}
    sample_predictions = {
        representation: [] for representation in REPRESENTATIONS
    }
    for representation_index, representation in enumerate(REPRESENTATIONS):
        validation_results[representation] = {}
        values = val_features.pop(representation)
        for label_index, (label_name, labels) in enumerate(
            (("target_identity", val_target), ("final_slot", val_slot))
        ):
            prediction, _ = linear_probe._predict(
                values.copy(), probes[representation][label_name], reference=None
            )
            validation_results[representation][label_name] = (
                linear_probe._summarize_predictions(
                    prediction,
                    labels,
                    normal_prediction=None,
                    donor_labels=None,
                    mean_cosine=None,
                    permutation_repeats=args.permutation_repeats,
                    permutation_seed=(
                        args.sample_seed + 1_000 * representation_index + 100 * label_index
                    ),
                )
            )
            for index, value in enumerate(prediction):
                while len(sample_predictions[representation]) <= index:
                    record = val_records[index]
                    sample_predictions[representation].append(
                        {
                            "episode": record["episode"],
                            "target_identity": record["target_cup"],
                            "final_slot": record["final_ball_slot"],
                            "predictions": {},
                        }
                    )
                sample_predictions[representation][index]["predictions"][label_name] = (
                    linear_probe.CLASSES[int(value)]
                )
        del values
        gc.collect()

    output = {
        "config": config.name,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "split": {
            "episode_held_out": True,
            "jointly_balanced_target_identity_and_final_slot": True,
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "train_episode_ids": train_ids.tolist(),
            "val_episode_ids": val_ids.tolist(),
        },
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "representation_dimensions": dimensions,
        "training_probe_results": training_results,
        "validation_probe_results": validation_results,
        "validation_samples": sample_predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    compact = {
        representation: {
            label: {
                "accuracy": validation_results[representation][label]["accuracy"],
                "correct": validation_results[representation][label]["correct"],
                "p": validation_results[representation][label][
                    "label_permutation_null_p_ge_observed"
                ],
                "cv_accuracy": training_results[representation][label]["cv_best_accuracy"],
            }
            for label in ("target_identity", "final_slot")
        }
        for representation in REPRESENTATIONS
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
