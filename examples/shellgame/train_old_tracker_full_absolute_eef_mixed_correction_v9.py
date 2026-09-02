"""Fine-tune the validated V6 EEF policy with safe, error-aware V9 replay.

This recipe intentionally keeps the V6 tracker, memory, current-frame reader,
normalization, and action architecture unchanged.  Only the already-approved
full-action trainable filter is used.  Global training mass is exactly:

* 60% phase-balanced nominal demonstrations;
* 15% V6 replay: 9% recovery, 3% grasp, 3% early lift;
* 25% V9 replay: 10% hard initial recovery, 7% low 1--4 mm correction,
  4% aligned descent continuation, 2% grasp, and 2% early lift.

Therefore the complete target remains 60/30/5/5 for nominal / recovery /
grasp / lift, while 15% of the batch explicitly rehearses the last validated
V6 behavior.  V9 difficulty is measured from the current EEF and the first
aligned Oracle target, not inferred from a fixed phase or frame number.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v2 as _v2
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v3 as _v3
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v6 as _v6
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer

CONFIG_NAME = "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v9_260819"
V9_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819"
)
V6_ROOT = pathlib.Path(_v6.CORRECTION_ROOT)
NOMINAL_ROOT = pathlib.Path(_v2.NOMINAL_ROOT)
V9_METRICS_PATH = V9_ROOT / "xy_sampling_metrics_v9.npz"
V9_AUDIT_PATH = V9_ROOT / "safe_balanced_recovery_v9_oracle_supervision_audit.json"
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/"
    "5999/params"
)

EXPECTED_OBSERVE_TASK = "Observe the ball moving under a cup and remember which cup contains it."
SHORT_GRASP_TASK = "Grasp and lift the cup containing the ball."
LONG_GRASP_TASK = "The shell game has ended. Grasp and lift the cup containing the ball."

EPISODE_FRAMES = 155
NOMINAL_EPISODES = 5_000
V6_EPISODES = 1_200
V9_EPISODES = 1_200
NOMINAL_ROWS_PER_EPISODE = 175
V6_ROWS_PER_EPISODE = 10
V9_ROWS_PER_EPISODE = 25

SOURCE_FRACTIONS = {"nominal": 0.60, "v6": 0.15, "v9": 0.25}
V6_GROUP_NAMES = ("recovery", "grasp", "early_lift")
V6_GROUP_ROWS_PER_EPISODE = (6, 2, 2)
V9_GROUP_NAMES = (
    "hard_initial_recovery",
    "low_1_4mm_le40mm",
    "aligned_continuation",
    "grasp",
    "early_lift",
)
V9_GROUP_ROWS_PER_EPISODE = (10, 7, 4, 2, 2)

NOMINAL_FILTERED_ROWS = NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE
V6_FILTERED_ROWS = V6_EPISODES * V6_ROWS_PER_EPISODE
V9_FILTERED_ROWS = V9_EPISODES * V9_ROWS_PER_EPISODE


def _source_weight(source_fraction: float, source_rows: int) -> float:
    return source_fraction / SOURCE_FRACTIONS["nominal"] * NOMINAL_FILTERED_ROWS / source_rows


V6_PER_ROW_WEIGHT = _source_weight(SOURCE_FRACTIONS["v6"], V6_FILTERED_ROWS)
V9_PER_ROW_WEIGHT = _source_weight(SOURCE_FRACTIONS["v9"], V9_FILTERED_ROWS)


def _dataset_repo_id(dataset) -> pathlib.Path:
    """Recover the source path through transform/video wrapper layers."""
    current = dataset
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        repo_id = getattr(current, "repo_id", None)
        if repo_id:
            return pathlib.Path(str(repo_id)).expanduser().resolve()
        current = getattr(current, "_dataset", None)
    raise ValueError("Could not recover LeRobot repo_id from the wrapped dataset")


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_prompt(root: pathlib.Path, grasp_task: str) -> None:
    tasks = _jsonl(root / "meta/tasks.jsonl")
    expected = [
        {"task_index": 0, "task": EXPECTED_OBSERVE_TASK},
        {"task_index": 1, "task": grasp_task},
    ]
    if tasks != expected:
        raise ValueError(f"Prompt contract mismatch for {root}: actual={tasks}, expected={expected}")


def validate_data_contracts() -> None:
    for root in (NOMINAL_ROOT, V6_ROOT, V9_ROOT):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing V9 recipe data source: {root}")
    _validate_prompt(NOMINAL_ROOT, SHORT_GRASP_TASK)
    _validate_prompt(V6_ROOT, LONG_GRASP_TASK)
    _validate_prompt(V9_ROOT, LONG_GRASP_TASK)
    if not pathlib.Path(DEFAULT_INIT_CHECKPOINT).is_dir():
        raise FileNotFoundError(f"Missing validated V6 initialization: {DEFAULT_INIT_CHECKPOINT}")
    if not V9_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Missing {V9_METRICS_PATH}; run build_eef_xy_sampling_metrics_v9.py first")
    audit = json.loads(V9_AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("ok") is not True or audit.get("raw_audit", {}).get("quota_complete") is not True:
        raise ValueError(f"V9 audit is not complete: {V9_AUDIT_PATH}")
    prompt_audit = audit.get("prompt_audit", {})
    if (
        prompt_audit.get("grasp_task") != LONG_GRASP_TASK
        or int(prompt_audit.get("episodes_checked", -1)) != V9_EPISODES
    ):
        raise ValueError(f"V9 prompt audit is missing or stale: {V9_AUDIT_PATH}")


def _evenly_spaced_unique(
    group: np.ndarray,
    count: int,
    *,
    label: str,
    episode: int,
) -> np.ndarray:
    group = np.asarray(group, dtype=np.int64)
    if len(group) < count:
        raise ValueError(f"Episode {episode} has {len(group)} {label} rows; need {count}")
    positions = np.rint(np.linspace(0, len(group) - 1, count)).astype(np.int64)
    chosen = group[positions]
    if len(np.unique(chosen)) != count:
        raise RuntimeError(f"{label} selector duplicated a source row in episode {episode}")
    return chosen


def _resize_each_episode(
    indices: np.ndarray,
    episodes: np.ndarray,
    expected_episodes: np.ndarray,
    rows_per_episode: int,
    *,
    seed: int,
    label: str,
) -> np.ndarray:
    output = []
    for episode in expected_episodes:
        group = indices[episodes == episode]
        if len(group) == 0:
            raise ValueError(f"V9 {label} has no rows for episode {int(episode)}")
        output.append(
            _v3._resize_group(  # noqa: SLF001
                group,
                rows_per_episode,
                seed=seed + int(episode),
            )
        )
    return np.concatenate(output)


def _allocate(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or weights.ndim != 1 or not len(weights) or np.any(weights < 0):
        raise ValueError(f"Invalid allocation: total={total}, weights={weights}")
    raw = weights / weights.sum() * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _balance_strata(
    indices: np.ndarray,
    strata: np.ndarray,
    target_size: int,
    *,
    seed: int,
    label: str,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    strata = np.asarray(strata)
    if len(indices) == 0 or len(indices) != len(strata):
        raise ValueError(f"V9 {label} received empty or mismatched rows")
    values = np.unique(strata)
    targets = _allocate(target_size, np.ones(len(values), dtype=np.float64))
    output = []
    for offset, (value, count) in enumerate(zip(values, targets, strict=True)):
        output.append(
            _v3._resize_group(  # noqa: SLF001
                indices[strata == value],
                int(count),
                seed=seed + 97 * offset,
            )
        )
    merged = np.concatenate(output)
    if len(merged) != target_size:
        raise RuntimeError(f"V9 {label} emitted {len(merged)} rows; expected {target_size}")
    return merged


def _sample_v6(dataset, indices: list[int]) -> list[int]:
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    full_episode = np.asarray(hf["episode_index"], dtype=np.int64)
    full_frame = np.asarray(hf["frame_index"], dtype=np.int64)
    full_phase = np.asarray(hf["phase_id"], dtype=np.int64)
    full_mask = np.asarray(hf["action_mask"], dtype=bool)
    eligible = selected[full_mask[selected] & (full_frame[selected] >= 60) & (full_frame[selected] <= 153)]
    if not len(eligible):
        raise ValueError("V6 replay selected no eligible rows")
    episode = full_episode[eligible]
    target_phase = full_phase[eligible + 1]
    expected_episodes = np.unique(episode)
    first_lift = np.full(V6_EPISODES, EPISODE_FRAMES, dtype=np.int64)
    all_lift = full_phase == 11
    np.minimum.at(first_lift, full_episode[all_lift], full_frame[all_lift])
    lift_step = full_frame[eligible] + 1 - first_lift[episode]

    output = []
    for ep in expected_episodes:
        rows = eligible[episode == ep]
        phases = target_phase[episode == ep]
        steps = lift_step[episode == ep]
        recovery = rows[np.isin(phases, (8, 9))]
        grasp = rows[phases == 10]
        lift = rows[(phases == 11) & (steps >= 0) & (steps < 10)]
        output.append(
            np.concatenate(
                [
                    _evenly_spaced_unique(
                        recovery,
                        V6_GROUP_ROWS_PER_EPISODE[0],
                        label="V6 recovery",
                        episode=int(ep),
                    ),
                    _evenly_spaced_unique(
                        grasp,
                        V6_GROUP_ROWS_PER_EPISODE[1],
                        label="V6 grasp",
                        episode=int(ep),
                    ),
                    _evenly_spaced_unique(
                        lift,
                        V6_GROUP_ROWS_PER_EPISODE[2],
                        label="V6 early lift",
                        episode=int(ep),
                    ),
                ]
            )
        )
    merged = np.concatenate(output)
    expected = len(expected_episodes) * V6_ROWS_PER_EPISODE
    if len(merged) != expected or len(np.unique(merged)) != len(merged):
        raise RuntimeError(
            f"V6 replay emitted {len(merged)} rows ({len(np.unique(merged))} unique); expected {expected}"
        )
    logging.info(
        "V9 recipe V6 replay: episodes=%d rows=%d per_episode=%s global_mass=.15",
        len(expected_episodes),
        len(merged),
        dict(zip(V6_GROUP_NAMES, V6_GROUP_ROWS_PER_EPISODE, strict=True)),
    )
    return np.random.default_rng(260911 + len(indices)).permutation(merged).tolist()


def _load_v9_metrics() -> dict[str, np.ndarray]:
    with np.load(V9_METRICS_PATH, allow_pickle=False) as payload:
        if int(payload["schema_version"]) != 2:
            raise ValueError(f"Unsupported V9 metrics schema: {payload['schema_version']}")
        converted_root = pathlib.Path(str(payload["converted_root"])).resolve()
        if converted_root != V9_ROOT.resolve():
            raise ValueError(f"V9 sidecar source mismatch: sidecar={converted_root}, expected={V9_ROOT.resolve()}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _sample_v9(dataset, indices: list[int]) -> list[int]:
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    metrics = _load_v9_metrics()
    if len(metrics["episode_index"]) != len(hf):
        raise ValueError(f"V9 metrics rows={len(metrics['episode_index'])} do not match dataset rows={len(hf)}")
    hf_episode = np.asarray(hf["episode_index"], dtype=np.int64)[selected]
    hf_frame = np.asarray(hf["frame_index"], dtype=np.int64)[selected]
    if not np.array_equal(hf_episode, metrics["episode_index"][selected]):
        raise ValueError("V9 sidecar episode_index does not match LeRobot rows")
    if not np.array_equal(hf_frame, metrics["frame_index"][selected]):
        raise ValueError("V9 sidecar frame_index does not match LeRobot rows")

    episode = metrics["episode_index"][selected]
    frame = metrics["frame_index"][selected]
    phase = metrics["target_phase"][selected]
    lift_step = metrics["target_lift_step"][selected]
    action_mask = metrics["action_mask"][selected]
    error = metrics["xy_error_m"][selected]
    height = metrics["height_above_grasp_m"][selected]
    final_slot = metrics["final_slot"][selected]
    correction_sector = metrics["correction_sector"][selected]
    initial_error = metrics["initial_xy_error_m"][selected]

    eligible = action_mask & (frame >= 60) & (frame <= 153)
    recovery = eligible & np.isin(phase, (8, 9))
    # All accepted V9 episodes are >5 mm from the live cup centre.  Because
    # the commanded target intentionally includes <=2 mm grasp jitter, use a
    # 4 mm target-error floor so every designed episode still contributes.
    hard_mask = recovery & (initial_error > 0.005) & (error > 0.004)
    low_mask = recovery & (error >= 0.001) & (error <= 0.004) & (height >= -0.003) & (height <= 0.040)
    aligned_mask = recovery & (error < 0.001) & (height >= -0.003) & (height <= 0.060)
    grasp_mask = eligible & (phase == 10)
    lift_mask = eligible & (phase == 11) & (lift_step >= 0) & (lift_step < 10)
    masks = (hard_mask, low_mask, aligned_mask, grasp_mask, lift_mask)
    if any(not np.any(mask) for mask in masks):
        raise ValueError(
            f"V9 sampling has an empty group: "
            f"{dict(zip(V9_GROUP_NAMES, [int(mask.sum()) for mask in masks], strict=True))}"
        )
    expected_episodes = np.unique(episode[eligible])
    targets = np.asarray(V9_GROUP_ROWS_PER_EPISODE) * len(expected_episodes)

    hard = _resize_each_episode(
        selected[hard_mask],
        episode[hard_mask],
        expected_episodes,
        V9_GROUP_ROWS_PER_EPISODE[0],
        seed=260921,
        label="hard_initial_recovery",
    )
    # Balance actual correction direction within each final spatial cup.  The
    # combined stratum prevents the historical right-cup / positive-Y skew.
    low_strata = final_slot[low_mask].astype(np.int16) * 16 + correction_sector[low_mask]
    low = _balance_strata(
        selected[low_mask],
        low_strata,
        int(targets[1]),
        seed=260922,
        label="low_1_4mm_le40mm",
    )
    height_band = np.where(height[aligned_mask] <= 0.015, 0, np.where(height[aligned_mask] <= 0.030, 1, 2))
    aligned_strata = final_slot[aligned_mask].astype(np.int16) * 3 + height_band
    aligned = _balance_strata(
        selected[aligned_mask],
        aligned_strata,
        int(targets[2]),
        seed=260923,
        label="aligned_continuation",
    )
    grasp = _resize_each_episode(
        selected[grasp_mask],
        episode[grasp_mask],
        expected_episodes,
        V9_GROUP_ROWS_PER_EPISODE[3],
        seed=260924,
        label="grasp",
    )
    lift = _resize_each_episode(
        selected[lift_mask],
        episode[lift_mask],
        expected_episodes,
        V9_GROUP_ROWS_PER_EPISODE[4],
        seed=260925,
        label="early_lift",
    )
    groups = (hard, low, aligned, grasp, lift)
    merged = np.concatenate(groups)
    expected = len(expected_episodes) * V9_ROWS_PER_EPISODE
    if len(merged) != expected:
        raise RuntimeError(f"V9 sampler emitted {len(merged)} rows; expected {expected}")

    logging.info(
        "V9 measured-error sampling: episodes=%d raw=%s targets=%s output=%d "
        "global_mass={'hard_initial':.10,'low_1_4mm':.07,'aligned':.04,'grasp':.02,'lift':.02}",
        len(expected_episodes),
        dict(zip(V9_GROUP_NAMES, [int(mask.sum()) for mask in masks], strict=True)),
        dict(zip(V9_GROUP_NAMES, [len(group) for group in groups], strict=True)),
        len(merged),
    )
    return np.random.default_rng(260926 + len(selected)).permutation(merged).tolist()


def _v9_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    source = _dataset_repo_id(dataset)
    if source == NOMINAL_ROOT.resolve():
        return _full_joint._balanced_full_action_indices(dataset, indices, None)  # noqa: SLF001
    if source == V6_ROOT.resolve():
        return _sample_v6(dataset, indices)
    if source == V9_ROOT.resolve():
        return _sample_v9(dataset, indices)
    raise ValueError(f"Unknown dataset source in V9 recipe: {source}")


def build_config(args):
    parent = _v2.build_config(args)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        datasets=[
            _v2._eef_data_config(str(V9_ROOT)),  # noqa: SLF001
            _v2._eef_data_config(str(V6_ROOT)),  # noqa: SLF001
            _v2._eef_data_config(str(NOMINAL_ROOT)),  # noqa: SLF001
        ],
        weights=[V9_PER_ROW_WEIGHT, V6_PER_ROW_WEIGHT, 1.0],
        # Preserve the validated nominal action/state coordinate system.
        use_merged_norm_stats=False,
    )
    return dataclasses.replace(
        parent,
        name=CONFIG_NAME,
        exp_name=args.exp_name,
        data=data,
        resume=False,
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.init_checkpoint == str(_full_joint.OLD_QUERY_ACTION_CHECKPOINT):
        args.init_checkpoint = DEFAULT_INIT_CHECKPOINT
    if args.steps < 2:
        raise ValueError("V9 mixed replay training requires at least two steps")
    validate_data_contracts()
    logging.info(
        "V9 exact source mass nominal/v6/v9=60/15/25 weights=(1,%.9f,%.9f) metrics=%s init=%s frozen=tracker+memory",
        V6_PER_ROW_WEIGHT,
        V9_PER_ROW_WEIGHT,
        V9_METRICS_PATH,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _v9_indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
