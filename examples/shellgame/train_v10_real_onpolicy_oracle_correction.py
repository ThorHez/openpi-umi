"""Small-step V10 fine-tune with real V10 on-policy Oracle corrections.

Sampling mass is 60% nominal, 25% validated V6 preservation replay, and 15%
real V10 on-policy correction.  The tracker and memory stay frozen.  The new
source contains no model-generated supervision: its trainable rows begin at
the post-policy switch observation and point only to consecutive Oracle
action chunks.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v2 as _v2
from examples.shellgame import train_old_tracker_full_absolute_eef_mixed_correction_v9 as _v9
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from scripts.mem import train_pi0_mem_compress as _trainer


CONFIG_NAME = "pi0_shellgame_v10_real_onpolicy_oracle_correction_260820"
NOMINAL_ROOT = _v9.NOMINAL_ROOT
V6_ROOT = _v9.V6_ROOT
ONPOLICY_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_v10_real_onpolicy_oracle_correction_150_260820"
)
ONPOLICY_AUDIT = ONPOLICY_ROOT / "v10_real_onpolicy_oracle_supervision_audit.json"
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/"
    "499/params"
)

NOMINAL_EPISODES = 5_000
V6_EPISODES = 1_200
ONPOLICY_EPISODES = 150
NOMINAL_ROWS_PER_EPISODE = 175
V6_ROWS_PER_EPISODE = 25
ONPOLICY_ROWS_PER_EPISODE = 15
SOURCE_FRACTIONS = {"nominal": 0.60, "v6": 0.25, "onpolicy": 0.15}


def _source_weight(fraction: float, rows: int) -> float:
    nominal_rows = NOMINAL_EPISODES * NOMINAL_ROWS_PER_EPISODE
    return fraction / SOURCE_FRACTIONS["nominal"] * nominal_rows / rows


V6_PER_ROW_WEIGHT = _source_weight(
    SOURCE_FRACTIONS["v6"], V6_EPISODES * V6_ROWS_PER_EPISODE
)
ONPOLICY_PER_ROW_WEIGHT = _source_weight(
    SOURCE_FRACTIONS["onpolicy"], ONPOLICY_EPISODES * ONPOLICY_ROWS_PER_EPISODE
)


def _sample_phase_groups(
    dataset,
    indices: list[int],
    *,
    expected_episodes: int,
    group_rows: tuple[int, int, int],
    label: str,
) -> list[int]:
    hf = _full_joint._find_hf_dataset(dataset)  # noqa: SLF001
    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)
    phase = np.asarray(hf["phase_id"], dtype=np.int64)
    action_mask = np.asarray(hf["action_mask"], dtype=bool)
    eligible = selected[
        action_mask[selected]
        & (frame[selected] >= 60)
        & (frame[selected] <= 154)
    ]
    target = eligible + 1
    if np.any(episode[target] != episode[eligible]):
        raise ValueError(f"{label}: +1 target crosses episode boundary")
    target_phase = phase[target]
    eligible_episode = episode[eligible]
    # The trainer performs an episode-level train/validation split before this
    # hook.  Require balanced groups only for episodes present in the current
    # split, exactly as the validated V9 sampler does.
    expected = np.unique(eligible_episode)
    if len(expected) == 0:
        raise ValueError(f"{label}: current split contains no eligible episodes")

    first_lift = np.full(expected_episodes, 1_000, dtype=np.int64)
    lift_rows = phase == 11
    np.minimum.at(first_lift, episode[lift_rows], frame[lift_rows])
    lift_step = frame[target] - first_lift[eligible_episode]
    masks = (
        np.isin(target_phase, (8, 9)),
        target_phase == 10,
        (target_phase == 11) & (lift_step >= 0) & (lift_step < 10),
    )
    names = ("recovery", "grasp", "early_lift")
    groups = tuple(
        _v9._resize_each_episode(  # noqa: SLF001
            eligible[mask],
            eligible_episode[mask],
            expected,
            count,
            seed=261100 + group_index * 1000,
            label=f"{label} {name}",
        )
        for group_index, (mask, count, name) in enumerate(
            zip(masks, group_rows, names, strict=True)
        )
    )
    merged = np.concatenate(groups)
    required = len(expected) * sum(group_rows)
    if len(merged) != required:
        raise RuntimeError(f"{label}: emitted {len(merged)}, expected {required}")
    logging.info(
        "%s groups=%s unique=%d emitted=%d",
        label,
        dict(zip(names, [len(group) for group in groups], strict=True)),
        len(np.unique(merged)),
        len(merged),
    )
    return np.random.default_rng(261199 + len(selected)).permutation(merged).tolist()


def _indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    source = _v9._dataset_repo_id(dataset)  # noqa: SLF001
    if source == NOMINAL_ROOT.resolve():
        return _full_joint._balanced_full_action_indices(dataset, indices, None)  # noqa: SLF001
    if source == V6_ROOT.resolve():
        return _sample_phase_groups(
            dataset,
            indices,
            expected_episodes=V6_EPISODES,
            group_rows=(18, 4, 3),
            label="V10-onpolicy V6 preservation",
        )
    if source == ONPOLICY_ROOT.resolve():
        return _sample_phase_groups(
            dataset,
            indices,
            expected_episodes=ONPOLICY_EPISODES,
            group_rows=(10, 3, 2),
            label="V10 real on-policy Oracle",
        )
    raise ValueError(f"Unknown V10-onpolicy source: {source}")


def _validate_data() -> None:
    for root in (NOMINAL_ROOT, V6_ROOT, ONPOLICY_ROOT):
        if not root.is_dir():
            raise FileNotFoundError(root)
    if not pathlib.Path(DEFAULT_INIT_CHECKPOINT).is_dir():
        raise FileNotFoundError(DEFAULT_INIT_CHECKPOINT)
    if not ONPOLICY_AUDIT.is_file():
        raise FileNotFoundError(ONPOLICY_AUDIT)
    audit = json.loads(ONPOLICY_AUDIT.read_text(encoding="utf-8"))
    if audit.get("ok") is not True:
        raise ValueError(f"On-policy audit failed: {ONPOLICY_AUDIT}")
    raw_audit = audit.get("raw_audit", {})
    if raw_audit.get("episodes") != ONPOLICY_EPISODES:
        raise ValueError(f"On-policy audit episode count mismatch: {raw_audit}")
    if raw_audit.get("model_generated_actions_supervised") is not False:
        raise ValueError("On-policy audit does not prove Oracle-only supervision")


def build_config(args):
    parent = _v2.build_config(args)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        datasets=[
            _v2._eef_data_config(str(ONPOLICY_ROOT)),  # noqa: SLF001
            _v2._eef_data_config(str(V6_ROOT)),  # noqa: SLF001
            _v2._eef_data_config(str(NOMINAL_ROOT)),  # noqa: SLF001
        ],
        weights=[ONPOLICY_PER_ROW_WEIGHT, V6_PER_ROW_WEIGHT, 1.0],
        use_merged_norm_stats=False,
    )
    return dataclasses.replace(
        parent,
        name=CONFIG_NAME,
        exp_name=args.exp_name,
        data=data,
        # Keep fresh runs as the default, but allow an interrupted run to
        # restore the full Orbax train_state (optimizer, EMA, RNG, and step).
        resume=os.environ.get("OPENPI_RESUME_TRAINING", "0") == "1",
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.init_checkpoint == str(_full_joint.OLD_QUERY_ACTION_CHECKPOINT):
        args.init_checkpoint = DEFAULT_INIT_CHECKPOINT
    _validate_data()
    logging.info(
        "V10 real-onpolicy source mass nominal/v6/onpolicy=60/25/15; "
        "weights=(1,%.9f,%.9f); init=%s; tracker+memory frozen",
        V6_PER_ROW_WEIGHT,
        ONPOLICY_PER_ROW_WEIGHT,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _indices  # noqa: SLF001
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
