"""Action-only exact-state overfit probe for failed V10 suffixes.

Only six episodes are used: two deterministic copies of paired-evaluation
episodes 0, 1, and 17.  Memory and the tracker remain frozen.  Every eligible
row points to the complete consecutive Oracle suffix, and world-frame X/Y
dimensions receive 4x loss weight.  This intentionally tests wiring and
learnability on the same states; it is not a generalization experiment.
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
from examples.shellgame import train_old_tracker_full_joint_grasp as _full_joint
from openpi.training import config as _config
from openpi.training import config_pi0_mem as _config_pi0_mem
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


CONFIG_NAME = "pi0_shellgame_v10_failure_suffix_overfit_probe_260820"
PROBE_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_v10_failure_suffix_overfit_probe_6ep_260820"
)
PROBE_AUDIT = PROBE_ROOT / "v10_real_onpolicy_oracle_supervision_audit.json"
DEFAULT_INIT_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/"
    "499/params"
)
EXPECTED_EPISODES = 6
XY_LOSS_WEIGHT = 4.0
ACTION_LOSS_MASK = (XY_LOSS_WEIGHT, XY_LOSS_WEIGHT, 1.0, 1.0, 1.0, 1.0, 1.0) + (0.0,) * 25


def _probe_data_config():
    return _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7(
        repo_id=str(PROBE_ROOT),
        # Preserve V10's nominal EEF normalization statistics.
        assets=_config.AssetsConfig(asset_id=".", assets_dir=str(_v2.NOMINAL_ROOT)),
        base_config=_config.UmiDataConfig(
            action_loss_mask=ACTION_LOSS_MASK,
            robot_type="ARM=1 G=0 H=0",
        ),
        num_frames=_full_joint.TOTAL_INPUT_FRAMES,
        frame_stride=1,
    )


def _oracle_suffix_indices(dataset, indices: list[int], classifier_config) -> list[int]:
    del classifier_config
    hf = _full_joint._find_hf_dataset(dataset)
    selected = np.asarray(indices, dtype=np.int64)
    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)
    action_mask = np.asarray(hf["action_mask"], dtype=bool)
    keep = action_mask[selected] & (frame[selected] >= 60) & (frame[selected] <= 154)
    emitted = selected[keep]
    present_episodes = np.unique(episode[selected])
    emitted_counts = {
        int(ep): int(np.count_nonzero(episode[emitted] == ep)) for ep in present_episodes
    }
    if not emitted_counts or any(count != 95 for count in emitted_counts.values()):
        raise RuntimeError(
            "Probe requires all 95 consecutive Oracle-aligned rows per split episode; "
            f"got {emitted_counts}"
        )
    logging.info(
        "Exact-state Oracle suffix sampler: input=%d emitted=%d per_episode=%s",
        len(selected),
        len(emitted),
        emitted_counts,
    )
    return emitted.tolist()


def _validate_data() -> None:
    if not PROBE_ROOT.is_dir():
        raise FileNotFoundError(PROBE_ROOT)
    if not pathlib.Path(DEFAULT_INIT_CHECKPOINT).is_dir():
        raise FileNotFoundError(DEFAULT_INIT_CHECKPOINT)
    if not PROBE_AUDIT.is_file():
        raise FileNotFoundError(PROBE_AUDIT)
    audit = json.loads(PROBE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("ok") is not True:
        raise ValueError(f"Probe conversion audit failed: {audit}")
    raw = audit.get("raw_audit", {})
    if raw.get("episodes") != EXPECTED_EPISODES:
        raise ValueError(f"Probe episode count mismatch: {raw}")
    if raw.get("model_generated_actions_supervised") is not False:
        raise ValueError("Probe audit does not prove Oracle-only supervision")


def build_config(args):
    parent = _v2.build_config(args)
    model = dataclasses.replace(parent.model, action_loss_mask=ACTION_LOSS_MASK)
    data = _config.MultiDataConfigFactory(
        state_pad_dim=96,
        datasets=[_probe_data_config()],
        weights=[1.0],
        use_merged_norm_stats=False,
    )
    return dataclasses.replace(
        parent,
        name=CONFIG_NAME,
        exp_name=args.exp_name,
        model=model,
        data=data,
        freeze_filter=model.get_freeze_filter_full_action(),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.peak_lr * 0.1,
        ),
        num_train_steps=args.steps,
        # Six duplicated episodes -> one held-out duplicate and five train episodes.
        val_ratio=0.17,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        resume=False,
    )


def main() -> None:
    args = _full_joint.parse_args()
    if args.init_checkpoint == str(_full_joint.OLD_QUERY_ACTION_CHECKPOINT):
        args.init_checkpoint = DEFAULT_INIT_CHECKPOINT
    if not 100 <= args.steps <= 500:
        raise ValueError("Keep the exact-state probe between 100 and 500 steps")
    _validate_data()
    logging.info(
        "Exact-state overfit: episodes=%d all_oracle_rows=true xy_weight=%.1f "
        "gripper_weight=%.1f init=%s tracker+memory=frozen",
        EXPECTED_EPISODES,
        XY_LOSS_WEIGHT,
        args.gripper_loss_weight,
        args.init_checkpoint,
    )
    _config_pi0_mem.VideoFrameDataset = _full_joint.FixedPrefixCurrentVideoDataset
    _trainer._filter_memory_classifier_frame_range = _oracle_suffix_indices
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()
