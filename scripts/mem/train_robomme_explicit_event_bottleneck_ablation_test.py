from argparse import Namespace

import numpy as np

from scripts.mem import train_robomme_explicit_event_bottleneck_ablation as training


def _args(mode: str) -> Namespace:
    return Namespace(
        fixed_dir=training.base.DEFAULT_FIXED,
        teacher_dir=training.base.DEFAULT_TEACHER,
        feature_dir=training.DEFAULT_FEATURES,
        anchor_dir=training.parser_base.anchor_base.DEFAULT_ANCHORS,
        unmask_h5=training.DEFAULT_UNMASK_H5,
        unmask_binding_labels=mode,
    )


def test_native_single_corrects_noisy_regions_without_changing_goal_arity():
    data = training.NativeUnmaskBindingDataset("train", _args("native_single"))
    try:
        rows = data.rows[data.fixed["task_ids"][data.rows] == 0]
        assert data.native_binding_audit["corrected_original_targets"] == 12
        assert not np.any(data.fixed["goal_color_ids"][rows, 1])
        assert np.all(data.write_mask[rows, 0].sum(axis=-1) == 1)
    finally:
        data.close()


def test_native_full_restores_dual_prompt_and_dense_initial_bindings():
    data = training.NativeUnmaskBindingDataset("train", _args("native_full"))
    try:
        rows = data.rows[data.fixed["task_ids"][data.rows] == 0]
        assert data.native_binding_audit["dual_goal_episodes"] == 23
        dual = rows[data.fixed["goal_color_ids"][rows, 1] > 0]
        assert len(dual) == 23
        assert np.all(data.write_mask[dual, 0].sum(axis=-1) == 2)
        for row in dual:
            length = int(data.fixed["step_mask"][row].sum())
            colors = data.fixed["goal_color_ids"][row]
            fields = colors - 1
            assert np.all(data.table_mask[row, : length + 1, fields])
            assert np.all(data.table_targets[row, 1 : length + 1, fields] > 0)
            assert len(set(data.table_targets[row, 1, fields].tolist())) == 2
    finally:
        data.close()
