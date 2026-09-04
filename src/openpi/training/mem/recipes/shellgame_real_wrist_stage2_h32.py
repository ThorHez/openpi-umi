"""32-step variant of the real ShellGame deployment action contract."""

from __future__ import annotations

import dataclasses

from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _h16

DATASET_ROOT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10_h32"
)
HISTORY_FRAMES = _h16.HISTORY_FRAMES
CURRENT_START_FRAME = _h16.CURRENT_START_FRAME
ACTION_HORIZON = 32
ACTION_DIM = _h16.ACTION_DIM
REAL_SWAP_FRAME_INDICES = _h16.REAL_SWAP_FRAME_INDICES

# The data transform is horizon-agnostic: the parquet feature and model config
# supply the chunk length. Reusing it keeps the H16/H32 comparison identical in
# every other aspect of the training/deployment interface.
data_config_type = _h16.data_config_type

MODEL_CONFIG = dataclasses.replace(_h16.MODEL_CONFIG, action_horizon=ACTION_HORIZON)
