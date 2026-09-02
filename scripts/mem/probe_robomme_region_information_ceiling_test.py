from __future__ import annotations

import numpy as np

from openpi.tasks.robomme import unified_gt_teacher as contract
from scripts.mem.probe_robomme_region_information_ceiling import rollout_episode


def test_symbolic_rollout_tracks_color_and_ordered_regions() -> None:
    fields = len(contract.STATE_FIELDS)
    data = {
        "task_id": np.asarray([1, 2]),
        "goal_color_ids": np.asarray([[1, 0], [2, 0]]),
        "queried_ordinal": np.asarray([0, 2]),
        "event_ids": np.asarray(
            [
                [1, 3, 0],
                [5, 5, 3],
            ]
        ),
        "entity_ids": np.asarray([[1, 0, 0], [0, 0, 0]]),
        "region_a_ids": np.asarray([[1, 1, 0], [1, 2, 1]]),
        "region_b_ids": np.asarray([[0, 3, 0], [0, 0, 2]]),
        "step_mask": np.asarray([[True, True, False], [True, True, True]]),
        "state_targets": np.zeros((2, 4, fields), dtype=np.int32),
    }
    color = rollout_episode(data, 0, use_swap_regions=True)
    ordered = rollout_episode(data, 1, use_swap_regions=True)
    red_field = contract.STATE_FIELDS.index("red_cell")
    ordinal_0 = contract.STATE_FIELDS.index("ordered_cell_0")
    ordinal_1 = contract.STATE_FIELDS.index("ordered_cell_1")
    assert color[-1, red_field] == 3
    assert ordered[-1, ordinal_0] == 2
    assert ordered[-1, ordinal_1] == 1

