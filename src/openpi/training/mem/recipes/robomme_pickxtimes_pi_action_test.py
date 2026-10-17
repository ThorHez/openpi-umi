import json

import h5py
import numpy as np

from openpi.training.mem.recipes import robomme_pickxtimes_pi_action


def test_frozen_memory_lookup_is_causal(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "source_episode_name": "episode_7"}) + "\n",
        encoding="utf-8",
    )
    memory_path = tmp_path / "memory.h5"
    with h5py.File(memory_path, "w") as output:
        group = output.create_group("episode_7")
        group.create_dataset("initial_memory", data=np.zeros((4, 2), dtype=np.float32))
        predicted = group.create_group("predicted")
        predicted.create_dataset("stage_memories", data=np.stack((np.ones((4, 2)), np.full((4, 2), 2))))
        predicted.create_dataset("visible_timesteps", data=np.asarray([10, 20], dtype=np.int32))
    lookup = robomme_pickxtimes_pi_action.FrozenMemoryLookup.load(root, memory_path, "predicted")
    np.testing.assert_array_equal(lookup.at(0, 9), 0)
    np.testing.assert_array_equal(lookup.at(0, 10), 1)
    np.testing.assert_array_equal(lookup.at(0, 20), 2)


def test_action_only_lookup_is_zero(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "source_episode_name": "episode_7"}) + "\n",
        encoding="utf-8",
    )
    memory_path = tmp_path / "memory.h5"
    with h5py.File(memory_path, "w") as output:
        group = output.create_group("episode_7")
        group.create_dataset("initial_memory", data=np.ones((4, 2), dtype=np.float32))
        predicted = group.create_group("predicted")
        predicted.create_dataset("stage_memories", data=np.ones((1, 4, 2), dtype=np.float32))
        predicted.create_dataset("visible_timesteps", data=np.asarray([10], dtype=np.int32))
    lookup = robomme_pickxtimes_pi_action.FrozenMemoryLookup.load(root, memory_path, "action_only")
    np.testing.assert_array_equal(lookup.at(0, 100), 0)
