from pathlib import Path

import numpy as np

from openpi.training.mem.recipes import shellgame_framesamp_v10_action_adapter as recipe


def test_framesamp_memory_lookup(tmp_path: Path):
    memories = np.arange(2 * 512 * 1024, dtype=np.float16).reshape(2, 512, 1024)
    np.save(tmp_path / "memory.npy", memories)
    np.save(tmp_path / "episode_index.npy", np.asarray([1, 3], dtype=np.int32))
    np.save(tmp_path / "final_label.npy", np.asarray([0, 2], dtype=np.int8))
    (tmp_path / "metadata.json").write_text(
        '{"format":"mme_framesamp_v10_bank_v1"}\n'
    )
    (tmp_path / "_COMPLETE").write_text("ok\n")

    lookup = recipe.FrameSampMemoryLookup.load(tmp_path)
    assert not lookup.has(0)
    assert lookup.has(1)
    assert lookup.has(3)
    assert lookup.at(3).shape == (512, 1024)
    np.testing.assert_array_equal(lookup.at(1), memories[0].astype(np.float32))


def test_framesamp_v10_model_contract():
    model = recipe.make_model_config()
    assert model.semantic_memory_tokens == 512
    assert model.semantic_memory_width == 1024
    assert model.parallel_semantic_adapter_enabled
    assert model.old_memory_condition_strength == 0.0
