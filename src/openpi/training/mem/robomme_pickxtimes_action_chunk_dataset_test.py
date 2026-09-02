import numpy as np

from openpi.training.mem import robomme_pickxtimes_action_chunk_dataset as chunk_data


def test_chunk_normalization_ignores_padded_targets():
    arrays = chunk_data.ActionChunkArrays(
        visual_features=np.zeros((1, 1152), dtype=np.float16),
        robot_goal=np.zeros((1, 19), dtype=np.float32),
        poses=np.asarray([[[1.0] * 6, [100.0] * 6]], dtype=np.float32),
        close_targets=np.zeros((1, 2), dtype=np.float32),
        action_mask=np.asarray([[1.0, 0.0]], dtype=np.float32),
        phase_targets=np.zeros((1,), dtype=np.int32),
        memory_bank=np.zeros((1, 128, 64), dtype=np.float16),
        memory_indices=np.zeros((1,), dtype=np.int32),
        episode_indices=np.zeros((1,), dtype=np.int32),
        timesteps=np.zeros((1,), dtype=np.int32),
    )
    stats = chunk_data.compute_normalization(arrays)
    assert np.array_equal(stats.pose_mean, np.ones(6))
    assert np.all(stats.pose_std == 1e-4)


def test_pool_spatial_patch_tokens_preserves_quadrants():
    grid = np.arange(16 * 16, dtype=np.float32).reshape(1, 256, 1)
    pooled = chunk_data.pool_spatial_patch_tokens(grid)
    expected_first = np.mean(np.arange(4)[:, None] * 16 + np.arange(4)[None, :])
    assert pooled.shape == (1, 16, 1)
    assert np.isclose(pooled[0, 0, 0], expected_first)
