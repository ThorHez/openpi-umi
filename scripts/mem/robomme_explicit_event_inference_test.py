import jax
import numpy as np
import pytest

from scripts.mem.robomme_explicit_event_inference import _chunks_from_frames
from scripts.mem.robomme_explicit_event_inference import _chunks_from_frames_gpu


@pytest.mark.parametrize("frame_count", [12, 13, 60])
def test_gpu_grid_chunks_match_cpu_reference(frame_count: int) -> None:
    frames = np.random.default_rng(frame_count).integers(0, 256, size=(frame_count, 32, 40, 3), dtype=np.uint8)

    cpu_chunks = _chunks_from_frames(frames)
    device_chunks = np.asarray(jax.device_get(_chunks_from_frames_gpu(frames)))

    np.testing.assert_allclose(device_chunks, cpu_chunks, rtol=0.0, atol=5e-4)
