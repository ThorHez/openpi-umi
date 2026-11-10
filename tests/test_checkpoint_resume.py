"""Exercise native resume against real Orbax checkpoints on CPU."""

import dataclasses
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
optax = pytest.importorskip("optax")
pytest.importorskip("orbax.checkpoint")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openpi.training import checkpoints
from openpi.training.utils import TrainState


def _initial_state(*, ema):
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    params = nnx.state(model)
    schedule = optax.linear_schedule(0.02, 0.001, 10)
    tx = optax.adam(schedule)
    return TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params),
        tx=tx,
        ema_decay=0.9 if ema else None,
        ema_params=params if ema else None,
    )


def _update(state):
    grads = jax.tree.map(lambda value: jnp.ones_like(value) * (state.step + 1), state.params)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = (None if state.ema_params is None else
           jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, state.ema_params, params))
    return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state, ema_params=ema)


def _assert_state_equal(actual, expected):
    for field in ("step", "params", "opt_state", "ema_params"):
        values = (getattr(actual, field), getattr(expected, field))
        assert jax.tree.structure(values[0]) == jax.tree.structure(values[1])
        for left, right in zip(*(jax.tree.leaves(value) for value in values), strict=True):
            np.testing.assert_array_equal(left, right)


def _manifest(root):
    return {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("ema", [False, True])
def test_resume_selected_checkpoint_preserves_optimizer_and_source(tmp_path, ema):
    source = tmp_path / "previous" / "platform"
    output = tmp_path / "new" / "platform"
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None))
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        previous = _update(_initial_state(ema=ema))
        later = _update(previous)
        writer, _ = checkpoints.initialize_checkpoint_dir(source, keep_period=1, overwrite=False, resume=False)
        with writer:
            checkpoints.save_state(writer, previous, loader, 0)
            writer.wait_until_finished()
            checkpoints.save_state(writer, later, loader, 1)
        manifest = _manifest(source)
        selected = checkpoints.resolve_resume_checkpoint(str(source / "0"), output, target_steps=5)
        template = jax.eval_shape(lambda: _initial_state(ema=ema))
        native_manager, native_resuming = checkpoints.initialize_checkpoint_dir(
            source, keep_period=1, overwrite=False, resume=True,
        )
        assert native_resuming
        with native_manager:
            _assert_state_equal(checkpoints.restore_state(native_manager, template, loader), later)
        device_sharding = jax.sharding.SingleDeviceSharding(device)
        sharding = jax.tree.map(lambda _: device_sharding, template)
        restored = checkpoints.restore_state_from_checkpoint(selected, template, loader, sharding=sharding, target_steps=5)
        assert all(value.sharding == device_sharding for value in jax.tree.leaves(restored))
        _assert_state_equal(restored, previous)

        continued = _update(restored)
        _assert_state_equal(continued, later)
        writer, resuming = checkpoints.initialize_checkpoint_dir(output, keep_period=None, overwrite=False, resume=False)
        assert not resuming
        with writer:
            checkpoints.save_state(writer, continued, loader, 1)
        assert (output / "1" / "train_state").is_dir()
        assert _manifest(source) == manifest


def test_resume_rejects_overlapping_output_and_exhausted_target(tmp_path):
    source = tmp_path / "previous" / "platform" / "12"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="input and output directories must be separate"):
        checkpoints.resolve_resume_checkpoint(str(source), source.parent, target_steps=14)
    with pytest.raises(ValueError, match="target total steps must exceed"):
        checkpoints.resolve_resume_checkpoint(str(source), tmp_path / "new", target_steps=13)
