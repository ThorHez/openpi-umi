"""Check RTC serving selection and checkpoint preprocessing without weights."""

import dataclasses
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as model_module
from openpi.models import pi0_config
import openpi.policies
from openpi.policies import policy as policy_module
from openpi.policies import rtc_policy
import openpi.training

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class _TrainConfig:
    model: pi0_config.Pi0Config
    data: object
    assets_dirs: tuple = ()
    policy_metadata: dict = dataclasses.field(default_factory=lambda: {"robot": "test"})


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect the defining module during class creation.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def factory(monkeypatch):
    # Exercise the real serving factory while keeping training/LeRobot imports
    # out of this checkpoint-free CPU test.
    config_module = types.ModuleType("openpi.training.config")
    config_module.TrainConfig = _TrainConfig
    checkpoints_module = types.ModuleType("openpi.training.checkpoints")
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, checkpoints_module.__name__, checkpoints_module)
    monkeypatch.setattr(openpi.training, "config", config_module, raising=False)
    monkeypatch.setattr(openpi.training, "checkpoints", checkpoints_module, raising=False)
    return _load_source("_kai0_test_policy_factory", ROOT / "src/openpi/policies/policy_config.py")


@pytest.fixture
def checkpoint_parts(factory, monkeypatch, tmp_path):
    calls = {}

    def create_data(_assets_dirs, model_config):
        calls["data_model_config"] = model_config
        return types.SimpleNamespace(
            robot_type=None,
            asset_id="robot",
            use_quantile_norm=False,
            normalize_masks=None,
            data_transforms=transforms.Group(),
            model_transforms=transforms.Group(),
        )

    def load_model(model_config, params):
        calls["loaded_config"] = model_config
        assert params == {"weights": "fixture"}
        return types.SimpleNamespace(action_horizon=model_config.action_horizon, action_dim=model_config.action_dim)

    def capture_policy(model, **kwargs):
        calls["policy_kwargs"] = kwargs
        return types.SimpleNamespace(model=model, **kwargs)

    monkeypatch.setattr(factory.download, "maybe_download", lambda _path: tmp_path)
    monkeypatch.setattr(model_module, "restore_params", lambda _path, **_kwargs: {"weights": "fixture"})
    monkeypatch.setattr(pi0_config.Pi0RTCConfig, "load", load_model)
    monkeypatch.setattr(policy_module, "Policy", capture_policy)
    return calls, types.SimpleNamespace(create=create_data)


def test_factory_preserves_all_pi0_configuration_and_original_recipe(factory, checkpoint_parts, tmp_path):
    calls, data = checkpoint_parts
    original = pi0_config.Pi0Config(
        pi05=True,
        action_dim=20,
        action_horizon=16,
        max_token_len=123,
        discrete_state_input=False,
        action_loss_mask=(1.0,) * 19 + (0.0,),
        image_keys=("left_wrist_0_rgb", "right_wrist_0_rgb"),
        depth_keys=("depth_0",),
        allow_all_masked_images=True,
    )
    train_config = _TrainConfig(model=original, data=data)
    policy = factory.create_trained_policy(train_config, tmp_path, client_rtc=True, norm_stats={})

    converted = calls["loaded_config"]
    assert type(converted) is pi0_config.Pi0RTCConfig
    assert calls["data_model_config"] is converted
    assert converted.model_type == original.model_type
    for field in dataclasses.fields(original):
        assert getattr(converted, field.name) == getattr(original, field.name), field.name
    assert train_config.model is original
    assert type(train_config.model) is pi0_config.Pi0Config
    assert policy.metadata["robot"] == "test"
    assert policy.metadata["rtc"]["mode"] == "client"
    assert "rtc" not in train_config.policy_metadata


def test_factory_rejects_derived_architecture_before_download(factory, monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class MemoryArchitectureConfig(pi0_config.Pi0Config):
        memory_tokens: int = 8

    def unexpected_download(_path):
        pytest.fail("Unsupported architectures must fail before opening a checkpoint")

    monkeypatch.setattr(factory.download, "maybe_download", unexpected_download)
    train_config = _TrainConfig(model=MemoryArchitectureConfig(), data=None)
    with pytest.raises(ValueError, match=r"only ordinary Pi0/Pi05"):
        factory.create_trained_policy(train_config, tmp_path, client_rtc=True)


@pytest.mark.parametrize("already_rtc", [False, True])
def test_factory_rejects_pytorch_checkpoint_before_loading(factory, monkeypatch, tmp_path, already_rtc):
    (tmp_path / "model.safetensors").write_bytes(b"not loaded")
    monkeypatch.setattr(factory.download, "maybe_download", lambda _path: tmp_path)
    model_config = pi0_config.Pi0RTCConfig() if already_rtc else pi0_config.Pi0Config()
    train_config = _TrainConfig(model=model_config, data=None)
    with pytest.raises(ValueError, match=r"requires a JAX checkpoint"):
        factory.create_trained_policy(train_config, tmp_path, client_rtc=not already_rtc)


def test_factory_physical_guidance_uses_training_delta_and_normalization(
    factory, checkpoint_parts, monkeypatch, tmp_path
):
    calls, data = checkpoint_parts
    seen = []

    def fixed_horizon_actions(inputs):
        assert inputs["actions"].shape == (8, 2)
        seen.append(inputs["actions"].copy())
        return inputs

    def create_data(_assets_dirs, _model_config):
        return types.SimpleNamespace(
            robot_type=None,
            asset_id="robot",
            use_quantile_norm=False,
            normalize_masks=None,
            data_transforms=transforms.Group(inputs=[fixed_horizon_actions, transforms.DeltaActions((True, False))]),
            model_transforms=transforms.Group(),
        )

    data.create = create_data
    stats = transforms.NormStats(mean=np.array([2.0, 4.0]), std=np.array([2.0, 3.0]))
    train_config = _TrainConfig(model=pi0_config.Pi0Config(action_horizon=8, action_dim=4), data=data)
    factory.create_trained_policy(train_config, tmp_path, client_rtc=True, norm_stats={"actions": stats})
    action_transform = calls["policy_kwargs"]["rtc_action_transform"]
    observation = {"state": np.array([3.0, 5.0], dtype=np.float32)}
    physical = np.array([[7.0, 10.0], [9.0, 13.0]], dtype=np.float32)
    normalized = action_transform(observation, physical)

    expected_delta = physical - [3.0, 0.0]
    expected = (expected_delta - stats.mean) / (stats.std + 1e-6)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6)
    assert normalized.shape == (2, 2)
    np.testing.assert_array_equal(seen[0][2:], np.repeat(physical[-1:], 6, axis=0))
    np.testing.assert_array_equal(physical, [[7.0, 10.0], [9.0, 13.0]])
    np.testing.assert_array_equal(observation["state"], [3.0, 5.0])


@pytest.fixture
def serving(factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", factory)
    monkeypatch.setattr(openpi.policies, "policy_config", factory, raising=False)
    module = _load_source("_kai0_test_serve_policy", ROOT / "scripts/serve_policy.py")
    # Tyro inspects the source of dataclass fields while constructing its CLI.
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("use_default", [False, True])
@pytest.mark.parametrize(
    ("enabled", "mode", "expected_client"),
    [(False, "server", False), (False, "client", False), (True, "server", False), (True, "client", True)],
)
def test_serve_routes_checkpoint_and_default_modes(
    serving, factory, monkeypatch, use_default, enabled, mode, expected_client
):
    calls = []
    training_config = object()
    policy = object()
    monkeypatch.setattr(openpi.training.config, "get_config", lambda _name: training_config, raising=False)

    def create_trained_policy(config, path, **kwargs):
        calls.append((config, path, kwargs))
        return policy

    monkeypatch.setattr(factory, "create_trained_policy", create_trained_policy)
    args = serving.Args(
        policy=serving.Default() if use_default else serving.Checkpoint(config="existing_recipe", dir="/checkpoint"),
        default_prompt="fold cloth",
        rtc=rtc_policy.RTCConfig(enabled=enabled, mode=mode, num_steps=7),
    )
    assert serving.create_policy(args) is policy
    assert len(calls) == 1
    assert calls[0][0] is training_config
    assert calls[0][2]["client_rtc"] is expected_client
    assert calls[0][2]["sample_kwargs"] == ({"num_steps": 7} if enabled else None)
    assert calls[0][2]["default_prompt"] == "fold cloth"


@pytest.mark.parametrize("mode", ["server", "client"])
@pytest.mark.parametrize("num_steps", [0, -1])
def test_invalid_rtc_denoising_steps_fail_before_checkpoint_loading(serving, factory, monkeypatch, mode, num_steps):
    def unexpected_load(*_args, **_kwargs):
        pytest.fail("Invalid denoising steps must fail before loading a checkpoint")

    monkeypatch.setattr(factory, "create_trained_policy", unexpected_load)
    args = serving.Args(
        policy=serving.Checkpoint(config="existing_recipe", dir="/checkpoint"),
        rtc=rtc_policy.RTCConfig(enabled=True, mode=mode, num_steps=num_steps),
    )
    with pytest.raises(ValueError, match=r"num_steps must be positive"):
        serving.create_policy(args)


def test_documented_client_mode_command_parses(serving):
    args = serving.tyro.cli(
        serving.Args,
        args=[
            "--rtc.enabled",
            "--rtc.mode",
            "client",
            "policy:checkpoint",
            "--policy.config=existing_pi05_recipe",
            "--policy.dir=/checkpoints/model",
        ],
    )
    assert args.rtc.enabled
    assert args.rtc.mode == "client"
    assert isinstance(args.policy, serving.Checkpoint)
    assert args.policy.config == "existing_pi05_recipe"
    assert args.policy.dir == "/checkpoints/model"


@pytest.mark.parametrize("mode", ["client", "server"])
def test_client_mode_serves_whole_chunk_policy_without_server_wrapper(serving, monkeypatch, mode):
    original = types.SimpleNamespace(metadata={"rtc": {"mode": mode}})
    wrapped = types.SimpleNamespace(metadata={"rtc": {"mode": "server"}})
    calls = {"wrapped": 0}
    monkeypatch.setattr(serving, "create_policy", lambda _args: original)
    monkeypatch.setattr(serving.socket, "gethostname", lambda: "test")
    monkeypatch.setattr(serving.socket, "gethostbyname", lambda _host: "127.0.0.1")

    def wrap(policy, _config):
        assert policy is original
        calls["wrapped"] += 1
        return wrapped

    class Server:
        def __init__(self, **kwargs):
            calls["server"] = kwargs

        def serve_forever(self):
            calls["served"] = True

    monkeypatch.setattr(rtc_policy, "RealTimeChunkingPolicy", wrap)
    monkeypatch.setattr(serving.websocket_policy_server, "WebsocketPolicyServer", Server)
    serving.main(serving.Args(rtc=rtc_policy.RTCConfig(enabled=True, mode=mode)))
    assert calls["wrapped"] == (1 if mode == "server" else 0)
    assert calls["server"]["policy"] is (wrapped if mode == "server" else original)
    assert calls["served"] is True
