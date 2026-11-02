import dataclasses
import enum
import logging
import socket
from typing import Literal

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import rtc_policy as _rtc_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Optional serving adapter for prepared single-arm elec_meta observations.
    profile: Literal["default", "elec-meta"] = "default"

    # RTC server mode returns one action per controller tick. Client mode
    # enables kai0's whole-chunk protocol with client-managed execution.
    rtc: _rtc_policy.RTCConfig = dataclasses.field(default_factory=_rtc_policy.RTCConfig)

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    sample_kwargs: dict | None = None,
    client_rtc: bool = False,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
            client_rtc=client_rtc,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.rtc.enabled and args.rtc.num_steps <= 0:
        raise ValueError("RTC num_steps must be positive")
    if args.profile == "elec-meta" and (
        not args.rtc.enabled or args.rtc.mode != "client" or not isinstance(args.policy, Checkpoint)
    ):
        raise ValueError("elec-meta profile requires --rtc.enabled --rtc.mode client and policy:checkpoint")
    sample_kwargs = {"num_steps": args.rtc.num_steps} if args.rtc.enabled else None
    client_rtc = args.rtc.enabled and args.rtc.mode == "client"
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.profile == "elec-meta":
                from openpi.policies.elec_meta_config import adapt_elec_meta_config

                train_config = adapt_elec_meta_config(train_config)
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
                client_rtc=client_rtc,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
                client_rtc=client_rtc,
            )


def main(args: Args) -> None:
    policy = create_policy(args)

    if args.rtc.enabled and args.rtc.mode == "server":
        policy = _rtc_policy.RealTimeChunkingPolicy(policy, args.rtc)

    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        **({"ping_interval": None} if args.profile == "elec-meta" else {}),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
