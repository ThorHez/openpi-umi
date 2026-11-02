
# Running openpi models remotely

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

The `env` argument specifies which $\pi_0$ checkpoint should be loaded. Under the hood, this script will execute a command like the following, which you can use to start a policy server, e.g. for checkpoints you trained yourself (here an example for the DROID environment):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

This will start a policy server that will serve the policy specified by the `config` and `dir` arguments. The policy will be served on the specified port (default: 8000).

### Real-time chunking (RTC)

Flow-based pi0 and pi0.5 policies can use inference-time RTC. There are two execution modes; choose the one that owns the action clock in your deployment.

#### Server-controlled execution (default)

The existing server-controlled implementation follows [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339):

```bash
uv run scripts/serve_policy.py --env ALOHA --rtc.enabled
```

The defaults adapt to the model's prediction horizon: `min_execution_horizon=H//2` and `initial_delay_steps=min(6, H//4)`, with a 10-entry delay buffer, 5 denoising steps, and guidance clipping at 5. Thus the paper's `H=50` setup uses `s_min=25, d_init=6`, while this repository's common UMI `H=16` setup uses `s_min=8, d_init=4`. Override them with flags such as `--rtc.min-execution-horizon=6` and `--rtc.initial-delay-steps=3`. RTC requires `initial_delay_steps <= min_execution_horizon <= prediction_horizon - initial_delay_steps`.

In this mode (`--rtc.mode server`, the default), send the newest observation on every controller tick. Each response contains exactly one action as a chunk with shape `(1, action_dim)` while the next full chunk is generated in the server's background thread. Existing runtime examples can therefore use `ActionChunkBroker(policy=client, action_horizon=1)`. Do not buffer several returned actions before sending the next observation; doing so would remove the execution/inference overlap that makes RTC real-time.

RTC is opt-in; ordinary policy serving keeps its existing chunk output. Models must expose guided sampling explicitly. MEM models and other models without that interface are rejected when RTC is enabled. Sharing a source branch does not enable RTC for those models. Reconnecting clears the current action chunk and the delay history before the next episode.

#### Client-controlled execution (kai0)

The kai0 implementation returns a complete action chunk and accepts the previous plan and latency estimate in each request. The robot client continues executing the old plan while inference runs, then discards the already elapsed prefix of the new plan. Start this JAX-only mode with an existing Pi0/Pi0.5 training recipe and checkpoint:

```bash
uv run scripts/serve_policy.py --rtc.enabled --rtc.mode client \
  policy:checkpoint --policy.config=<existing_pi0_or_pi05_recipe> \
  --policy.dir=<jax_checkpoint_directory>
```

The checkpoint directory must contain `params/`. The serving adapter uses `Pi0RTCConfig` with the same architecture and model type as the original Pi0/Pi0.5 configuration. It preserves the recipe's action layout, preprocessing, views, and normalization; training recipes and checkpoint weights require no change. This mode does not support PyTorch checkpoints, FAST, MEM, or other model families.

Each response has the ordinary postprocessed `actions` plus a complete `model_actions` buffer in normalized model coordinates. When action coordinates remain stable between observations, shift that buffer by the number of old-plan actions already executed at the time of the new observation, then send it as `prev_action_chunk` with `prev_action_chunk_normalized=True`. For actions relative to the current observation/state, use the physical-action path so preprocessing can rebase them. Pass the estimated number of controller ticks that the new inference will take as `inference_delay` (default zero); `execute_horizon` is the end of the guided prefix of the new plan and defaults to the valid previous-prefix length. The four prefix schedules are `ones`, `zeros`, `linear`, and `exp` (default), with `max_guidance_weight=0.5` by default; zero disables guidance. Both modes use `--rtc.num-steps` (default five); `--rtc.guidance-weight` applies only to server mode.

`execute_horizon` in this mode is a guidance cutoff, whereas server mode's `min_execution_horizon` is the number of actions to execute before starting another inference. These parameters are not interchangeable. Server mode's horizon-based defaults and delay buffer apply only to server mode.

See [kai0 RTC integration](kai0_rtc.md) for request fields, action-coordinate rules, timing, and source provenance. The migration includes the library and serving protocol; ROS drivers and hardware-specific kai0 clients are not copied into this repository.

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](../examples/simple_client/main.py).
