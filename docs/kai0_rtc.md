# kai0 RTC integration

This repository adapts kai0's inference-time real-time chunking (RTC) for existing Pi0 and Pi0.5 JAX policies. It returns whole chunks so the robot client can own execution, latency estimation, and plan replacement. Training continues to use the existing recipes and checkpoints.

For the single-arm electronic-component sorting deployment, use the dedicated [elec_meta RTC guide](elec_meta_rtc.md). It includes the serving profile, world-coordinate rebasing client, and a replay path that does not open robot hardware.

## Starting the server

```bash
uv run scripts/serve_policy.py --rtc.enabled --rtc.mode client \
  policy:checkpoint --policy.config=<existing_pi0_or_pi05_recipe> \
  --policy.dir=<jax_checkpoint_directory>
```

Use the same recipe used to train the checkpoint, including its data transforms and action layout. The serving adapter selects `Pi0RTCConfig`, retaining the Pi0/Pi0.5 architecture, original `PI0`/`PI05` model type, view selection, loss masks, and preprocessing. The checkpoint directory must contain JAX `params/`; PyTorch checkpoints and FAST/MEM/value/custom model families are unsupported by this adapter. No RTC-specific training recipe or retraining is needed.

The existing `--rtc.enabled` mode defaults to `--rtc.mode server`. That mode runs a background server worker and returns one action per request. Client mode returns a full chunk for each inference and leaves the execution clock with the robot client. Select one mode for an episode.

## Request and response

Add RTC fields at the top level of the ordinary observation dictionary. They are extracted before observation transforms and do not become model observation fields.

| Field | Meaning |
| --- | --- |
| `prev_action_chunk` | Previous plan aligned so row zero corresponds to the current observation time. Omit it on the first inference. |
| `prev_action_chunk_normalized` | `False` by default: the previous chunk is in the training action representation. Set `True` when supplying the returned `model_actions` buffer. |
| `inference_delay` | Predicted inference latency in controller ticks; default `0`. |
| `execute_horizon` | Exclusive end of the guided prefix of the new chunk; defaults to the valid previous-prefix length, or the model horizon when no previous chunk is supplied. |
| `enable_rtc` | Enables guidance when a previous chunk is supplied. Set `False` for ordinary unguided chunk sampling. |
| `mask_prefix_delay` | Optional overwrite of the delay prefix inside sampling; default `False`. |
| `prefix_attention_schedule` | `ones`, `zeros`, `linear`, or `exp`; default `exp`. These are guidance weights over action timesteps, not a change to transformer attention. |
| `max_guidance_weight` | Nonnegative guidance clipping limit; default `0.5`. Zero disables guidance. |

The server validates delay, execution horizon, chunk shape, and finite values. The required range is `0 <= inference_delay <= execute_horizon <= valid_previous_prefix_length <= prediction_horizon`, with `execute_horizon > 0`. When no previous chunk is supplied, the available length is the model horizon. Do not use a previous plan whose valid prefix has already been consumed.

Normalized previous chunks may have shorter temporal prefixes and action widths narrower than the model width; missing entries are padded and receive no guidance. Physical previous chunks must satisfy the training adapter's action-width requirements. A short physical prefix is temporarily extended by repeating its last action so adapters that require the full training horizon can process it, then trimmed back to its valid length before guidance.

Client mode uses `--rtc.num-steps` (default five) for denoising. The CLI option `--rtc.guidance-weight` applies only to server mode; client mode uses the per-request `max_guidance_weight`.

The serving policy pads previous chunks to a fixed model shape and compiles both the unguided and guided branches on the initial request, before returning the first plan. Changing the valid prefix length or action width then reuses that compiled sampler. Each prefix schedule is a static JAX option: warm up any additional schedules with representative observations before starting the controller's execution clock. Calling `Pi0RTC.sample_actions` directly bypasses the policy's fixed-shape preparation and requires its own warmup. Compilation latency must not be treated as an ordinary inference-delay estimate.

Responses contain:

- `actions`: the usual full chunk after output transforms, in the policy's deployment action representation.
- `model_actions`: the full normalized model output before output transforms, with shape `(prediction_horizon, model_action_dim)`. The usual model width is 32, but clients should read the actual array shape.
- The ordinary policy/server timing fields.

### Action coordinates

For recipes whose action coordinates remain stable between observations, use `model_actions` and `prev_action_chunk_normalized=True`. Keep the complete model width and shift only its temporal axis. This avoids mixing physical commands and normalized model outputs. `model_actions` is an inference buffer; execute `actions` through the normal robot control path.

For `prev_action_chunk_normalized=False`, previous actions pass through the configured training action preprocessing and normalization. They must already have the same encoding, units, frame, and raw layout as the training action input. This flag does not convert arbitrary controller outputs: joint commands produced by IK cannot be passed to an EEF policy, and transformed or smoothed execution commands may not be reversible into training coordinates. Recipes whose action preprocessing needs additional fields still need those fields in the observation.

For actions expressed relative to the current state or observation frame, a normalized old buffer may use an obsolete reference after the observation changes. Use the physical-action path for those recipes, supplying the remaining plan in its training input representation so the configured transforms can rebase it against the new observation. Normalized buffers are not automatically rebased across observations.

Do not send unnormalized `actions` with `prev_action_chunk_normalized=True`; the model would interpret physical values as normalized values. Reset the previous buffer at every episode boundary or policy/checkpoint change.

### Python request example

This example constructs requests only; it does not control hardware. The observation dictionaries must follow the selected recipe's normal input schema. The counters and delay estimate below must come from the deployment client's controller clock.

```python
import numpy as np
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
initial = client.infer(initial_observation)  # No previous chunk on the first call.

# Capture current_observation after this many actions of the initial plan.
# This normalized-buffer example assumes stable action coordinates.
executed_before_observation = 4
remaining = np.asarray(initial["model_actions"])[executed_before_observation:]
predicted_delay_ticks = 2
assert 0 <= predicted_delay_ticks <= len(remaining)

request = {
    **current_observation,
    "prev_action_chunk": remaining,
    "prev_action_chunk_normalized": True,
    "inference_delay": predicted_delay_ticks,
    "execute_horizon": len(remaining),
    "enable_rtc": True,
    "mask_prefix_delay": False,
    "prefix_attention_schedule": "exp",
    "max_guidance_weight": 0.5,
}
# In deployment, call infer in an inference worker while the controller keeps
# executing the old plan. Do not call it on the controller's blocking path.
next_plan = client.infer(request)

# After measuring actual elapsed controller ticks since current_observation:
pending_actions = np.asarray(next_plan["actions"])[actual_delay_ticks:]
pending_model_actions = np.asarray(next_plan["model_actions"])[actual_delay_ticks:]
```

For a relative-action recipe, replace the previous-chunk fields with the remaining physical plan in the training input representation and `prev_action_chunk_normalized=False`; retain the same controller timing and alignment rules.

## Execution and alignment

A request describes a new action plan starting at the observation's capture time. If the old chunk began at controller tick `t_old` and the new observation was taken at `t_obs`, shift the previous normalized chunk by `t_obs - t_old` before sending it. `inference_delay` estimates how many further controller ticks will pass before the new result becomes usable. `execute_horizon` marks where guidance toward that remaining old plan ends.

The client loop needs the following sequence:

1. Obtain the initial full chunk without a previous plan, then start executing its `actions`.
2. Capture a new observation while there are still enough old-plan actions remaining. Record the tick corresponding to that observation.
3. Shift the saved `model_actions` buffer to that tick and issue a new inference in a worker while the controller continues the old plan. Supply the estimated delay and a valid guidance cutoff.
4. When the result is available, measure the actual elapsed controller ticks since the observation, discard that many rows of both new `actions` and `model_actions`, and replace the pending plan. Use measured delay to update the next latency estimate.
5. Start another inference before the current valid plan runs out. Clear buffers at the next episode.

Dropping the predicted delay from the new chunk is insufficient when the actual delay differs. Track the controller tick clock and discard the actual elapsed prefix. A synchronous loop that stops execution while waiting for inference does not provide execution/inference overlap. The library does not add a robot scheduler or infer which actions the controller has executed.

### How the two modes differ

| Property | Server mode | kai0 client mode |
| --- | --- | --- |
| Response `actions` | One action, shape `(1, D)` | Full predicted chunk |
| Execution owner | One server request per controller tick | Robot client tracks its own tick clock |
| Next inference | Server background thread | Client issues an inference while executing the old plan |
| Delay estimate | Server counts tick requests during inference | Client supplies `inference_delay` and measures actual elapsed ticks |
| Horizon control | `min_execution_horizon`: actions consumed before starting inference; default `H//2` | `execute_horizon`: guidance cutoff in the new chunk |
| Default guidance limit | `5.0`, controlled by `--rtc.guidance-weight` | `0.5`, controlled by request `max_guidance_weight` |
| Prefix weights | Existing server RTC mask | kai0 `ones`/`zeros`/`linear`/`exp` schedules |

Both modes default to five denoising steps (`--rtc.num-steps`). Server mode additionally defaults to `initial_delay_steps=min(6, H//4)` and a 10-entry delay buffer. Its `initial_delay_steps <= min_execution_horizon <= H - initial_delay_steps` constraint is specific to its scheduling algorithm. Those scheduling defaults and horizon semantics should not be copied into the client payload.

## Source and adaptation

The source is the local kai0 checkout at `/Users/hezhili/code/kai0`, commit `9d93078c757840f50e75248c5c5a94ab7b41e13a`, under Apache-2.0. The relevant original implementation is:

- `src/openpi/models/pi0_rtc.py`: `get_prefix_weights` and flow sampling with VJP guidance.
- `src/openpi/models/pi0_config.py`: `Pi0RTCConfig` and checkpoint architecture compatibility.
- `src/openpi/policies/policy.py`: top-level previous-chunk and latency fields.
- `train_deploy_alignment/inference/arx/inference/arx_openpi_inference_rtc.py` and `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_rtc.py`: examples of client-owned asynchronous execution.

This adaptation shares the current repository's Pi0 model rather than duplicating kai0's older model architecture. It preserves the kai0 prefix schedules and guidance interface while completing field forwarding, making normalized buffers explicit, supporting general action widths and short prefixes, and validating request boundaries. The kai0 exponential schedule is `w * expm1(w) / (e - 1)`; the pre-existing server mode uses a different mask and retains its behavior.

The ROS packages, robot drivers, cameras, physical command publishing, and hardware-specific client scripts are outside this migration. Local interface and sampling checks do not establish GPU checkpoint performance or physical-robot behavior.
