# Normalization statistics

Following common practice, our models normalize the proprioceptive state inputs and action targets during policy training and inference. The statistics used for normalization are computed over the training data and stored alongside the model checkpoint.

## One shared distribution for several datasets

To compute shared statistics from three datasets with per-sample repeat weights
`1:2:2`, configure a `MultiDataConfigFactory` with `weights=[1.0, 2.0, 2.0]` and
an independent `shared_norm_stats_path`. The path may name `norm_stats.json` or
its containing directory. For example, retain the existing child dataset
factories and add the shared destination to the multi-dataset configuration:

```python
MultiDataConfigFactory(
    datasets=[dataset_a_factory, dataset_b_factory, dataset_c_factory],
    weights=[1.0, 2.0, 2.0],
    state_pad_dim=32,
    use_merged_norm_stats=True,
    shared_norm_stats_path="/storage/shared_norms/three_datasets_1_2_2/norm_stats.json",
)
```

Compute the shared asset using that training configuration:

```bash
python scripts/compute_norm_stats.py \
  --config-name YOUR_MULTI_DATASET_CONFIG \
  --output-dir /storage/shared_norms/three_datasets_1_2_2
```

For a configuration exported from the platform, use its local Python file
directly so the selected datasets, repeat weights, processors and overrides
come from the exported configuration:

```bash
python scripts/compute_norm_stats.py \
  --config-file /path/to/training_config.py \
  --output-dir /storage/shared_norms/three_datasets_1_2_2
```

Select exactly one of `--config-name` and `--config-file`. The file entry point
requires an explicit independent `--output-dir` and a synchronous callable
`build_config(output_dir)`. It imports the explicitly selected local Python
module and calls that builder with a configuration-only placeholder path
`OUTPUT_DIR/.config-only-output`; it does not create the placeholder or invoke
the module's training `main`. It does not download a configuration module. Run
it in the registered OpenPI/platform adapter environment required by the
exported file. Source compatibility, full coverage, provenance and output
protection checks are identical for both entry points.

The computation reads **every original row from every dataset** twice. It applies
the same repack and dataset transforms as training, then measures `state` and
`actions` before normalization and model padding. Both small datasets and the
last incomplete batch are included. The action horizons produced by the
training dataset loader contribute all their action vectors, including its
existing episode-boundary behavior. Image decoding required by a processor can
still run even though images do not contribute to statistics. This command does
not run model inference or training.

Each original row from source `i` has weight `wi`. Thus the source's total mass
is `Ni * wi`, where `Ni` is its row count. With integer weights, the resulting
mean and population standard deviation match statistics on the concatenation
of datasets repeated that many times; no repeated dataset copies are created.
Positive fractional weights are also supported without rounding. The source
weights are internally divided by their common maximum for numerical stability;
this does not change the distribution, and provenance retains the original
weights. Relative weights too small to represent in float64 are rejected.
The source
proportions are **not fixed to 20%, 40%, 40%** unless the three row counts match.
The training sampler uses these same per-row weights with replacement, so its
source proportions approach `Ni * wi / sum(Nj * wj)` over many draws. A training
epoch does not deterministically visit each row `wi` times.

Quantiles come from the pooled raw vectors, never an average of previously
computed per-dataset quantiles. The first pass computes float64 weighted moments
and global extrema; the second pass accumulates weighted counts into 5000 fixed
bins over those extrema. `q01` and `q99` use the lower edge of the bin reaching
1% or 99% cumulative mass. They approximate weighted inverse-CDF quantiles to
one bin and are not exact NumPy linear-interpolation quantiles. Constant
dimensions retain their exact constant value. Shared `NormStats` records carry
`quantile_method="q01_q99"` to distinguish the quantile pair used by transforms.

With quantile normalization enabled, these new artifacts use
`x_norm = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1`, even when diagnostic
`min/max` are present. The inverse uses the same bounds and epsilon. Explicit
normalization masks remain effective; model-only padding is never included in
the statistics. Unmarked legacy artifacts retain their existing normalization
behavior, so old checkpoints are not silently assigned a new scale. Training
and serving must both use a runtime that supports the new artifact marker.

The output is one `norm_stats.json` containing `state` and `actions` under
`norm_stats`, plus a top-level `shared_normalization` record. The provenance
records each source's path, weight, data and processor fingerprints, actual row
and vector counts, full coverage, and the weighted histogram method. The scan
rejects empty sources, non-finite values, incompatible feature dimensions or
processors, invalid weights, and data or processor drift. Source files are
fingerprinted before and after scanning, and both passes must return identical
transformed numeric values.

The shared scan requires finite LeRobot datasets and rejects `--max-frames`,
because a partial scan cannot claim full coverage. An explicit `--output-dir`
is required unless `shared_norm_stats_path` already supplies a destination.
It must be outside each input dataset and each child's old normalization asset
directory. Existing output is preserved unless `--overwrite` is supplied;
overwriting an input asset remains forbidden. Publication uses an atomic file
operation, so failed scans do not publish a partial statistics document.

Set `shared_norm_stats_path` to this asset for training. Every dataset then
uses that same distribution, and checkpoint assets retain its provenance for
inference. Existing single-dataset configurations remain supported; their
default output follows `AssetsConfig.assets_dir` and `asset_id` (falling back to
the configuration asset root and repo ID). Use an explicit asset ID or output
directory when the repo ID is an absolute local dataset path. For a single
LeRobot dataset, `--max-frames` caps the deterministic prefix of original rows;
omit it to include the full dataset. Existing single-source RLDS dataset
batching remains unchanged.

## Reloading normalization statistics

### Select the training formula explicitly

Set `TrainConfig.normalization_method` to `"minmax"`, `"quantile"`, or
`"zscore"` to apply one formula to all resolved dataset configurations:

```python
TrainConfig(
    # ... the same dataset, model and shared statistic bindings ...
    normalization_method="minmax",  # or "quantile" / "zscore"
)
```

For Min-Max, `lo=min` and `hi=max`; for quantile normalization, `lo=q01`
and `hi=q99`. Both use `2 * (x - lo) / (hi - lo + 1e-6) - 1` without clipping.
Z-score uses `(x - mean) / (std + 1e-6)`. Explicit choices require their own
statistics and never fall back to other bounds or the file's quantile marker.
Normalization masks still preserve excluded dimensions, and inverse transforms
use the same formula, bounds and epsilon while retaining model-only padding.

The selected method is saved in each checkpoint normalization artifact. Serving
restores it when the supplied recipe does not specify a method, and rejects a
conflicting explicit choice before loading model weights. Portable policy
configuration retains the same choice. `None` keeps the legacy bool and artifact
rules, so historical recipes are not silently reinterpreted.

Switching only this consumer formula does not change raw-sample statistics, and
can reuse a shared artifact containing both bound pairs after a fresh preflight.
Changing data, preprocessing or source implementation still invalidates its
provenance. In particular, upgrading older code may require recomputing an old
artifact whose implementation hashes no longer match; hashes are not bypassed.

When you fine-tune one of our models on a new dataset, you need to decide whether to (A) reuse existing normalization statistics or (B) compute new statistics over your new training data. Which option is better for you depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. Below, we list all the available pre-training normalization statistics for each model.

**If your target robot matches one of these pre-training statistics, consider reloading the same normalization statistics.** By reloading the normalization statistics, the actions in your dataset will be more "familiar" to the model, which can lead to better performance. You can reload the normalization statistics by adding an `AssetsConfig` to your training config that points to the corresponding checkpoint directory and normalization statistics ID, like below for the `Trossen` (aka ALOHA) robot statistics of the `pi0_base` checkpoint:

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

For an example of a full training config that reloads normalization statistics, see the `pi0_aloha_pen_uncap` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

**Note:** To successfully reload normalization statistics, it's important that your robot + dataset are following the action space definitions used in pre-training. We provide a detailed description of our action space definitions below.

**Note #2:** Whether reloading normalization statistics is beneficial depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. We recommend to always try both, reloading and training with a fresh set of statistics computed on your new dataset (see [main README](../README.md) for instructions on how to compute new statistics), and pick the one that works better for your task.


## Provided Pre-training Normalization Statistics

Below is a list of all the pre-training normalization statistics we provide. We provide them for both, the `pi0_base` and `pi0_fast_base` models. For `pi0_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_base/assets` and for `pi0_fast_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_fast_base/assets`.
| Robot | Description | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF dual arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| Franka Emika (DROID) | 7-DoF arm with parallel gripper based on the DROID setup | droid |
| Franka Emika (non-DROID) | Franka FR3 arm with Robotiq 2F-85 gripper | franka |
| UR5e | 6-DoF UR5e arm with Robotiq 2F-85 gripper | ur5e |
| UR5e bi-manual | Bi-manual UR5e setup with Robotiq 2F-85 grippers | ur5e_dual |
| ARX | Bi-manual ARX-5 robot arm setup with parallel gripper | arx |
| ARX mobile | Mobile version of bi-manual ARX-5 robot arm setup mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with 2x ARX-5 arms | fibocom_mobile |


## Pi0 Model Action Space Definitions

Out of the box, both the `pi0_base` and `pi0_fast_base` use the following action space definitions (left and right are defined looking from behind the robot towards the workspace):
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

The proprioceptive state uses the same definitions as the action space, except for the base x-y position (the last two dimensions) for mobile robots, which we don't include in the proprioceptive state.

For 7-DoF robots (e.g. Franka), we use the first 7 dimensions of the action space for the joint actions, and the 8th dimension for the gripper action.

General info for Pi robots:
- Joint angles are expressed in radians, with position zero corresponding to the zero position reported by each robot's interface library, except for ALOHA, where the standard ALOHA code uses a slightly different convention (see the [ALOHA example code](../examples/aloha_real/README.md) for details).
- Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
- Control frequencies are either 20 Hz for UR5e and Franka, and 50 Hz for ARX and Trossen (ALOHA) arms.

For DROID, we use the original DROID action configuration, with joint velocity actions in the first 7 dimensions and gripper actions in the 8th dimension + a control frequency of 15 Hz.
