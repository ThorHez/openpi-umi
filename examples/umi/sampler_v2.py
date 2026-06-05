from typing import Optional, Dict, Any
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st


# 默认配置（当不提供 dataset_config 时使用）
DEFAULT_LOWDIM_KEYS = [
    'robot0_eef_rot_axis_angle',
    'robot0_gripper_width',
    'robot0_eef_pos',
    "robot0_demo_start_pose",

    'robot1_eef_rot_axis_angle',
    'robot1_gripper_width',
    'robot1_eef_pos',
    "robot1_demo_start_pose",
]

DEFAULT_RGB_KEYS = ['camera0_rgb', 'camera1_rgb']

DEFAULT_KEY_HORIZON = {
    'action': 16,
    'camera0_rgb': 2,
    'robot0_eef_rot_axis_angle': 2,
    'robot0_gripper_width': 2,
    'robot0_eef_pos': 2,
    'robot0_demo_start_pose': 2,
    'camera1_rgb': 2,
    'robot1_eef_rot_axis_angle': 2,
    'robot1_gripper_width': 2,
    'robot1_eef_pos': 2,
    'robot1_demo_start_pose': 2,
}

DEFAULT_KEY_LATENCY_STEPS = {
    'action': 0,
    'camera0_rgb': 0,
    'robot0_eef_rot_axis_angle': 0,
    'robot0_gripper_width': 0,
    'robot0_eef_pos': 0,
    'robot0_demo_start_pose': 0,
    'camera1_rgb': 0,
    'robot1_eef_rot_axis_angle': 0,
    'robot1_gripper_width': 0,
    'robot1_eef_pos': 0,
    'robot1_demo_start_pose': 0,
}

DEFAULT_KEY_DOWN_SAMPLE_STEPS = {
    'action': 3,
    'camera0_rgb': 3,
    'robot0_eef_rot_axis_angle': 3,
    'robot0_gripper_width': 3,
    'robot0_eef_pos': 3,
    'robot0_demo_start_pose': 3,
    'camera1_rgb': 3,
    'robot1_eef_rot_axis_angle': 3,
    'robot1_gripper_width': 3,
    'robot1_eef_pos': 3,
    'robot1_demo_start_pose': 3,
}


def build_sampler_config_from_dataset_config(dataset_config: Dict[str, Any]) -> tuple:
    """
    从 dataset_config 构建 sampler 所需的配置
    """
    dataset = dataset_config.get("dataset", {})
    low_dim_obs_horizon = dataset.get("low_dim_obs_horizon", 2)
    img_obs_horizon = dataset.get("img_obs_horizon", 2)
    obs_down_sample_steps = dataset.get("obs_down_sample_steps", 3)
    action_horizon = dataset.get("action_horizon", 16)

    enabled_robot_ids = set()
    for robot in dataset_config.get("robots", []):
        if robot.get("enabled", False):
            enabled_robot_ids.add(robot["id"])

    lowdim_keys = []
    rgb_keys = []
    key_horizon = {"action": action_horizon}
    key_latency_steps = {"action": 0}
    key_down_sample_steps = {"action": obs_down_sample_steps}

    for robot_id in enabled_robot_ids:
        robot_keys = [
            f'robot{robot_id}_eef_rot_axis_angle',
            f'robot{robot_id}_gripper_width',
            f'robot{robot_id}_eef_pos',
            f'robot{robot_id}_demo_start_pose',
            f'robot{robot_id}_eef_rot_axis_angle_wrt_start',
        ]
        for key in robot_keys:
            key_horizon[key] = low_dim_obs_horizon
            key_latency_steps[key] = 0
            key_down_sample_steps[key] = obs_down_sample_steps

        for other_robot_id in enabled_robot_ids:
            if other_robot_id != robot_id:
                rel_keys = [
                    f'robot{robot_id}_eef_pos_wrt{other_robot_id}',
                    f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}',
                ]
                for key in rel_keys:
                    key_horizon[key] = low_dim_obs_horizon
                    key_latency_steps[key] = 0
                    key_down_sample_steps[key] = obs_down_sample_steps

    lowdim_keys = list(dataset_config.get("load_keys", []))
    del_camera_keys = []
    for key in lowdim_keys:
        if key.startswith('camera') or key.startswith('head_'):
            del_camera_keys.append(key)
    for key in del_camera_keys:
        lowdim_keys.remove(key)

    for key in lowdim_keys:
        if key not in key_horizon:
            key_horizon[key] = low_dim_obs_horizon
        if key not in key_latency_steps:
            key_latency_steps[key] = 0
        if key not in key_down_sample_steps:
            key_down_sample_steps[key] = obs_down_sample_steps

    # 单帧特征：不做多帧堆叠
    for key in dataset_config.get("single_frame_features", {}).keys():
        key_horizon[key] = 1
        key_latency_steps[key] = 0
        key_down_sample_steps[key] = 1

    for image_name, image_def in dataset_config.get("images", {}).items():
        source_key = image_def.get("source_key")
        if source_key:
            rgb_keys.append(source_key)
            key_horizon[source_key] = img_obs_horizon
            key_latency_steps[source_key] = 0
            key_down_sample_steps[source_key] = obs_down_sample_steps

    return lowdim_keys, rgb_keys, key_horizon, key_latency_steps, key_down_sample_steps


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(self,
                 replay_buffer: dict[str, np.ndarray],
                 episode_ends: list[int],
                 key_horizon: dict = None,
                 key_latency_steps: dict = None,
                 key_down_sample_steps: dict = None,
                 rgb_keys: list = None,
                 lowdim_keys: list = None,
                 episode_mask: Optional[np.ndarray] = None,
                 action_padding: bool = True,
                 repeat_frame_prob: float = 0.0,
                 max_duration: Optional[float] = None,
                 dataset_config: Optional[Dict[str, Any]] = None
                 ):
        if dataset_config is not None:
            config_lowdim_keys, config_rgb_keys, config_key_horizon, config_key_latency_steps, config_key_down_sample_steps = \
                build_sampler_config_from_dataset_config(dataset_config)
            lowdim_keys = lowdim_keys if lowdim_keys is not None else config_lowdim_keys
            rgb_keys = rgb_keys if rgb_keys is not None else config_rgb_keys
            key_horizon = key_horizon if key_horizon is not None else config_key_horizon
            key_latency_steps = key_latency_steps if key_latency_steps is not None else config_key_latency_steps
            key_down_sample_steps = key_down_sample_steps if key_down_sample_steps is not None else config_key_down_sample_steps
        else:
            lowdim_keys = lowdim_keys if lowdim_keys is not None else DEFAULT_LOWDIM_KEYS
            rgb_keys = rgb_keys if rgb_keys is not None else DEFAULT_RGB_KEYS
            key_horizon = key_horizon if key_horizon is not None else DEFAULT_KEY_HORIZON
            key_latency_steps = key_latency_steps if key_latency_steps is not None else DEFAULT_KEY_LATENCY_STEPS
            key_down_sample_steps = key_down_sample_steps if key_down_sample_steps is not None else DEFAULT_KEY_DOWN_SAMPLE_STEPS

        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True
            if episode_mask is not None and not episode_mask[i]:
                continue
            start_idx = 0 if i == 0 else episode_ends[i - 1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps[
                    'action'] + 1:
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))

        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key not in replay_buffer:
                print(f"Warning: lowdim key '{key}' not found in replay_buffer, skipping...")
                continue
            if key.endswith('eef_pos'):
                self.num_robot += 1
            self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]

        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)

        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps

        self.ignore_rgb_is_applied = False

    def __len__(self):
        return len(self.indices)

    def _interp_lowdim(self, input_arr, idxs, interpolation_start, interpolation_end, is_rot_axis_angle=False):
        """Helper: interpolate lowdim sequence at (float) idxs."""
        if is_rot_axis_angle:
            slerp = st.Slerp(
                times=np.arange(interpolation_start, interpolation_end),
                rotations=st.Rotation.from_rotvec(input_arr[interpolation_start: interpolation_end])
            )
            return slerp(idxs).as_rotvec()
        else:
            interp = si.interp1d(
                x=np.arange(interpolation_start, interpolation_end),
                y=input_arr[interpolation_start: interpolation_end],
                axis=0, assume_sorted=True
            )
            return interp(idxs)

    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]
        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]

            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid

                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - i * this_downsample_steps + this_latency_steps for i in range(this_horizon)],
                    dtype=np.float32
                )[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)

                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                # Non-numeric features (e.g. string action_source) are sampled by index.
                if not np.issubdtype(input_arr.dtype, np.number):
                    output = input_arr[idx_with_latency.astype(np.int64)]
                elif this_horizon == 1:
                    output = input_arr[idx_with_latency.astype(np.int64)]
                elif 'rot' in key and key.endswith('axis_angle'):
                    output = self._interp_lowdim(
                        input_arr, idx_with_latency, interpolation_start, interpolation_end, is_rot_axis_angle=True
                    )
                elif 'rot' in key and key.endswith('quat'):
                    # keep old behavior if ever used
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=st.Rotation.from_quat(input_arr[interpolation_start: interpolation_end])
                    )
                    output = slerp(idx_with_latency).as_quat()
                else:
                    output = self._interp_lowdim(
                        input_arr, idx_with_latency, interpolation_start, interpolation_end, is_rot_axis_angle=False
                    )

            arr = np.array(output)
            # Keep non-numeric single-frame features (e.g. string action_source) as-is.
            if np.issubdtype(arr.dtype, np.number):
                arr = arr.astype(np.float32)
            result[key] = arr

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # =========================
        # action (FUTURE pose-trajectory) with interpolation smoothing
        # =========================
        input_arr = self.replay_buffer['action']  # shape [T, D] where D = num_robot*(3+3+1)=7*num_robot
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']

        # future indices (float) for k=0..H-1
        idx_future = np.array(
            [current_idx + k * action_down_sample_steps for k in range(action_horizon)],
            dtype=np.float32
        )
        idx_future = np.clip(idx_future, start_idx, end_idx - 1)

        interpolation_start = max(int(idx_future[0]) - 5, start_idx)
        interpolation_end = min(int(idx_future[-1]) + 2 + 5, end_idx)

        # If action is constructed by concatenating per-robot [pos(3), rotvec(3), grip(1)]
        D = input_arr.shape[-1]
        if D % 7 != 0:
            raise ValueError(f"Expect action dim multiple of 7 (pos3+rotvec3+grip1 per robot), got D={D}")
        num_robot = D // 7

        output = np.zeros((action_horizon, D), dtype=np.float32)

        for r in range(num_robot):
            off = r * 7
            pos_arr = input_arr[:, off:off + 3]
            rot_arr = input_arr[:, off + 3:off + 6]     # axis-angle (rotvec)
            grip_arr = input_arr[:, off + 6:off + 7]    # (T,1)

            pos_out = self._interp_lowdim(pos_arr, idx_future, interpolation_start, interpolation_end, is_rot_axis_angle=False)
            rot_out = self._interp_lowdim(rot_arr, idx_future, interpolation_start, interpolation_end, is_rot_axis_angle=True)
            grip_out = self._interp_lowdim(grip_arr, idx_future, interpolation_start, interpolation_end, is_rot_axis_angle=False)

            output[:, off:off + 3] = pos_out.astype(np.float32)
            output[:, off + 3:off + 6] = rot_out.astype(np.float32)
            output[:, off + 6:off + 7] = grip_out.astype(np.float32)

        # padding policy (keep same as original, but should rarely trigger due to clip)
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        else:
            # If idx_future got clipped and you still want "repeat last valid future" semantics,
            # we can enforce by detecting beyond-end indices. Here we mimic original padding:
            max_valid = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
            # indices that would be beyond max_valid-1 in original slicing:
            k_valid = int(np.ceil((max_valid - current_idx) / action_down_sample_steps))
            k_valid = max(1, min(action_horizon, k_valid))
            if k_valid < action_horizon:
                output[k_valid:] = output[k_valid - 1:k_valid]

        result['action'] = output
        return result

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply


def test_config():
    from dataset_config_loader import load_dataset_config
    dataset_config = load_dataset_config("/root/openpi-umi/examples/umi/bimanual_dataset_config.yaml")

    lowdim_keys, rgb_keys, key_horizon, key_latency_steps, key_down_sample_steps = build_sampler_config_from_dataset_config(
        dataset_config)
    print(f"lowdim_keys: {lowdim_keys}")
    print(f"rgb_keys: {rgb_keys}")
    print(f"key_horizon: {key_horizon}")
    print(f"key_latency_steps: {key_latency_steps}")
    print(f"key_down_sample_steps: {key_down_sample_steps}")


if __name__ == "__main__":
    test_config()
