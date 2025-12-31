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
    
    Args:
        dataset_config: 数据集配置字典
        
    Returns:
        (lowdim_keys, rgb_keys, key_horizon, key_latency_steps, key_down_sample_steps)
    """
    dataset = dataset_config.get("dataset", {})
    low_dim_obs_horizon = dataset.get("low_dim_obs_horizon", 2)
    img_obs_horizon = dataset.get("img_obs_horizon", 2)
    obs_down_sample_steps = dataset.get("obs_down_sample_steps", 3)
    action_horizon = dataset.get("action_horizon", 16)
    
    # 获取启用的机器人 ID
    enabled_robot_ids = set()
    for robot in dataset_config.get("robots", []):
        if robot.get("enabled", False):
            enabled_robot_ids.add(robot["id"])
    
    lowdim_keys = []
    rgb_keys = []
    key_horizon = {"action": action_horizon}
    key_latency_steps = {"action": 0}
    key_down_sample_steps = {"action": obs_down_sample_steps}
    
    # 为每个启用的机器人添加 lowdim keys
    for robot_id in enabled_robot_ids:
        robot_keys = [
            f'robot{robot_id}_eef_rot_axis_angle',
            f'robot{robot_id}_gripper_width',
            f'robot{robot_id}_eef_pos',
            f'robot{robot_id}_demo_start_pose',
            f'robot{robot_id}_eef_rot_axis_angle_wrt_start',
        ]
        for key in robot_keys:
            # lowdim_keys.append(key)
            key_horizon[key] = low_dim_obs_horizon
            key_latency_steps[key] = 0
            key_down_sample_steps[key] = obs_down_sample_steps
        
        # 添加相对于其他机器人的特征 (双臂时)
        for other_robot_id in enabled_robot_ids:
            if other_robot_id != robot_id:
                rel_keys = [
                    f'robot{robot_id}_eef_pos_wrt{other_robot_id}',
                    f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}',
                ]
                for key in rel_keys:
                    # lowdim_keys.append(key)
                    key_horizon[key] = low_dim_obs_horizon
                    key_latency_steps[key] = 0
                    key_down_sample_steps[key] = obs_down_sample_steps
    
    lowdim_keys = list(dataset_config.get("load_keys", []))
    del_camera_keys = []
    for key in lowdim_keys:
        if key.startswith('camera'):
            del_camera_keys.append(key)
    for key in del_camera_keys:
        lowdim_keys.remove(key)

    # 处理图像配置
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

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
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
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=True,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None,
        dataset_config: Optional[Dict[str, Any]]=None
    ):
        # 如果提供了 dataset_config，从中构建配置
        if dataset_config is not None:
            config_lowdim_keys, config_rgb_keys, config_key_horizon, config_key_latency_steps, config_key_down_sample_steps = \
                build_sampler_config_from_dataset_config(dataset_config)
            lowdim_keys = lowdim_keys if lowdim_keys is not None else config_lowdim_keys
            rgb_keys = rgb_keys if rgb_keys is not None else config_rgb_keys
            key_horizon = key_horizon if key_horizon is not None else config_key_horizon
            key_latency_steps = key_latency_steps if key_latency_steps is not None else config_key_latency_steps
            key_down_sample_steps = key_down_sample_steps if key_down_sample_steps is not None else config_key_down_sample_steps
        else:
            # 使用默认配置
            lowdim_keys = lowdim_keys if lowdim_keys is not None else DEFAULT_LOWDIM_KEYS
            rgb_keys = rgb_keys if rgb_keys is not None else DEFAULT_RGB_KEYS
            key_horizon = key_horizon if key_horizon is not None else DEFAULT_KEY_HORIZON
            key_latency_steps = key_latency_steps if key_latency_steps is not None else DEFAULT_KEY_LATENCY_STEPS
            key_down_sample_steps = key_down_sample_steps if key_down_sample_steps is not None else DEFAULT_KEY_DOWN_SAMPLE_STEPS
        # load gripper_width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True # initialize for each episode
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
            self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
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
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
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
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)
                
            result[key] = np.array(output).astype(np.float32)

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply


def test_config():
    from dataset_config_loader import load_dataset_config
    dataset_config = load_dataset_config("/root/openpi-umi/examples/umi/bimanual_dataset_config.yaml")

    lowdim_keys, rgb_keys, key_horizon, key_latency_steps, key_down_sample_steps = build_sampler_config_from_dataset_config(dataset_config)
    print(f"lowdim_keys: {lowdim_keys}")
    print(f"rgb_keys: {rgb_keys}")
    print(f"key_horizon: {key_horizon}")
    print(f"key_latency_steps: {key_latency_steps}")
    print(f"key_down_sample_steps: {key_down_sample_steps}")


if __name__ == "__main__":
    test_config()