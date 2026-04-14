"""
数据集配置加载器

从 YAML 配置文件加载数据集特征定义
"""

from typing import Dict, Any, List, Tuple, Set
import yaml
import re

# pyarrow / HF datasets dtype aliases that need normalization.
# "str" is a NumPy alias, but pyarrow expects "string".
_DTYPE_ALIASES = {
    "str": "string",
    "str_": "string",
    "unicode": "string",
    "unicode_": "string",
}

# dtypes that pyarrow Array2D/3D/4D/5D extension types cannot store.
# For these we fall back to a compatible numeric type.
_ARRAY_ND_INCOMPATIBLE = {
    "bool": "int8",
}


def normalize_dtype(dtype: str, ndim: int = 0) -> str:
    """Normalize a config dtype to one that HF datasets / pyarrow accepts.

    Args:
        dtype: Raw dtype string from the YAML config.
        ndim: Number of dimensions of the feature shape.  When >= 2 the
              feature will be stored as Array2D+ which has extra restrictions.
    """
    if not isinstance(dtype, str):
        return dtype
    dtype = _DTYPE_ALIASES.get(dtype, dtype)
    if ndim >= 2:
        dtype = _ARRAY_ND_INCOMPATIBLE.get(dtype, dtype)
    return dtype


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """
    加载数据集配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_shape(shape_list: List, variables: Dict[str, int]) -> Tuple:
    """
    解析形状定义，替换变量
    
    Args:
        shape_list: 形状列表，可能包含变量如 "{seq_len}"
        variables: 变量字典，如 {"seq_len": 2, "action_horizon": 16}
        
    Returns:
        解析后的形状元组
    """
    parsed = []
    for item in shape_list:
        if isinstance(item, str) and item.startswith("{") and item.endswith("}"):
            var_name = item[1:-1]
            if var_name in variables:
                parsed.append(variables[var_name])
            else:
                raise ValueError(f"Unknown variable in shape: {item}")
        else:
            parsed.append(int(item))
    return tuple(parsed)


def extract_robot_ids_from_name(feature_name: str) -> Set[int]:
    """
    从特征名称中提取机器人 ID
    
    例如:
        "observation.robot0_eef_pos" -> {0}
        "observation.robot1_eef_pos" -> {1}
        "actions" -> set() (空集，通用特征)
    
    Args:
        feature_name: 特征名称
        
    Returns:
        机器人 ID 集合
    """
    # 匹配 robot 后面跟着数字的模式，如 robot0, robot1
    robot_pattern = re.compile(r'robot(\d+)')
    matches = robot_pattern.findall(feature_name)
    
    return {int(m) for m in matches}


def is_feature_enabled(feature_name: str, enabled_robot_ids: Set[int]) -> bool:
    """
    根据启用的机器人判断特征是否应该启用
    
    规则:
        - 如果特征名称中没有机器人 ID，则始终启用 (通用特征如 actions)
        - 如果特征名称中有机器人 ID，需要所有相关机器人都启用才启用该特征
    
    Args:
        feature_name: 特征名称
        enabled_robot_ids: 启用的机器人 ID 集合
        
    Returns:
        是否启用该特征
    """
    required_robots = extract_robot_ids_from_name(feature_name)
    
    # 没有机器人 ID 的特征始终启用
    if not required_robots:
        return True
    
    # 需要所有相关机器人都启用
    return required_robots.issubset(enabled_robot_ids)


def build_features_from_config(
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    根据配置构建特征定义
    
    特征启用规则:
        - 根据 robots 配置中启用的机器人自动决定特征是否启用
        - 特征名称中包含 robot0, robot1 等会自动关联对应机器人
        - 不包含机器人 ID 的通用特征 (如 actions) 始终启用
        - actions 特征的维度会根据启用的机器人数量自动扩展 (action_dim * num_robot)
    
    Args:
        config: 配置字典
        
    Returns:
        LeRobot 特征定义字典
    """
    # 获取启用的机器人 ID
    enabled_robot_ids = set(get_enabled_robots(config))
    num_robot = len(enabled_robot_ids)
    
    # 获取变量值
    dataset_config = config.get("dataset", {})
    low_dim_obs_horizon = dataset_config.get("low_dim_obs_horizon", 2)
    img_obs_horizon = dataset_config.get("img_obs_horizon", 2)
    act_horizon = dataset_config.get("action_horizon", 16)
    obs_down_sample_steps = dataset_config.get("obs_down_sample_steps", 3)
    action_dim_per_robot = dataset_config.get("action_dim", 10)
    
    # 当有多个机器人时，action_dim 需要乘以机器人数量
    total_action_dim = action_dim_per_robot * max(num_robot, 1)
    
    variables = {
        "low_dim_obs_horizon": low_dim_obs_horizon,
        "img_obs_horizon": img_obs_horizon,
        "obs_down_sample_steps": obs_down_sample_steps,
        "action_dim": total_action_dim,  # 使用总的 action_dim
        "action_horizon": act_horizon,
    }
    
    features = {}
    
    # 处理普通特征 + 单帧特征
    feature_sections = [config.get("features", {}), config.get("single_frame_features", {})]
    for section in feature_sections:
        for feature_name, feature_def in section.items():
        # 根据启用的机器人自动判断特征是否启用
            if not is_feature_enabled(feature_name, enabled_robot_ids):
                continue

            shape = parse_shape(feature_def["shape"], variables)
            dtype = normalize_dtype(feature_def["dtype"], ndim=len(shape))
            features[feature_name] = {
                "dtype": dtype,
                "shape": shape,
                "names": feature_def.get("names", [feature_name]),
            }
    
    # 处理图像特征
    for image_name, image_def in config.get("images", {}).items():
        # 根据启用的机器人自动判断特征是否启用
        if not is_feature_enabled(image_name, enabled_robot_ids):
            continue
            
        shape = tuple(image_def["shape"])
        
        img_dtype = normalize_dtype(image_def["dtype"], ndim=len(shape))

        if image_def.get("per_timestep", False):
            for i in range(img_obs_horizon):
                features[f"{image_name}_{i}"] = {
                    "dtype": img_dtype,
                    "shape": shape,
                    "names": ["height", "width", "channel"],
                }
        else:
            features[image_name] = {
                "dtype": img_dtype,
                "shape": shape,
                "names": ["height", "width", "channel"],
            }
    
    return features


def get_enabled_features(config: Dict[str, Any]) -> List[str]:
    """
    获取所有启用的特征名称
    
    根据启用的机器人自动判断特征是否启用
    
    Args:
        config: 配置字典
        
    Returns:
        启用的特征名称列表
    """
    enabled_robot_ids = set(get_enabled_robots(config))
    img_obs_horizon = config.get("dataset", {}).get("img_obs_horizon", 2)
    enabled = []
    
    for section_name in ("features", "single_frame_features"):
        for feature_name in config.get(section_name, {}).keys():
            if is_feature_enabled(feature_name, enabled_robot_ids):
                enabled.append(feature_name)
            
    for image_name, image_def in config.get("images", {}).items():
        if is_feature_enabled(image_name, enabled_robot_ids):
            if image_def.get("per_timestep", False):
                # 添加拆分后的特征名
                for i in range(img_obs_horizon):
                    enabled.append(f"{image_name}_{i}")
            else:
                enabled.append(image_name)
            
    return enabled


def get_enabled_robots(config: Dict[str, Any]) -> List[int]:
    """
    获取启用的机器人 ID 列表
    
    Args:
        config: 配置字典
        
    Returns:
        启用的机器人 ID 列表
    """
    robots = []
    for robot in config.get("robots", []):
        if robot.get("enabled", False):
            robots.append(robot["id"])
    return robots


def print_config_summary(config: Dict[str, Any], features: Dict[str, Dict[str, Any]]):
    """
    打印配置摘要
    
    Args:
        config: 配置字典
        features: 构建的特征字典
    """
    dataset_config = config.get("dataset", {})
    
    print("\n📋 数据集配置摘要:")
    print("-" * 50)
    print(f"  FPS: {dataset_config.get('fps', 20)}")
    print(f"  Robot Type: {dataset_config.get('robot_type', 'umi')}")
    print(f"  Low Dim Obs Horizon: {dataset_config.get('low_dim_obs_horizon', 2)}")
    print(f"  Img Obs Horizon: {dataset_config.get('img_obs_horizon', 2)}")
    print(f"  Obs Down Sample Steps: {dataset_config.get('obs_down_sample_steps', 3)}")
    print(f"  Action Horizon: {dataset_config.get('action_horizon', 16)}")
    print(f"  Action Dim: {dataset_config.get('action_dim', 10)}")
    
    enabled_robots = get_enabled_robots(config)
    print(f"  Enabled Robots: {enabled_robots}")
    
    print(f"\n📦 启用的特征 ({len(features)} 个):")
    print("-" * 50)
    for name, feat in features.items():
        print(f"  • {name}: shape={feat['shape']}, dtype={feat['dtype']}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    验证配置文件
    
    Args:
        config: 配置字典
        
    Returns:
        警告/错误消息列表
    """
    warnings = []
    
    # 检查必需的配置项
    if "dataset" not in config:
        warnings.append("缺少 'dataset' 配置段")
    
    if "features" not in config:
        warnings.append("缺少 'features' 配置段")
    
    # 检查是否有启用的机器人
    enabled_robots = get_enabled_robots(config)
    if not enabled_robots:
        warnings.append("没有启用任何机器人，请在 robots 配置中至少启用一个机器人")
    
    # 检查是否有启用的特征
    enabled_features = get_enabled_features(config)
    if not enabled_features:
        warnings.append("没有启用任何特征")
    
    # 检查必需的特征
    required_features = ["actions"]
    for feat in required_features:
        if feat not in config.get("features", {}):
            warnings.append(f"缺少必需特征定义: {feat}")
    
    return warnings


# 命令行工具：验证配置文件
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证数据集配置文件")
    parser.add_argument("config", type=str, help="配置文件路径")
    args = parser.parse_args()
    
    print(f"🔍 加载配置文件: {args.config}")
    config = load_dataset_config(args.config)
    
    # 验证
    warnings = validate_config(config)
    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")
    
    # 构建特征
    features = build_features_from_config(config)
    
    # 打印摘要
    print_config_summary(config, features)

