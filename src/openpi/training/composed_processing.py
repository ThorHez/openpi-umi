"""One explicit data-processing composition, independent of optional backends.

The schema is data-only. Units, coordinate frames and rotation encodings describe
the existing dataset; this module never performs IK or implicit unit/pose
conversion. NumPy and training libraries are imported only inside runtime calls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import stat


SCHEMA_VERSION = 1
SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_ECMASCRIPT_TRIM = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_FIELDS = {
    "schema_version", "robot_model", "arm_mode", "state_representation", "state_fields", "state_history",
    "action_representation", "action_key", "action_layout", "action_mode", "rotation_representation",
    "position_unit", "joint_unit", "coordinate_frame", "gripper_range", "cameras", "prompt_key",
    "normalization", "model_transform", "aux_keys",
}
_OPTIONAL_FIELDS = {"position_unit", "joint_unit", "coordinate_frame", "gripper_range",
                    "action_mode", "rotation_representation", "prompt_key"}


def _fail(message, code="COMPOSITION_INVALID"):
    raise ValueError(code + ": " + message)


def _object(value, fields, label, optional=()):
    if (not isinstance(value, dict) or set(value) - set(fields)
            or set(fields) - set(optional) - set(value)):
        _fail(label + " 必须只包含完整的已声明字段")
    return value


def _text(value, label, maximum=256):
    if not isinstance(value, str):
        _fail(label + " 必须是字符串")
    # Match the browser's ECMAScript String.trim, rather than Python's wider
    # control-character whitespace set, for identical schema normalization.
    value = value.strip(_ECMASCRIPT_TRIM)
    if not 1 <= len(value) <= maximum or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        _fail(label + " 为空、过长或包含控制字符")
    return value


def _key(value, label):
    value = _text(value, label)
    # Field paths use literal slash/dot segments, never filesystem traversal or
    # executable expressions. No string here is evaluated or imported.
    if (value.startswith(("/", "~")) or "\\" in value or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(char in value for char in "()[]{}'\"`;$")):
        _fail(label + " 必须是数据字段名，不能是路径或代码表达式")
    return value


def _enum(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        _fail(label + " 不是支持的枚举值")
    return value


def _integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(label + " 必须是范围内的整数")
    return value


def _boolean(value, label):
    if type(value) is not bool:
        _fail(label + " 必须是布尔值")
    return value


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        _fail(label + " 必须是有限数值")
    return float(value)


def _unspecified(value):
    return value is None or isinstance(value, str) and not value.strip(_ECMASCRIPT_TRIM)


def normalize_composition(value):
    """Validate v1 inputs and return every canonical key, including optional declarations."""
    value = _object(value, _FIELDS, "组合数据处理", optional=_OPTIONAL_FIELDS)
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        _fail("不支持的组合数据处理版本")
    result = {"schema_version": SCHEMA_VERSION, "robot_model": _text(value["robot_model"], "机器人型号", 100)}
    for key, choices in {
        "arm_mode": {"single", "bimanual"}, "state_representation": {"eef", "joint"},
        "action_representation": {"eef", "joint"},
        "normalization": {"quantile", "zscore"}, "model_transform": {"standard"},
    }.items():
        result[key] = _enum(value[key], choices, key)
    # These declarations do not perform conversions. Unknown units or frames
    # stay unspecified rather than receiving invented dataset semantics.
    for key, choices in (("position_unit", {"m", "mm"}), ("joint_unit", {"rad", "deg"}),
                         ("action_mode", {"absolute", "delta"}),
                         ("rotation_representation", {"axis_angle", "quaternion", "rotation_6d", "none"})):
        declared = value.get(key)
        result[key] = "" if _unspecified(declared) else _enum(declared, choices, key)
    coordinate_frame = value.get("coordinate_frame")
    result["coordinate_frame"] = "" if _unspecified(coordinate_frame) else _text(coordinate_frame, "坐标系", 100)
    result["state_history"] = _integer(value["state_history"], 1, 64, "状态历史长度")
    result["action_key"] = _key(value["action_key"], "动作字段")
    prompt_key = value.get("prompt_key")
    result["prompt_key"] = "" if _unspecified(prompt_key) else _key(prompt_key, "任务字段")
    gripper_range = value.get("gripper_range")
    result["gripper_range"] = None
    if gripper_range is not None:
        if not isinstance(gripper_range, list) or len(gripper_range) != 2:
            _fail("夹爪范围必须是两个有限数值")
        result["gripper_range"] = [_number(item, "夹爪范围") for item in gripper_range]
        if result["gripper_range"][0] >= result["gripper_range"][1]:
            _fail("夹爪最小值必须小于最大值")
    state_fields = value["state_fields"]
    if not isinstance(state_fields, list) or not 1 <= len(state_fields) <= 32:
        _fail("状态字段数量必须是 1 到 32")
    result["state_fields"] = []
    for field in state_fields:
        field = _object(field, {"key", "width", "arm", "kind", "normalize"}, "状态字段")
        result["state_fields"].append({
            "key": _key(field["key"], "状态字段"), "width": _integer(field["width"], 1, 128, "状态宽度"),
            "arm": _enum(field["arm"], {"left", "right", "shared"}, "状态机械臂"),
            "kind": _enum(field["kind"], {"position", "rotation", "joint", "gripper", "other"}, "状态语义"),
            "normalize": _boolean(field["normalize"], "状态归一化标志"),
        })
    if len({field["key"] for field in result["state_fields"]}) != len(result["state_fields"]):
        _fail("状态字段不能重复")
    action_layout = value["action_layout"]
    if not isinstance(action_layout, list) or not 1 <= len(action_layout) <= 16:
        _fail("动作块数量必须是 1 到 16")
    result["action_layout"] = []
    for block in action_layout:
        block = _object(block, {"arm", "kind", "width", "normalize"}, "动作块")
        result["action_layout"].append({
            "arm": _enum(block["arm"], {"left", "right"}, "动作机械臂"),
            "kind": _enum(block["kind"], {"position", "rotation", "joint", "gripper"}, "动作语义"),
            "width": _integer(block["width"], 1, 128, "动作块宽度"),
            "normalize": _boolean(block["normalize"], "动作归一化标志"),
        })
    if len({(field["arm"], field["kind"]) for field in result["action_layout"]}) != len(result["action_layout"]):
        _fail("同一机械臂的动作语义块不能重复")
    for key in ("state_fields", "action_layout"):
        if sum(field["width"] for field in result[key]) > 512:
            _fail("每时刻状态和动作总宽度分别不能超过 512")
    arms = {"left"} if result["arm_mode"] == "single" else {"left", "right"}
    if result["arm_mode"] == "single" and any(field["arm"] == "right" for field in result["state_fields"]):
        _fail("单臂状态不能包含右臂字段")
    for key, representation in (("action_layout", result["action_representation"]),):
        fields = result[key]
        found = {field["arm"] for field in fields if field["arm"] != "shared"}
        if found != arms:
            _fail("单/双臂声明与状态或动作所属机械臂不一致")
        for arm in arms:
            arm_fields = [field for field in fields if field["arm"] == arm]
            kinds = {field["kind"] for field in arm_fields}
            if representation == "eef":
                if not {"position", "rotation"} <= kinds or "joint" in kinds:
                    _fail("EEF 布局必须显式包含位置和旋转，不能混入关节块")
                if any(field["kind"] == "position" and field["width"] != 3 for field in arm_fields):
                    _fail("EEF 位置块宽度必须是 3")
                if any(field["kind"] == "rotation" and field["width"] not in {3, 4, 6} for field in arm_fields):
                    _fail("EEF 状态旋转宽度只能是明确的 3、4 或 6")
            elif "joint" not in kinds or kinds & {"position", "rotation"}:
                _fail("Joint 布局必须包含明确宽度的关节块，不能混入位置或旋转块")
    rotation_width = {"axis_angle": 3, "quaternion": 4, "rotation_6d": 6, "none": 0, "": 0}[result["rotation_representation"]]
    if result["action_representation"] == "joint" and rotation_width:
        _fail("关节动作的旋转编码必须是 none")
    if result["action_representation"] == "eef" and result["rotation_representation"]:
        if not rotation_width or any(field["kind"] == "rotation" and field["width"] != rotation_width
                                     for field in result["action_layout"]):
            _fail("EEF 动作旋转块宽度与声明编码不一致")
    cameras = value["cameras"]
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 3:
        _fail("组合处理支持 1 到 3 路明确的视觉输入")
    result["cameras"] = []
    for camera in cameras:
        camera = _object(camera, {"key", "slot", "kind", "layout", "time_index", "depth_min", "depth_max"}, "视觉输入")
        index = camera["time_index"]
        if index is not None:
            index = _integer(index, -64, 63, "显式图像时间索引")
        clean = {"key": _key(camera["key"], "视觉字段"), "slot": _enum(camera["slot"], set(SLOTS), "模型视觉槽位"),
                 "kind": _enum(camera["kind"], {"rgb", "depth"}, "视觉类型"),
                 "layout": _enum(camera["layout"], {"chw", "hwc"}, "图像布局"), "time_index": index,
                 "depth_min": _number(camera["depth_min"], "深度最小值"),
                 "depth_max": _number(camera["depth_max"], "深度最大值")}
        if clean["depth_min"] >= clean["depth_max"]:
            _fail("深度最小值必须小于最大值")
        result["cameras"].append(clean)
    if len({camera["key"] for camera in result["cameras"]}) != len(result["cameras"]):
        _fail("视觉源字段不能重复")
    if len({camera["slot"] for camera in result["cameras"]}) != len(result["cameras"]):
        _fail("模型视觉槽位不能重复")
    source_keys = [field["key"] for field in result["state_fields"] + result["cameras"]]
    if (len(set(source_keys)) != len(source_keys) or result["action_key"] in source_keys
            or result["prompt_key"] in source_keys or result["action_key"] == result["prompt_key"]):
        _fail("状态、视觉、动作和任务字段不能互相占用")
    aux_keys = value["aux_keys"]
    if (not isinstance(aux_keys, list) or len(aux_keys) > 2 or not all(type(key) is str and key in {"value_target", "is_positive"} for key in aux_keys)
            or len(set(aux_keys)) != len(aux_keys)):
        _fail("辅助字段只支持唯一的 value_target / is_positive")
    result["aux_keys"] = list(aux_keys)
    if any(key in {*source_keys, result["action_key"], result["prompt_key"]} for key in result["aux_keys"]):
        _fail("辅助字段不能与其它输入字段重名")
    return result


def normalization_masks(composition):
    spec = normalize_composition(composition)
    return {target: tuple(field["normalize"] for field in spec[source] for _ in range(field["width"]))
            for target, source in (("state", "state_fields"), ("actions", "action_layout"))}


@dataclasses.dataclass(frozen=True)
class ComposedRepack:
    """Read literal dataset keys, including explicit nested slash paths."""
    composition: dict

    def __call__(self, data):
        import numpy as np
        spec = normalize_composition(self.composition)
        def get(key):
            if key in data:
                return data[key]
            value = data
            for part in key.split("/"):
                if not isinstance(value, dict) or part not in value:
                    _fail("样本缺少声明的数据字段：" + key, "COMPOSITION_DATA_INVALID")
                value = value[part]
            return value
        if spec["prompt_key"]:
            prompt = get(spec["prompt_key"])
        else:
            # Only unclaimed task/prompt inputs may be used as automatic text;
            # a numeric state/action source can legitimately have either name.
            claimed = {field["key"] for field in spec["state_fields"] + spec["cameras"]}
            claimed.update([spec["action_key"], *spec["aux_keys"]])
            prompt = ""
            for key in ("task", "prompt"):
                if any(source == key or source.startswith(key + "/") for source in claimed) or key not in data:
                    continue
                candidate = data[key]
                if isinstance(candidate, np.ndarray) and candidate.shape == ():
                    candidate = candidate.item()
                if not isinstance(candidate, str):
                    _fail("自动任务字段必须是标量文本：" + key, "COMPOSITION_DATA_INVALID")
                if candidate.strip():
                    prompt = candidate
                    break
        result = {"actions": get(spec["action_key"]), "prompt": prompt}
        for index, field in enumerate(spec["state_fields"]):
            result["_composition_state_" + str(index)] = get(field["key"])
        for index, camera in enumerate(spec["cameras"]):
            result["_composition_camera_" + str(index)] = get(camera["key"])
        result.update({key: get(key) for key in spec["aux_keys"]})
        return result


@dataclasses.dataclass(frozen=True)
class _LeRobotImageLayout:
    index: int
    key: str
    decoded_shape: tuple[int, ...]
    source_layout: str


@dataclasses.dataclass(frozen=True)
class LeRobotComposedRepack:
    """Restore declared source layout after LeRobot's image/video CHW decoder.

    Only the factory installs this boundary with verified codec metadata. Direct
    ComposedRepack callers keep their original array contract. Conversion never
    guesses an axis or selects a history frame.
    """
    composition: dict
    image_layouts: tuple[_LeRobotImageLayout, ...]

    def __call__(self, data):
        import numpy as np
        result = ComposedRepack(self.composition)(data)
        for camera in self.image_layouts:
            target = "_composition_camera_" + str(camera.index)
            try:
                image = np.asarray(result[target])
            except (TypeError, ValueError) as error:
                raise ValueError("COMPOSITION_DATA_INVALID: 无法读取 LeRobot 解码图像：" + camera.key) from error
            if image.shape != camera.decoded_shape:
                _fail("LeRobot image/video RGB 解码结果必须是与源元信息一致的 CHW/TCHW 三通道图像：" + camera.key,
                      "COMPOSITION_DATA_INVALID")
            if camera.source_layout == "hwc":
                result[target] = np.transpose(image, (1, 2, 0) if image.ndim == 3 else (0, 2, 3, 1))
        return result


@dataclasses.dataclass(frozen=True)
class ComposedInputs:
    composition: dict
    action_horizon: int
    model_action_dim: int

    def __call__(self, data):
        import numpy as np
        spec = normalize_composition(self.composition)
        def numeric(value, label):
            try:
                value = np.asarray(value)
            except (TypeError, ValueError) as error:
                raise ValueError("COMPOSITION_DATA_INVALID: 无法读取数值字段 " + label) from error
            if value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
                _fail("数值字段必须是有限实数：" + label, "COMPOSITION_DATA_INVALID")
            return value
        def gripper(value, label):
            if spec["gripper_range"] is None:
                return
            lo, hi = spec["gripper_range"]
            if np.any(value < lo) or np.any(value > hi):
                _fail("夹爪数值超出显式声明范围：" + label, "COMPOSITION_DATA_INVALID")
        states = []
        for index, field in enumerate(spec["state_fields"]):
            value = numeric(data["_composition_state_" + str(index)], field["key"])
            if spec["state_history"] == 1 and value.shape == (field["width"],):
                value = value.reshape(1, field["width"])
            if value.shape != (spec["state_history"], field["width"]):
                _fail("状态字段形状与历史长度/宽度不一致：" + field["key"], "COMPOSITION_DATA_INVALID")
            if field["kind"] == "gripper":
                gripper(value, field["key"])
            states.append(value)
        actions = numeric(data["actions"], spec["action_key"])
        width = sum(field["width"] for field in spec["action_layout"])
        if type(self.action_horizon) is not int or self.action_horizon <= 0 or actions.shape != (self.action_horizon, width):
            _fail("动作必须与模型 action_horizon 及声明布局宽度完全一致", "COMPOSITION_DATA_INVALID")
        if type(self.model_action_dim) is not int or self.model_action_dim < width:
            _fail("模型 action_dim 小于真实动作宽度", "COMPOSITION_MODEL_UNSUPPORTED")
        start = 0
        for field in spec["action_layout"]:
            if field["kind"] == "gripper":
                gripper(actions[:, start:start + field["width"]], "actions")
            start += field["width"]
        images, masks = {}, {}
        for index, camera in enumerate(spec["cameras"]):
            image = np.asarray(data["_composition_camera_" + str(index)])
            time_index = camera["time_index"]
            if time_index is not None:
                allowed_dims = {4} if camera["kind"] == "rgb" else {3, 4}
                if image.ndim not in allowed_dims or not -image.shape[0] <= time_index < image.shape[0]:
                    _fail("图像缺少有效的显式时间轴/时间索引：" + camera["key"], "COMPOSITION_DATA_INVALID")
                image = image[time_index]
            if camera["kind"] == "rgb":
                if image.ndim != 3 or image.shape[0 if camera["layout"] == "chw" else -1] != 3:
                    _fail("RGB 图像形状必须与明确 CHW/HWC 布局一致，历史图像须选择时间索引", "COMPOSITION_DATA_INVALID")
                if image.dtype == np.uint8:
                    pass
                elif image.dtype.kind == "f" and np.all(np.isfinite(image)) and np.all((image >= 0) & (image <= 1)):
                    image = (image * 255).astype(np.uint8)
                else:
                    _fail("RGB 仅支持 uint8 或有限的 0..1 浮点图像", "COMPOSITION_DATA_INVALID")
                if camera["layout"] == "chw":
                    image = np.transpose(image, (1, 2, 0))
            else:
                if image.ndim == 3:
                    axis = 0 if camera["layout"] == "chw" else -1
                    if image.shape[axis] != 1:
                        _fail("深度图只能有一个显式通道，历史图像须选择时间索引", "COMPOSITION_DATA_INVALID")
                    image = np.squeeze(image, axis=axis)
                if image.ndim != 2:
                    _fail("深度图必须是 HW 或显式单通道 CHW/HWC", "COMPOSITION_DATA_INVALID")
                image = numeric(image, camera["key"])
                scaled = np.clip((image - camera["depth_min"]) / (camera["depth_max"] - camera["depth_min"]), 0.0, 1.0)
                image = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=-1)
            if not 1 <= image.shape[0] <= 4096 or not 1 <= image.shape[1] <= 4096:
                _fail("图像空间尺寸无效或超过 4096", "COMPOSITION_DATA_INVALID")
            images[camera["slot"]] = image
            masks[camera["slot"]] = np.True_
        reference = next(iter(images.values()))
        for slot in SLOTS:
            if slot not in images:
                images[slot], masks[slot] = np.zeros_like(reference, dtype=np.uint8), np.False_
        prompt = data["prompt"]
        if isinstance(prompt, np.ndarray) and prompt.shape == ():
            prompt = prompt.item()
        if not isinstance(prompt, str):
            _fail("任务字段必须是标量文本", "COMPOSITION_DATA_INVALID")
        if spec["prompt_key"] and not prompt.strip():
            _fail("任务字段必须是非空标量文本", "COMPOSITION_DATA_INVALID")
        result = {"state": np.concatenate(states, axis=-1), "actions": actions,
                  "image": {slot: images[slot] for slot in SLOTS},
                  "image_mask": {slot: masks[slot] for slot in SLOTS}, "prompt": prompt}
        for key in spec["aux_keys"]:
            value = np.asarray(data[key])
            if value.size != 1 or value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
                _fail("辅助监督字段必须是有限标量：" + key, "COMPOSITION_DATA_INVALID")
            if key == "is_positive" and value.reshape(-1)[0] not in (0, 1):
                _fail("is_positive 只能是显式的 0 或 1", "COMPOSITION_DATA_INVALID")
            result[key] = value
        return result


def _read_lerobot_metadata(repo_id):
    """Read one bounded, non-symlink LeRobot metadata file without modifying it."""
    if not isinstance(repo_id, str) or not Path(repo_id).is_absolute():
        _fail("组合处理需要登记的本机数据集绝对路径", "COMPOSITION_DATA_INVALID")
    source = Path(repo_id) / "meta/info.json"
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            file_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_info.st_mode) or file_info.st_size > 1024 * 1024:
                _fail("LeRobot 元信息必须是 1MiB 内的普通文件", "COMPOSITION_DATA_INVALID")
            content = stream.read(1024 * 1024 + 1)
    except OSError as error:
        raise ValueError("COMPOSITION_DATA_INVALID: 无法安全读取本机 LeRobot 元信息") from error
    if len(content) > 1024 * 1024:
        _fail("LeRobot 元信息超过读取上限", "COMPOSITION_DATA_INVALID")
    try:
        info = json.loads(content)
        if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
            raise ValueError("missing features")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("COMPOSITION_DATA_INVALID: 无法读取 LeRobot 特征元信息") from error
    return info, hashlib.sha256(content).hexdigest()


def _action_metadata_contract(repo_id, action_key, horizon, width):
    """Select chunk reading only from actual bounded LeRobot feature metadata."""
    info, metadata_sha256 = _read_lerobot_metadata(repo_id)
    try:
        shape = info["features"][action_key]["shape"]
    except (ValueError, KeyError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("COMPOSITION_DATA_INVALID: 无法核验声明的动作字段元信息") from error
    if not isinstance(shape, list) or any(type(item) is not int or item <= 0 for item in shape):
        _fail("动作特征形状无效", "COMPOSITION_DATA_INVALID")
    if shape == [width]:
        sequence_keys = [action_key]
    elif shape == [horizon, width]:
        sequence_keys = []
    else:
        _fail("动作元信息必须是单步 [D] 或预分块 [H,D]，并与模型/布局一致", "COMPOSITION_DATA_INVALID")
    return {"action_sequence_keys": sequence_keys, "action_metadata_sha256": metadata_sha256}


def _lerobot_image_layouts(spec, info):
    layouts = []
    for index, camera in enumerate(spec["cameras"]):
        feature = info["features"].get(camera["key"])
        # Numeric/custom arrays and depth images retain their original contract.
        # No decoder behavior is inferred from the shape or the camera name.
        if (camera["kind"] != "rgb" or not isinstance(feature, dict)
                or not isinstance(feature.get("dtype"), str) or feature["dtype"] not in {"image", "video"}):
            continue
        shape = feature.get("shape")
        if (not isinstance(shape, list) or len(shape) not in {3, 4}
                or any(type(size) is not int or size <= 0 for size in shape)):
            _fail("LeRobot RGB 源元信息需要明确的三维图像或四维历史图像：" + camera["key"], "COMPOSITION_DATA_INVALID")
        history = len(shape) == 4
        if history != (camera["time_index"] is not None):
            _fail("LeRobot 历史图像需要显式时间索引，单帧图像不能声明时间索引：" + camera["key"], "COMPOSITION_DATA_INVALID")
        if history and not -shape[0] <= camera["time_index"] < shape[0]:
            _fail("图像时间索引越界：" + camera["key"], "COMPOSITION_DATA_INVALID")
        frame = shape[-3:]
        if frame[0 if camera["layout"] == "chw" else -1] != 3:
            _fail("RGB 源元信息须明确三通道并与声明 CHW/HWC 布局一致：" + camera["key"], "COMPOSITION_DATA_INVALID")
        height, width = frame[1:] if camera["layout"] == "chw" else frame[:2]
        decoded = (shape[0], 3, height, width) if history else (3, height, width)
        layouts.append(_LeRobotImageLayout(index, camera["key"], decoded, camera["layout"]))
    return tuple(layouts)


def _action_sequence_keys(repo_id, action_key, horizon, width):
    """Retain the tuple interface used by older composition callers/tests."""
    return tuple(_action_metadata_contract(repo_id, action_key, horizon, width)["action_sequence_keys"])


def _view_contract(spec, model_config):
    """Resolve the standard model's selected canonical RGB/depth view slots."""
    depth = getattr(model_config, "depth_keys", None)
    depth = [] if depth is None else depth
    images = getattr(model_config, "image_keys", None)
    if images is None and isinstance(depth, (tuple, list)):
        images = [slot for slot in SLOTS if slot not in depth]
    for keys in (images, depth):
        if (not isinstance(keys, (tuple, list)) or any(key not in SLOTS for key in keys)
                or len(set(keys)) != len(keys)):
            _fail("组合处理的模型 image_keys/depth_keys 必须使用唯一 canonical 视觉槽", "COMPOSITION_MODEL_UNSUPPORTED")
    if set(images) & set(depth) or not (images or depth):
        _fail("模型 RGB/深度视图不可交叉或全部为空", "COMPOSITION_MODEL_UNSUPPORTED")
    selected = set(images) | set(depth)
    actual = {camera["slot"] for camera in spec["cameras"]}
    if not actual <= selected:
        _fail("模型选择的视觉槽必须包含本资产所有实际视图，禁止忽略已声明视觉或产生全空视图", "COMPOSITION_MODEL_UNSUPPORTED")
    for camera in spec["cameras"]:
        if (camera["kind"] == "depth") != (camera["slot"] in depth):
            _fail("视觉种类与模型 depth_keys 不一致；请配置模型 depth_keys，或使用匹配的标准 RGB 配方", "COMPOSITION_MODEL_UNSUPPORTED")
    return {"image_keys": list(images), "depth_keys": list(depth)}


def composition_runtime_contract(factory, model_config):
    """Validate runtime boundaries without loading norm files or a tokenizer.

    The same checks are used by preflight and ``create``. Metadata is the actual
    bounded local file; its digest lets callers compare their frozen evidence.
    The returned dictionary contains only JSON values.
    """
    from openpi.models import pi0_config, pi0_gripper, pi0_value
    spec = normalize_composition(factory.composition)
    if type(model_config) not in {pi0_config.Pi0Config, pi0_gripper.Pi0GripperConfig, pi0_value.Pi0ValueConfig}:
        _fail("组合处理首版只支持标准 Pi0/Pi05、Value 和 Gripper 模型；MEM/Hybrid 使用原配方", "COMPOSITION_MODEL_UNSUPPORTED")
    action_dim, horizon = model_config.action_dim, model_config.action_horizon
    if type(action_dim) is not int or type(horizon) is not int or action_dim <= 0 or horizon <= 0:
        _fail("模型动作维度/预测长度无效", "COMPOSITION_MODEL_UNSUPPORTED")
    width = sum(field["width"] for field in spec["action_layout"])
    if width > action_dim:
        _fail("模型 action_dim 小于真实动作宽度", "COMPOSITION_MODEL_UNSUPPORTED")
    if type(model_config) is pi0_gripper.Pi0GripperConfig:
        expected_indices, start = [], 0
        for field in spec["action_layout"]:
            if field["kind"] == "gripper":
                if field["normalize"]:
                    _fail("Gripper 二分类头需要原始夹爪单位，禁止归一化其动作维度", "COMPOSITION_MODEL_UNSUPPORTED")
                expected_indices.extend(range(start, start + field["width"]))
            start += field["width"]
        indices = getattr(model_config, "gripper_binary_indices", None)
        if (not isinstance(indices, (tuple, list))
                or any(type(index) is not int or not 0 <= index < width for index in indices)
                or list(indices) != expected_indices):
            _fail("Gripper 头必须显式配置与动作布局逐项一致、位于真实动作范围内的 gripper_binary_indices", "COMPOSITION_MODEL_UNSUPPORTED")
    pi05 = getattr(model_config, "pi05", False) is True
    state_dim = sum(field["width"] for field in spec["state_fields"])
    if not pi05 and spec["state_history"] * state_dim > action_dim:
        _fail("Pi0 展平后状态宽度超过模型 action_dim，禁止静默截断", "COMPOSITION_MODEL_UNSUPPORTED")
    stage = factory.training_stage
    if not isinstance(stage, str) or stage not in {"sft", "value", "offline_rl", "uot"}:
        _fail("组合处理不支持此训练阶段", "COMPOSITION_STAGE_UNSUPPORTED")
    if stage == "value" and "value_target" not in spec["aux_keys"]:
        _fail("价值训练需要显式 value_target 辅助字段", "COMPOSITION_STAGE_UNSUPPORTED")
    if (stage == "value") != (type(model_config) is pi0_value.Pi0ValueConfig):
        _fail("价值模型与 value 训练阶段必须一致", "COMPOSITION_STAGE_UNSUPPORTED")
    if stage == "offline_rl" and "is_positive" not in spec["aux_keys"]:
        _fail("ACP 训练需要显式 is_positive 辅助字段", "COMPOSITION_STAGE_UNSUPPORTED")
    dropout = factory.acp_dropout
    if type(dropout) not in (int, float) or not 0 <= dropout <= 1:
        _fail("ACP dropout 必须是 0..1 的有限数值")
    metadata = _action_metadata_contract(factory.repo_id, spec["action_key"], horizon, width)
    views = _view_contract(spec, model_config)
    return {"training_stage": stage, "model_type": type(model_config).__name__, "pi05": pi05,
            "action_dim": action_dim, "action_horizon": horizon, "raw_state_dim": state_dim,
            "raw_action_dim": width, "state_history": spec["state_history"],
            "normalization": spec["normalization"], "acp_dropout": float(dropout),
            "action_loss_mask": [1.0] * width + [0.0] * (action_dim - width), **metadata, **views}


def compose_data_config(factory, assets_dirs, model_config):
    """Compile a composition into the existing DataConfig/transform interface."""
    contract = composition_runtime_contract(factory, model_config)
    from openpi import transforms
    from openpi.models import tokenizer
    spec = normalize_composition(factory.composition)
    action_dim, horizon = contract["action_dim"], contract["action_horizon"]
    stage, pi05, dropout = contract["training_stage"], contract["pi05"], contract["acp_dropout"]
    base = factory.create_base_config(assets_dirs, model_config)
    if base.repo_id != factory.repo_id:
        _fail("基础数据配置改变了已核验的数据集路径", "COMPOSITION_DATA_INVALID")
    metadata, metadata_sha256 = _read_lerobot_metadata(factory.repo_id)
    if metadata_sha256 != contract["action_metadata_sha256"]:
        _fail("创建组合处理时 LeRobot 元信息已改变，请重新预检", "COMPOSITION_DATA_INVALID")
    repack = LeRobotComposedRepack(spec, _lerobot_image_layouts(spec, metadata))
    sequence_keys = tuple(contract["action_sequence_keys"])
    model_inputs = [transforms.InjectDefaultPrompt(None), transforms.ResizeImages(224, 224)]
    if stage == "offline_rl":
        model_inputs.insert(1, transforms.ACPConditionPrompt(indicator_key="is_positive", dropout=float(dropout), default_positive=False))
    model_inputs.append(transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=bool(getattr(model_config, "discrete_state_input", False))))
    if pi05:
        model_inputs.extend([transforms.PadActionsOnly(action_dim), transforms.FlattenState()])
    else:
        model_inputs.extend([transforms.FlattenState(), transforms.PadStatesAndActions(action_dim)])
    model_inputs.append(transforms.KeepModelKeys(extra_keys=frozenset(spec["aux_keys"])))
    return dataclasses.replace(base, repack_transforms=transforms.Group(inputs=[repack]),
        data_transforms=transforms.Group(inputs=[ComposedInputs(spec, horizon, action_dim)], outputs=[]),
        model_transforms=transforms.Group(inputs=model_inputs), normalize_masks=normalization_masks(spec),
        use_quantile_norm=spec["normalization"] == "quantile", action_sequence_keys=sequence_keys,
        action_loss_mask=tuple(contract["action_loss_mask"]), prompt_from_task=spec["prompt_key"] == "task")
