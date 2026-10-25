"""Real NumPy composition and retained UMI equivalence without JAX or GPUs."""
from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
from functools import partial
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/openpi/training/composed_processing.py"
SPEC = importlib.util.spec_from_file_location("composed_processing_under_test", MODULE_PATH)
COMPOSED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPOSED
SPEC.loader.exec_module(COMPOSED)


def composition(bimanual=False):
    states, cameras, actions = [], [], []
    for robot in range(2 if bimanual else 1):
        arm = "left" if robot == 0 else "right"
        if bimanual:
            fields = [("eef_pos", 3, "position", True), ("eef_pos_wrt_start", 3, "position", True),
                      ("eef_rot_axis_angle", 6, "rotation", False),
                      ("eef_rot_axis_angle_wrt_start", 6, "rotation", False),
                      (f"eef_pos_wrt{1-robot}", 3, "position", True),
                      (f"eef_rot_axis_angle_wrt{1-robot}", 6, "rotation", False),
                      ("gripper_width", 1, "gripper", False)]
        else:
            fields = [("eef_pos", 3, "position", True), ("eef_rot_axis_angle", 6, "rotation", False),
                      ("eef_rot_axis_angle_wrt_start", 6, "rotation", False), ("gripper_width", 1, "gripper", False)]
        prefix = "" if bimanual else "observation."
        states.extend({"key": prefix + f"robot{robot}_{key}", "width": width, "arm": arm, "kind": kind, "normalize": norm}
                      for key, width, kind, norm in fields)
        cameras.append({"key": prefix + f"{arm}_wrist_0_rgb_0", "slot": f"{arm}_wrist_0_rgb", "kind": "rgb", "layout": "chw",
                        "time_index": None, "depth_min": 0.0, "depth_max": 1.0})
        actions.extend({"arm": arm, "kind": kind, "width": width, "normalize": norm}
                       for kind, width, norm in (("position", 3, True), ("rotation", 6, False), ("gripper", 1, False)))
    return {"schema_version": 1, "robot_model": "UMI", "arm_mode": "bimanual" if bimanual else "single",
            "state_representation": "eef", "state_fields": states, "state_history": 2,
            "action_representation": "eef", "action_key": "actions", "action_layout": actions,
            "action_mode": "absolute", "rotation_representation": "rotation_6d", "position_unit": "m", "joint_unit": "rad",
            "coordinate_frame": "dataset_declared_base", "gripper_range": [0, 0.1], "cameras": cameras,
            "prompt_key": "task", "normalization": "quantile", "model_transform": "standard", "aux_keys": []}


def joint_composition():
    spec = composition()
    spec.update(state_history=1, state_representation="joint", action_representation="joint", rotation_representation="none")
    spec["state_fields"] = [{"key": "observation.state", "width": 6, "arm": "shared", "kind": "joint", "normalize": True}]
    spec["action_layout"] = [{"arm": "left", "kind": "joint", "width": 6, "normalize": True}]
    return spec


class SchemaTests(unittest.TestCase):
    def test_schema_is_complete_strict_and_does_not_import_numpy_or_training_backends(self):
        result = subprocess.run([sys.executable, "-I", "-c",
            "import runpy,sys;runpy.run_path(" + repr(str(MODULE_PATH)) + ");assert 'numpy' not in sys.modules;assert 'jax' not in sys.modules"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = composition()
        normalized = COMPOSED.normalize_composition(spec)
        self.assertEqual(spec["gripper_range"], [0, 0.1])
        self.assertIs(type(normalized["gripper_range"][0]), float)
        for key in spec.keys() - {"position_unit", "joint_unit", "coordinate_frame", "gripper_range",
                                  "action_mode", "rotation_representation", "prompt_key"}:
            bad = deepcopy(spec);bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                COMPOSED.normalize_composition(bad)

    def test_optional_declarations_are_canonical_without_guessing_for_any_layout(self):
        optional = {"position_unit": "", "joint_unit": "", "coordinate_frame": "", "gripper_range": None,
                    "action_mode": "", "rotation_representation": "", "prompt_key": ""}
        for factory in (composition, lambda: composition(bimanual=True), joint_composition):
            original = factory()
            # Existing explicit declarations retain their values and normalized shape.
            self.assertEqual(COMPOSED.normalize_composition(original), original)
            missing = {key: value for key, value in original.items() if key not in optional}
            expected = {**original, **optional}
            self.assertEqual(COMPOSED.normalize_composition(missing), expected)
            for blank in (None, "", " \u00a0\ufeff "):
                explicit = {**original, **{key: blank for key in optional if key != "gripper_range"}, "gripper_range": None}
                with self.subTest(layout=original["arm_mode"], representation=original["action_representation"], blank=blank):
                    normalized = COMPOSED.normalize_composition(explicit)
                    self.assertEqual(normalized, expected)
                    self.assertEqual(COMPOSED.normalize_composition(normalized), expected)
                    self.assertEqual(set(normalized), set(original))
                    self.assertEqual(COMPOSED.normalization_masks(normalized), COMPOSED.normalization_masks(original))

    def test_supplied_optional_declarations_are_still_validated(self):
        for key, invalid_values in {
            "position_unit": ("cm", " m ", False, 0, [], "m\x1f"),
            "joint_unit": ("degrees", " rad ", False, 0, {}, "rad\x85"),
            "coordinate_frame": (False, 0, [], {}, "x" * 101, "base\x1f", "\x85"),
            "action_mode": ("infer", " absolute ", False, 0, [], "delta\x1f"),
            "rotation_representation": ("euler", " rotation_6d ", False, 0, [], "none\x85"),
            "prompt_key": (False, 0, [], {}, "x" * 257, "task\x1f", "/tmp/task", "task/../prompt"),
            "gripper_range": ("", [], [0], [0, 1, 2], (0, 1), [False, 1], [0, float("nan")],
                              [0, float("inf")], [0, 10**1000], [1, 1], [2, 1]),
        }.items():
            for invalid in invalid_values:
                for factory in (composition, joint_composition):
                    spec = factory();spec[key] = invalid
                    with self.subTest(key=key, value=invalid, representation=spec["action_representation"]), self.assertRaises(ValueError):
                        COMPOSED.normalize_composition(spec)

    def test_unlabeled_rotation_keeps_real_width_checks_and_explicit_encoding_contracts(self):
        for width in (3, 4, 6):
            spec = composition();spec.pop("rotation_representation")
            spec["action_layout"][1]["width"] = width
            with self.subTest(width=width):
                self.assertEqual(COMPOSED.normalize_composition(spec)["rotation_representation"], "")
        for kind, width in (("position", 4), ("rotation", 2), ("rotation", 5)):
            spec = composition();spec["rotation_representation"] = ""
            next(block for block in spec["action_layout"] if block["kind"] == kind)["width"] = width
            with self.subTest(kind=kind, width=width), self.assertRaises(ValueError):
                COMPOSED.normalize_composition(spec)
        for representation, encoding in (("eef", "none"), ("joint", "axis_angle"), ("joint", "quaternion"), ("joint", "rotation_6d")):
            spec = composition() if representation == "eef" else joint_composition()
            spec["rotation_representation"] = encoding
            with self.subTest(representation=representation, encoding=encoding), self.assertRaises(ValueError):
                COMPOSED.normalize_composition(spec)

    def test_unknown_code_paths_and_false_numeric_types_are_rejected(self):
        mutations = [lambda s: s.update(extra=1), lambda s: s.update(schema_version=True),
                     lambda s: s.update(state_history=True), lambda s: s.update(action_mode="infer"),
                     lambda s: s.update(position_unit="cm"), lambda s: s.update(model_transform="memory"),
                     lambda s: s["state_fields"][0].update(normalize=1), lambda s: s["action_layout"][0].update(width=True),
                     lambda s: s["cameras"][0].update(time_index=True), lambda s: s["cameras"][0].update(depth_max=float("inf")),
                     lambda s: s.update(gripper_range=[False, 1]), lambda s: s.update(gripper_range=[0, float("nan")]),
                     lambda s: s.update(action_key="/data/actions"), lambda s: s.update(action_key="../actions"),
                     lambda s: s.update(action_key="C:/actions"), lambda s: s.update(action_key="__import__('os')"),
                     lambda s: s.update(action_key="model:weights"), lambda s: s.update(action_key="state/../actions"),
                     lambda s: s.update(gripper_range=[0, 10**1000]), lambda s: s.update(aux_keys=["exec"])]
        for index, mutate in enumerate(mutations):
            spec = composition();mutate(spec)
            with self.subTest(index=index), self.assertRaises(ValueError):
                COMPOSED.normalize_composition(spec)

    def test_text_trimming_matches_ecmascript_without_silently_removing_controls(self):
        trim_chars = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
        for character in trim_chars:
            spec = composition();spec["robot_model"] = character + "UMI" + character
            spec["action_key"] = character + "actions" + character
            with self.subTest(character=hex(ord(character))):
                actual = COMPOSED.normalize_composition(spec)
                self.assertEqual(actual["robot_model"], "UMI")
                self.assertEqual(actual["action_key"], "actions")
        for character in ("\u001c", "\u001d", "\u001e", "\u001f", "\u0085"):
            spec = composition();spec["robot_model"] = character + "UMI" + character
            with self.subTest(control=hex(ord(character))), self.assertRaises(ValueError):
                COMPOSED.normalize_composition(spec)
        for character in ("\u180e", "\u200b"):
            spec = composition();spec["robot_model"] = character + "UMI" + character
            with self.subTest(preserved=hex(ord(character))):
                self.assertEqual(COMPOSED.normalize_composition(spec)["robot_model"], spec["robot_model"])

    def test_layout_and_input_collisions_are_rejected_while_shared_state_is_explicit(self):
        self.assertEqual(COMPOSED.normalize_composition(joint_composition())["state_fields"][0]["arm"], "shared")
        for mutate in (lambda s: s.update(arm_mode="bimanual"),
                       lambda s: s["action_layout"][1].update(width=4),
                       lambda s: s["action_layout"].append(deepcopy(s["action_layout"][0])),
                       lambda s: s["state_fields"][0].update(arm="right"),
                       lambda s: s["cameras"].append(deepcopy(s["cameras"][0])),
                       lambda s: s.update(action_key=s["state_fields"][0]["key"]),
                       lambda s: s.update(prompt_key=s["cameras"][0]["key"]),
                       lambda s: s.update(prompt_key="is_positive", aux_keys=["is_positive"])):
            spec = composition();mutate(spec)
            with self.subTest(spec=spec), self.assertRaises(ValueError):COMPOSED.normalize_composition(spec)


def actual_definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@unittest.skipIf(np is None, "NumPy required for real sample checks")
class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = {"np": np, "dataclasses": dataclasses, "transforms": types.SimpleNamespace(DataTransformFn=object),
                      "einops": types.SimpleNamespace(rearrange=lambda x, pattern: np.transpose(x, (1, 2, 0)))}
        actual_definitions(ROOT / "src/openpi/policies/umi_policy.py",
                           {"_parse_image", "UmiInputsV4_Horizon1", "UmiInputsV4_Bimanual_Horizon1"}, cls.legacy)
        cls.transforms = {"np": np, "dataclasses": dataclasses, "partial": partial, "DataTransformFn": object}
        actual_definitions(ROOT / "src/openpi/transforms.py", {"Normalize", "FlattenState"}, cls.transforms)

    def sample(self, spec):
        data = {field["key"]: np.full((spec["state_history"], field["width"]),
                0.025 if field["kind"] == "gripper" else (index + 1) / 100.0, dtype=np.float32)
                for index, field in enumerate(spec["state_fields"])}
        width = sum(field["width"] for field in spec["action_layout"])
        data[spec["action_key"]] = np.arange(16 * width, dtype=np.float32).reshape(16, width) / 1000
        start = 0
        for field in spec["action_layout"]:
            if field["kind"] == "gripper": data[spec["action_key"]][:, start:start+field["width"]] = 0.025
            start += field["width"]
        for index, camera in enumerate(spec["cameras"]):data[camera["key"]] = np.full((3, 224, 224), 30 + index, dtype=np.uint8)
        data[spec["prompt_key"]] = "move the cloth"
        return data

    def apply(self, spec, data, horizon=16, action_dim=32):
        return COMPOSED.ComposedInputs(spec, horizon, action_dim)(COMPOSED.ComposedRepack(spec)(data))

    def test_single_and_bimanual_umi_horizon1_are_numerically_equivalent(self):
        config_tree = ast.parse((ROOT / "src/openpi/training/config.py").read_text())
        for bimanual, processor, factory in ((False, "UmiInputsV4_Horizon1", "LeRobotUmiDataConfigPadded_V4_ImageHorizon1"),
                                            (True, "UmiInputsV4_Bimanual_Horizon1", "LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1")):
            with self.subTest(bimanual=bimanual):
                spec = composition(bimanual);data = self.sample(spec)
                if bimanual:legacy_data = dict(data)
                else:legacy_data = {key.removeprefix("observation."): value for key, value in data.items()}
                legacy_data["prompt"] = legacy_data.pop("task")
                expected = self.legacy[processor]()(deepcopy(legacy_data));actual = self.apply(spec, data)
                np.testing.assert_array_equal(actual["state"], expected["state"])
                np.testing.assert_array_equal(actual["actions"], expected["actions"])
                self.assertIs(actual["actions"], data["actions"])
                self.assertEqual(list(actual["image"]), ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"])
                for key in actual["image"]:
                    np.testing.assert_array_equal(actual["image"][key], expected["image"][key])
                    self.assertEqual(actual["image_mask"][key], expected["image_mask"][key])
                node = next(node for node in config_tree.body if isinstance(node, ast.ClassDef) and node.name == factory)
                masks = next(node.value for node in node.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "normalize_masks" for t in node.targets))
                legacy_masks = eval(compile(ast.Expression(masks), "<retained-masks>", "eval"),
                                    {"make_bool_mask": lambda *runs: tuple(bit for run in runs for bit in [run > 0] * abs(run))})
                self.assertEqual(COMPOSED.normalization_masks(spec), legacy_masks)
                normalizer = self.transforms["Normalize"](None)
                for key in ("state", "actions"):
                    size = actual[key].shape[-1]
                    stats = types.SimpleNamespace(mean=np.zeros(size), std=np.ones(size), q01=np.zeros(size), q99=np.ones(size), min=None, max=None)
                    for method in ("_normalize", "_normalize_quantile"):
                        expected_normal = getattr(normalizer, method)(expected[key], stats, norm_mask=legacy_masks[key])
                        actual_normal = getattr(normalizer, method)(actual[key], stats, norm_mask=COMPOSED.normalization_masks(spec)[key])
                        np.testing.assert_array_equal(actual_normal, expected_normal)
                np.testing.assert_array_equal(self.transforms["FlattenState"]()(dict(actual))["state"], expected["state"].flatten())

    def test_joint_state_action_and_history_keep_the_declared_axes(self):
        spec = joint_composition();sample = self.sample(spec);sample["observation.state"] = np.arange(6, dtype=np.float32)
        actual = self.apply(spec, sample)
        self.assertEqual(actual["state"].shape, (1, 6));self.assertEqual(actual["actions"].shape, (16, 6))
        np.testing.assert_array_equal(actual["state"][0], sample["observation.state"])
        spec["state_history"] = 3;sample["observation.state"] = np.arange(18).reshape(3, 6)
        self.assertEqual(self.apply(spec, sample)["state"].shape, (3, 6))
        spec["state_representation"] = "eef"  # A shared preconcatenated vector remains explicitly declared.
        self.assertEqual(self.apply(spec, sample)["state"].shape, (3, 6))

    def test_visual_layout_history_float_rgb_and_explicit_depth_scaling(self):
        spec = composition();sample = self.sample(spec);key = spec["cameras"][0]["key"]
        spec["cameras"][0].update(layout="hwc", time_index=-1)
        sample[key] = np.stack([np.zeros((4, 5, 3), dtype=np.float32), np.ones((4, 5, 3), dtype=np.float32)])
        result = self.apply(spec, sample);np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], 255)
        spec["cameras"][0].update(kind="depth", layout="chw", time_index=None, depth_min=100, depth_max=300)
        sample[key] = np.asarray([[100, 200, 300]], dtype=np.float32)
        image = self.apply(spec, sample)["image"]["left_wrist_0_rgb"]
        np.testing.assert_array_equal(image[0, :, 0], [0, 127, 255]);np.testing.assert_array_equal(image[..., 0], image[..., 2])

    def test_bad_shapes_nonfinite_gripper_and_implicit_frame_selection_are_rejected(self):
        spec = composition();sample = self.sample(spec);state_key = spec["state_fields"][0]["key"];image_key = spec["cameras"][0]["key"]
        mutations = [lambda d: d.update(actions=np.zeros((15, 10))), lambda d: d.update(actions=np.zeros((16, 11))),
                     lambda d: d["actions"].__setitem__((0, 0), np.nan), lambda d: d["actions"].__setitem__((0, 9), 1.0),
                     lambda d: d.update({state_key: np.zeros((3, 3))}), lambda d: d.update({image_key: np.zeros((2, 3, 224, 224))}),
                     lambda d: d.update({image_key: np.full((3, 224, 224), 2.0)}), lambda d: d.update(task=["bad"])]
        for index, mutate in enumerate(mutations):
            bad = deepcopy(sample);mutate(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):self.apply(spec, bad)
        with self.assertRaises(ValueError):self.apply(spec, sample, action_dim=9)

    def test_missing_gripper_range_skips_only_bounds_checks_and_keeps_real_values(self):
        spec = composition();spec.pop("gripper_range")
        for key in ("position_unit", "joint_unit", "coordinate_frame"):spec.pop(key)
        sample = self.sample(spec)
        state_key = next(field["key"] for field in spec["state_fields"] if field["kind"] == "gripper")
        sample[state_key][:] = 1200.0
        sample["actions"][:, -1] = -50.0
        output = self.apply(spec, sample)
        np.testing.assert_array_equal(output["state"][:, -1], [1200.0, 1200.0])
        self.assertIs(output["actions"], sample["actions"])
        for field in (state_key, "actions"):
            for nonfinite in (np.nan, np.inf, -np.inf):
                bad = deepcopy(sample);bad[field].flat[-1] = nonfinite
                with self.subTest(field=field, value=nonfinite), self.assertRaisesRegex(ValueError, "有限实数"):
                    self.apply(spec, bad)
        spec["gripper_range"] = None
        np.testing.assert_array_equal(self.apply(spec, sample)["state"], output["state"])

    def test_declared_gripper_range_checks_both_state_and_actions(self):
        spec = composition();sample = self.sample(spec)
        state_key = next(field["key"] for field in spec["state_fields"] if field["kind"] == "gripper")
        for field in (state_key, "actions"):
            for outside in (-0.1, 0.2):
                bad = deepcopy(sample);bad[field].flat[-1] = outside
                with self.subTest(field=field, value=outside), self.assertRaisesRegex(ValueError, "超出显式声明范围"):
                    self.apply(spec, bad)
        sample[state_key][0, 0] = 0.0;sample[state_key][-1, 0] = 0.1
        sample["actions"][0, -1] = 0.0;sample["actions"][-1, -1] = 0.1
        self.apply(spec, sample)

    def test_automatic_prompt_prefers_task_then_prompt_and_allows_no_language(self):
        spec = composition();sample = self.sample(spec);spec["prompt_key"] = ""
        cases = [({"task": "task text", "prompt": "fallback"}, "task text"),
                 ({"task": np.asarray("scalar task"), "prompt": "fallback"}, "scalar task"),
                 ({"prompt": "prompt text"}, "prompt text"),
                 ({"task": " \t", "prompt": np.asarray("scalar fallback")}, "scalar fallback"),
                 ({"task": "", "prompt": " \t"}, ""), ({}, "")]
        for candidates, expected in cases:
            data = {key: value for key, value in sample.items() if key != "task"};data.update(candidates)
            with self.subTest(candidates=candidates):
                output = self.apply(spec, data)
                self.assertEqual(output["prompt"], expected)
                self.assertIs(output["actions"], data["actions"])

    def test_automatic_prompt_rejects_bad_types_instead_of_stringifying_or_falling_back(self):
        spec = composition();sample = self.sample(spec);spec["prompt_key"] = None
        for key in ("task", "prompt"):
            for invalid in (1, False, None, {}, [], ["text"], np.asarray(["text"]), np.asarray(1)):
                bad = {name: value for name, value in sample.items() if name != "task"}
                bad[key] = invalid
                if key == "task":bad["prompt"] = "must not hide invalid task"
                with self.subTest(key=key, value=invalid), self.assertRaisesRegex(ValueError, "标量文本"):
                    self.apply(spec, bad)

    def test_automatic_prompt_skips_task_and_prompt_claimed_as_numeric_sources(self):
        spec = composition();sample = self.sample(spec);spec["prompt_key"] = ""
        spec["action_key"] = "task";sample["task"] = sample.pop("actions")
        sample["prompt"] = "free prompt"
        self.assertEqual(self.apply(spec, sample)["prompt"], "free prompt")
        old_key = spec["state_fields"][0]["key"]
        spec["state_fields"][0]["key"] = "prompt";sample["prompt"] = sample.pop(old_key)
        self.assertEqual(self.apply(spec, sample)["prompt"], "")
        spec = composition();sample = self.sample(spec);spec["prompt_key"] = ""
        old_key = spec["cameras"][0]["key"]
        spec["cameras"][0]["key"] = "task";sample["task"] = sample.pop(old_key)
        sample["prompt"] = "camera name does not imply text"
        self.assertEqual(self.apply(spec, sample)["prompt"], sample["prompt"])

    def test_automatic_prompt_skips_namespaces_claimed_by_nested_or_literal_slash_sources(self):
        for nested in (True, False):
            spec = composition();sample = self.sample(spec);spec["prompt_key"] = ""
            state_key = spec["state_fields"][0]["key"]
            state = sample.pop(state_key);actions = sample.pop("actions")
            spec["state_fields"][0]["key"] = "task/joints";spec["action_key"] = "prompt/actions"
            sample["task"] = {"joints": state};sample["prompt"] = {"actions": actions}
            if not nested:
                sample["task/joints"] = state;sample["prompt/actions"] = actions
                # The declared literal source still reserves the parent namespace.
                sample["task"] = 42;sample["prompt"] = np.asarray([1, 2])
            with self.subTest(nested=nested):
                output = self.apply(spec, sample)
                self.assertEqual(output["prompt"], "")
                np.testing.assert_array_equal(output["state"][:, :3], state)
                self.assertIs(output["actions"], actions)

    def test_explicit_prompt_never_falls_back_or_accepts_empty_text(self):
        spec = composition();sample = self.sample(spec);sample["prompt"] = "fallback"
        for value in ("", " \t", 1, None, np.asarray(["text"])):
            bad = dict(sample);bad["task"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "文本"):
                self.apply(spec, bad)
        sample.pop("task")
        with self.assertRaisesRegex(ValueError, "样本缺少声明的数据字段"):
            self.apply(spec, sample)
        spec["prompt_key"] = "language/text";sample["language"] = {"text": np.asarray("nested explicit")}
        self.assertEqual(self.apply(spec, sample)["prompt"], "nested explicit")

    def test_auxiliary_labels_are_explicit_and_numeric(self):
        spec = composition();spec["aux_keys"] = ["value_target", "is_positive"]
        sample = self.sample(spec);sample.update(value_target=np.asarray([-0.5]), is_positive=np.asarray([1]))
        self.assertEqual(self.apply(spec, sample)["value_target"][0], -0.5)
        sample["is_positive"] = np.asarray([2])
        with self.assertRaises(ValueError):self.apply(spec, sample)


@unittest.skipIf(np is None, "NumPy required for real pipeline checks")
class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name);(self.repo / "meta").mkdir()
        self.tokenizer_calls = []
        namespace = {"np": np, "dataclasses": dataclasses, "random": random, "DataTransformFn": object,
                     "image_tools": types.SimpleNamespace(resize_with_pad=lambda image, h, w: image),
                     "ACP_POSITIVE_TAG": "Advantage: positive", "ACP_NEGATIVE_TAG": "Advantage: negative"}
        actual_definitions(ROOT / "src/openpi/transforms.py", {
            "Group", "InjectDefaultPrompt", "ResizeImages", "TokenizePrompt", "PadActionsOnly", "PadStatesAndActions",
            "FlattenState", "KeepModelKeys", "ACPConditionPrompt", "InjectActionLossMask", "build_acp_tagged_task", "pad_to_dim"}, namespace)
        namespace["MODEL_KEYS"] = {"state", "actions", "image", "image_mask", "tokenized_prompt", "tokenized_prompt_mask", "value_target"}
        transforms = types.ModuleType("openpi.transforms")
        transforms.__dict__.update(namespace)
        transforms.TokenizeFASTInputs = type("TokenizeFASTInputs", (), {})
        transforms.TokenizeHybridInput = type("TokenizeHybridInput", (), {})
        @dataclasses.dataclass
        class Pi0Config:
            action_dim: int = 32
            action_horizon: int = 16
            max_token_len: int = 32
            pi05: bool = True
            discrete_state_input: bool = False
            image_keys: tuple[str, ...] | None = None
            depth_keys: tuple[str, ...] | None = None
        @dataclasses.dataclass
        class Pi0GripperConfig(Pi0Config):
            gripper_binary_indices: tuple[int, ...] | None = None
        class Pi0ValueConfig(Pi0Config):pass
        class Pi0MemConfig(Pi0Config):pass
        self.models = types.SimpleNamespace(Pi0Config=Pi0Config, Pi0ValueConfig=Pi0ValueConfig,
                                            Pi0GripperConfig=Pi0GripperConfig, Pi0MemConfig=Pi0MemConfig)
        @dataclasses.dataclass(frozen=True)
        class DataConfig:
            repo_id: str
            repack_transforms: object = None
            data_transforms: object = None
            model_transforms: object = None
            normalize_masks: object = None
            use_quantile_norm: bool = False
            action_sequence_keys: tuple = ()
            action_loss_mask: tuple | None = None
            prompt_from_task: bool = False
            robot_type: str | None = None
        self.base = DataConfig(str(self.repo), robot_type="ARM=1,G=0,H=0")
        calls = self.tokenizer_calls
        class PaligemmaTokenizer:
            def __init__(self, max_len):self.max_len = max_len
            def tokenize(self, prompt, state=None, robot_type=None):
                calls.append({"prompt": prompt, "state": state, "robot_type": robot_type})
                return np.asarray([1, 2], dtype=np.int32), np.asarray([True, True])
        package = types.ModuleType("openpi");package.__path__ = [];package.transforms = transforms
        models = types.ModuleType("openpi.models");models.__path__ = [];package.models = models
        modules = {"openpi": package, "openpi.transforms": transforms, "openpi.models": models}
        for name, value in (("pi0_config", Pi0Config), ("pi0_gripper", Pi0GripperConfig), ("pi0_value", Pi0ValueConfig), ("tokenizer", PaligemmaTokenizer)):
            module = types.ModuleType("openpi.models." + name)
            setattr(module, "PaligemmaTokenizer" if name == "tokenizer" else value.__name__, value)
            setattr(models, name, module);modules[module.__name__] = module
        self.patch = patch.dict(sys.modules, modules);self.patch.start();self.addCleanup(self.patch.stop)
        self.transforms = transforms
        shared = {"dataclasses": dataclasses, "_transforms": transforms}
        actual_definitions(ROOT / "src/openpi/training/config.py", {"_set_robot_type"}, shared)
        self.set_robot_type = shared["_set_robot_type"]

    def build(self, spec=None, *, model=None, stage="sft", dropout=0.0, action_shape=None):
        spec = deepcopy(spec if spec is not None else composition())
        model = model if model is not None else self.models.Pi0Config()
        shape = action_shape if action_shape is not None else [model.action_horizon, sum(x["width"] for x in spec["action_layout"])]
        (self.repo / "meta/info.json").write_text(json.dumps({"features": {spec["action_key"]: {"shape": shape}}}))
        factory = types.SimpleNamespace(repo_id=str(self.repo), composition=spec, training_stage=stage, acp_dropout=dropout,
                                        create_base_config=lambda *args: self.base)
        return COMPOSED.compose_data_config(factory, self.repo, model)

    def sample(self, spec):return NumericTests.sample(self, spec)

    def test_builder_keeps_paths_masks_norm_mode_and_loader_chunk_shape_explicit(self):
        spec = composition();actual = self.build(spec)
        self.assertEqual(actual.repo_id, str(self.repo));self.assertEqual(actual.normalize_masks, COMPOSED.normalization_masks(spec))
        self.assertTrue(actual.use_quantile_norm);self.assertEqual(actual.action_sequence_keys, ())
        spec["normalization"] = "zscore"
        self.assertFalse(self.build(spec, action_shape=[10]).use_quantile_norm)
        self.assertEqual(self.build(spec, action_shape=[10]).action_sequence_keys, ("actions",))
        with self.assertRaises(ValueError):self.build(spec, action_shape=[15, 10])
        with self.assertRaises(ValueError):self.build(spec, action_shape=[16, 9])

    def test_builder_accepts_optional_declarations_without_changing_training_contract(self):
        for factory in (composition, joint_composition):
            explicit = factory();expected = self.build(explicit)
            spec = {key: value for key, value in explicit.items()
                    if key not in {"position_unit", "joint_unit", "coordinate_frame", "gripper_range",
                                   "action_mode", "rotation_representation", "prompt_key"}}
            actual = self.build(spec)
            self.assertEqual(actual.normalize_masks, expected.normalize_masks)
            self.assertEqual(actual.action_loss_mask, expected.action_loss_mask)
            self.assertEqual(actual.action_sequence_keys, expected.action_sequence_keys)
            self.assertEqual(actual.use_quantile_norm, expected.use_quantile_norm)
            output = actual.data_transforms.inputs[0](actual.repack_transforms.inputs[0](self.sample(explicit)))
            self.assertEqual(output["state"].shape, (spec["state_history"], sum(field["width"] for field in spec["state_fields"])))
            self.assertEqual(output["actions"].shape, (16, sum(field["width"] for field in spec["action_layout"])))

    def test_automatic_prompt_reaches_tokenizer_without_forcing_lerobot_task_lookup(self):
        spec = composition();sample = self.sample(spec)
        self.assertTrue(self.build(spec).prompt_from_task)
        for explicit_key in ("", None):
            spec["prompt_key"] = explicit_key;config = self.build(spec)
            self.assertFalse(config.prompt_from_task)
            for keep_task in (True, False):
                data = deepcopy(sample)
                if not keep_task:data.pop("task")
                output = config.data_transforms.inputs[0](config.repack_transforms.inputs[0](data))
                for transform in config.model_transforms.inputs:output = transform(output)
                self.assertEqual(self.tokenizer_calls[-1]["prompt"], "move the cloth" if keep_task else "")
                self.assertIn("tokenized_prompt", output)

    def test_runtime_contract_is_json_and_does_not_load_norm_or_tokenizer(self):
        import hashlib
        spec = composition();model = self.models.Pi0Config()
        source = self.repo / "meta/info.json"
        source.write_text(json.dumps({"features": {"actions": {"shape": [10]}}}))
        factory = types.SimpleNamespace(repo_id=str(self.repo), composition=spec, training_stage="sft", acp_dropout=0.1,
            create_base_config=lambda *args: self.fail("lightweight contract must not load norm assets"))
        models_package = sys.modules["openpi.models"]
        tokenizer_module = models_package.tokenizer
        del models_package.tokenizer
        try:
            with patch.dict(sys.modules, {"openpi.models.tokenizer": None}):
                result = COMPOSED.composition_runtime_contract(factory, model)
        finally:
            models_package.tokenizer = tokenizer_module
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(result, {"training_stage": "sft", "model_type": "Pi0Config", "pi05": True,
            "action_dim": 32, "action_horizon": 16, "raw_state_dim": 16, "raw_action_dim": 10,
            "state_history": 2, "normalization": "quantile", "acp_dropout": 0.1,
            "action_loss_mask": [1.0] * 10 + [0.0] * 22,
            "image_keys": list(COMPOSED.SLOTS), "depth_keys": [],
            "action_sequence_keys": ["actions"], "action_metadata_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        source.write_text(json.dumps({"features": {"actions": {"shape": [16, 10]}}}))
        self.assertEqual(COMPOSED.composition_runtime_contract(factory, model)["action_sequence_keys"], [])
        for unsupported in (type("Pi0Config", (), {"action_dim": 32, "action_horizon": 16})(),
                            self.models.Pi0MemConfig()):
            with self.subTest(model=type(unsupported).__name__), self.assertRaises(ValueError):
                COMPOSED.composition_runtime_contract(factory, unsupported)

    def test_real_model_transforms_pad_only_after_normalization_and_retain_tokenizer_tag(self):
        spec = composition();config = self.build(spec);sample = self.sample(spec)
        output = config.repack_transforms.inputs[0](sample)
        output = config.data_transforms.inputs[0](output)
        self.assertEqual(output["state"].shape, (2, 16));self.assertEqual(output["actions"].shape, (16, 10))
        config = self.set_robot_type(config, config.robot_type)
        for transform in config.model_transforms.inputs:output = transform(output)
        self.assertEqual(output["state"].shape, (32,));self.assertEqual(output["actions"].shape, (16, 32))
        np.testing.assert_array_equal(output["actions"][:, :10], sample["actions"])
        np.testing.assert_array_equal(output["actions"][:, 10:], 0)
        self.assertEqual(self.tokenizer_calls[0]["robot_type"], "ARM=1,G=0,H=0")
        self.assertNotIn("prompt", output)

    def test_pi0_flattens_and_pads_state_without_silently_truncating(self):
        spec = joint_composition();model = self.models.Pi0Config(pi05=False)
        config = self.build(spec, model=model);sample = self.sample(spec)
        output = config.data_transforms.inputs[0](config.repack_transforms.inputs[0](sample))
        for transform in config.model_transforms.inputs:output = transform(output)
        self.assertEqual(output["state"].shape, (32,));self.assertEqual(output["actions"].shape, (16, 32))
        spec["state_history"] = 6
        with self.assertRaisesRegex(ValueError, "截断"):self.build(spec, model=model)

    def test_stage_auxiliary_contract_and_existing_acp_conditioning_are_preserved(self):
        spec = composition();spec["aux_keys"] = ["is_positive"]
        config = self.build(spec, stage="offline_rl");sample = self.sample(spec);sample["is_positive"] = np.asarray([1])
        output = config.data_transforms.inputs[0](config.repack_transforms.inputs[0](sample))
        for transform in config.model_transforms.inputs:output = transform(output)
        self.assertIn("Advantage: positive", self.tokenizer_calls[-1]["prompt"])
        self.assertNotIn("is_positive", output)  # The existing ACP transform consumes its indicator.
        spec["aux_keys"] = ["value_target"];model = self.models.Pi0ValueConfig()
        config = self.build(spec, model=model, stage="value");sample = self.sample(spec);sample["value_target"] = np.asarray([-0.5])
        output = config.data_transforms.inputs[0](config.repack_transforms.inputs[0](sample))
        for transform in config.model_transforms.inputs:output = transform(output)
        np.testing.assert_array_equal(output["value_target"], [-0.5])
        for stage in ("value", "offline_rl", "mem", ["sft"]):
            with self.subTest(stage=stage), self.assertRaises(ValueError):self.build(composition(), stage=stage)
        with self.assertRaises(ValueError):self.build(spec, model=self.models.Pi0Config(), stage="value")
        with self.assertRaises(ValueError):self.build(spec, model=model, stage="sft")
        for value in (True, -0.1, 1.1, float("nan")):
            with self.subTest(dropout=value), self.assertRaises(ValueError):self.build(composition(), dropout=value)

    def test_memory_hybrid_and_small_action_models_are_rejected(self):
        for model in (self.models.Pi0MemConfig(), types.SimpleNamespace(action_dim=32, action_horizon=16), self.models.Pi0Config(action_dim=9)):
            with self.subTest(model=type(model).__name__), self.assertRaises(ValueError):self.build(model=model)

    def test_dataset_loss_mask_covers_all_real_dimensions_and_overrides_stale_masks(self):
        spec = composition(bimanual=True)
        self.base = dataclasses.replace(self.base, action_loss_mask=(1.0,) * 10 + (0.0,) * 22)
        config = self.build(spec)
        self.assertEqual(config.action_loss_mask, (1.0,) * 20 + (0.0,) * 12)
        sample = self.sample(spec)
        output = config.data_transforms.inputs[0](config.repack_transforms.inputs[0](sample))
        for transform in config.model_transforms.inputs:output = transform(output)
        output = self.transforms.InjectActionLossMask(config.action_loss_mask)(output)
        np.testing.assert_array_equal(output["action_loss_mask"], [1.0] * 20 + [0.0] * 12)
        joint = joint_composition();joint["action_layout"][0]["width"] = 24
        self.assertEqual(self.build(joint).action_loss_mask, (1.0,) * 24 + (0.0,) * 8)

    def test_gripper_head_requires_explicit_real_layout_indices_and_raw_gripper_units(self):
        spec = composition(bimanual=True)
        model = self.models.Pi0GripperConfig(gripper_binary_indices=(9, 19))
        self.assertEqual(self.build(spec, model=model).action_loss_mask, (1.0,) * 20 + (0.0,) * 12)
        for indices in (None, (9,), (19, 9), (9, 9), (9, 20), (-1, 19), (True, 19), "9,19"):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "gripper_binary_indices"):
                self.build(spec, model=self.models.Pi0GripperConfig(gripper_binary_indices=indices))
        spec["action_layout"][2]["normalize"] = True
        with self.assertRaisesRegex(ValueError, "禁止归一化"):self.build(spec, model=model)
        spec = joint_composition();spec["action_layout"].append({"arm": "left", "kind": "gripper", "width": 2, "normalize": False})
        model = self.models.Pi0GripperConfig(gripper_binary_indices=(6, 7))
        self.assertEqual(self.build(spec, model=model).action_loss_mask, (1.0,) * 8 + (0.0,) * 24)

    def test_model_views_must_use_all_declared_canonical_cameras_and_match_depth_semantics(self):
        spec = composition()
        for image_keys, depth_keys in ((("base_0_rgb",), None), (("camera0_rgb",), None),
                                     (("left_wrist_0_rgb", "left_wrist_0_rgb"), None),
                                     (("left_wrist_0_rgb",), ("left_wrist_0_rgb",)),
                                     (None, ("left_wrist_0_rgb",)), ((), ())):
            with self.subTest(images=image_keys, depth=depth_keys), self.assertRaises(ValueError):
                self.build(spec, model=self.models.Pi0Config(image_keys=image_keys, depth_keys=depth_keys))
        # A subset is valid when it includes every actual asset camera.
        self.build(spec, model=self.models.Pi0Config(image_keys=("left_wrist_0_rgb",)))
        spec = composition(bimanual=True)
        with self.assertRaisesRegex(ValueError, "所有实际视图"):
            self.build(spec, model=self.models.Pi0Config(image_keys=("left_wrist_0_rgb",)))
        spec = composition();spec["cameras"][0]["kind"] = "depth"
        with self.assertRaisesRegex(ValueError, "depth_keys"):self.build(spec)
        model = self.models.Pi0Config(depth_keys=("left_wrist_0_rgb",))
        self.build(spec, model=model)
        factory = types.SimpleNamespace(repo_id=str(self.repo), composition=spec, training_stage="sft", acp_dropout=0.1)
        contract = COMPOSED.composition_runtime_contract(factory, model)
        self.assertEqual(contract["image_keys"], ["base_0_rgb", "right_wrist_0_rgb"])
        self.assertEqual(contract["depth_keys"], ["left_wrist_0_rgb"])

    def test_action_metadata_reader_rejects_symlink_fifo_and_oversized_json(self):
        source = self.repo / "meta/info.json";other = self.repo / "original.json"
        other.write_text(json.dumps({"features": {"actions": {"shape": [16, 10]}}}))
        source.symlink_to(other)
        with self.assertRaises(ValueError):COMPOSED._action_sequence_keys(str(self.repo), "actions", 16, 10)
        source.unlink();source.write_text(" " * (1024 * 1024 + 1))
        with self.assertRaises(ValueError):COMPOSED._action_sequence_keys(str(self.repo), "actions", 16, 10)
        source.unlink()
        import os
        os.mkfifo(source)
        with self.assertRaises(ValueError):COMPOSED._action_sequence_keys(str(self.repo), "actions", 16, 10)


if __name__ == "__main__":unittest.main()
