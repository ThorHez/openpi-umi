"""Check retained UMI contracts without importing the JAX training runtime.

The numerical checks execute the actual adapter/transform definitions, extracted
from their modules to avoid loading optional training dependencies. NumPy is the
only dependency required for those checks.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "src/openpi/policies/umi_policy.py"
CONFIG = ROOT / "src/openpi/training/config.py"
RECIPE_SOURCES = (CONFIG, *sorted((CONFIG.parent / "recipes").glob("*.py")))
TRANSFORMS = ROOT / "src/openpi/transforms.py"
ADAPTERS = (
    "UmiArxInputsV5",
    "UmiArxInputs_Bimanual_v2",
    "UmiArxInputs_Bimanual_With_Depth",
    "UmiArxInputs_Bimanual_With_Depth_Ray",
    "UmiArxInputs_DeskHeight_Bimanual",
    "UmiArxInputs_Headview_Bimanual",
    "UmiArxInputs_Headview_DeskHeight_Bimanual",
)


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def mapping(class_name):
    tree = ast.parse(CONFIG.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    field = next(node for node in cls.body if isinstance(node, ast.AnnAssign) and node.target.id == "mapping")
    factory = next(keyword.value for keyword in field.value.keywords if keyword.arg == "default_factory")
    return ast.literal_eval(factory.body)


class FactoryContractTests(unittest.TestCase):
    def test_desk_height_mapping_matches_each_processor(self):
        names = (
            "LeRobotUmiDataConfig_Bimamual_DeskHeight_ImageHorizon1_Hybrid",
            "LeRobotUmiDataConfig_Bimamual_HeadView_DeskHeight_ImageHorizon1_Hybrid",
            "LeRobotUmiDataConfig_Bimamual_HeadView_DeskHeight_ImageHorizon2_Hybrid",
        )
        for name in names:
            with self.subTest(factory=name):
                keys = mapping(name)
                for robot in (0, 1):
                    key = f"robot{robot}_eef_desk_height"
                    self.assertEqual(keys[key], key)

    def test_every_referenced_umi_adapter_exists(self):
        policy_classes = {node.name for node in ast.parse(POLICY.read_text()).body if isinstance(node, ast.ClassDef)}
        missing = {
            node.attr
            for node in ast.walk(ast.parse(CONFIG.read_text()))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "umi_policy"
            and node.attr not in policy_classes
        }
        self.assertEqual(missing, set())

    def test_ray_recipe_uses_head_rgb_factory(self):
        recipe = next(
            node for source in RECIPE_SOURCES for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TrainConfig"
            and any(keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "pi05_umi_bimanual_infer_with_depth_ray" for keyword in node.keywords)
        )
        data = next(keyword.value for keyword in recipe.keywords if keyword.arg == "data")
        self.assertEqual(data.func.id, "LeRobotUmiDataConfig_Bimanual_Inference_With_Depth_Ray")
        self.assertEqual(mapping(data.func.id)["head_rgb"], "head_rgb")


@unittest.skipIf(np is None, "NumPy is required for adapter numerical checks")
class AdapterNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {
            "dataclasses": dataclasses,
            "np": np,
            "transforms": SimpleNamespace(DataTransformFn=object),
            "DataTransformFn": object,
            "einops": SimpleNamespace(rearrange=lambda value, pattern: np.transpose(value, (1, 2, 0))),
        }
        definitions(POLICY, ["_parse_image", *ADAPTERS], namespace)
        definitions(TRANSFORMS, ["Transform_depth_to_3ch_image"], namespace)
        cls.namespace = namespace

    def inputs(self, temporal_images=True):
        data = {}
        for robot in (0, 1):
            for suffix, size in (
                ("eef_pos", 3), ("eef_pos_wrt_start", 3), ("eef_rot_axis_angle", 6),
                ("eef_rot_axis_angle_wrt_start", 6), (f"eef_pos_wrt{1-robot}", 3),
                (f"eef_rot_axis_angle_wrt{1-robot}", 6), ("gripper_width", 1), ("eef_desk_height", 1),
            ):
                data[f"robot{robot}_{suffix}"] = np.full((2, size), robot + 1, dtype=np.float32)
        shape = (2, 3, 224, 224) if temporal_images else (3, 224, 224)
        for camera in range(3):
            data[f"camera{camera}_rgb"] = np.full(shape, 30 + camera, dtype=np.uint8)
        for key in ("head_rgb", "base_0_rgb", "base_0_depth"):
            data[key] = np.full((3, 224, 224), 70, dtype=np.uint8)
        return data

    def test_all_retained_adapters_emit_expected_state_and_images(self):
        contracts = (
            ("UmiArxInputsV5", 10, True, False),
            ("UmiArxInputs_Bimanual_v2", 56, True, False),
            ("UmiArxInputs_Bimanual_With_Depth", 56, False, True),
            ("UmiArxInputs_Bimanual_With_Depth_Ray", 56, False, True),
            ("UmiArxInputs_DeskHeight_Bimanual", 52, True, False),
            ("UmiArxInputs_Headview_Bimanual", 50, True, False),
            ("UmiArxInputs_Headview_DeskHeight_Bimanual", 52, True, False),
        )
        for name, state_dim, temporal, depth in contracts:
            with self.subTest(adapter=name):
                result = self.namespace[name]()(self.inputs(temporal))
                self.assertEqual(result["state"].shape, (2, state_dim))
                self.assertEqual(len(result["image"]), 4 if depth else 3)
                self.assertEqual(set(result["image"]), set(result["image_mask"]))
                for image in result["image"].values():
                    self.assertEqual(image.shape, (224, 224, 3))
                    self.assertEqual(image.dtype, np.uint8)
                self.assertIn("prompt", result)

    def test_depth_transform_accepts_equivalent_chw_hwc_and_hw_inputs(self):
        transform = self.namespace["Transform_depth_to_3ch_image"](depth_column_name="depth")
        depth = np.full((224, 224), 800, dtype=np.uint16)
        depth[0, 0] = 0
        expected = transform({"depth": depth})["depth"]
        for variant in (depth[..., None], depth[None, ...]):
            actual = transform({"depth": variant})["depth"]
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(expected.shape, (3, 224, 224))
        self.assertEqual(expected[2, 0, 0], 0)
        self.assertEqual(expected[2, 1, 1], 255)

    def test_ray_adapter_pads_depth_without_requesting_depth_input(self):
        data = self.inputs(temporal_images=False)
        del data["base_0_depth"]
        result = self.namespace["UmiArxInputs_Bimanual_With_Depth_Ray"]()(data)
        np.testing.assert_array_equal(result["image"]["base_0_depth"], 0)
        np.testing.assert_array_equal(result["image"]["base_0_rgb"], 70)


if __name__ == "__main__":
    unittest.main()
