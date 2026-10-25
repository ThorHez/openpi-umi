"""Recipe split equivalence and real import ordering with training backends stubbed.

The subprocess probes execute the actual facade, dataclasses, recipe files and
RoboArena expansion. Only optional model/transform/backend dependencies and the
three pre-existing task factories are replaced by inert CPU records. This is
Python registry verification, not a model or GPU training acceptance test.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/openpi/training/config.py"
FAMILIES = ("sft", "recap", "mem", "rlt", "uot")
SOURCES = (CONFIG, *(CONFIG.parent / "recipes" / (name + ".py") for name in FAMILIES))


def registry(path):
    tree = ast.parse(path.read_text())
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "_CONFIGS" for target in node.targets))


def fingerprint(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


class RecipeSplitTests(unittest.TestCase):
    def test_all_original_literal_recipes_and_registry_elements_are_preserved(self):
        baseline = json.loads((ROOT / "source_unification.json").read_text())["recipe_split_baseline"]
        found, elements = {}, []
        for source in SOURCES[1:]:
            values = registry(source).elts
            elements.extend(fingerprint(node) for node in values)
            for node in values:
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig":
                    name = next(keyword.value.value for keyword in node.keywords if keyword.arg == "name")
                    self.assertNotIn(name, found)
                    found[name] = fingerprint(node)
        self.assertEqual(len(baseline["recipes"]), 122)
        for name, digest in baseline["recipes"].items():
            self.assertEqual(found.get(name), digest, name)
        missing = Counter(baseline["registry_element_fingerprints"]) - Counter(elements)
        self.assertEqual(missing, Counter())
        self.assertEqual(len(baseline["registry_element_fingerprints"]), 126)

    def test_training_families_follow_model_and_objective_instead_of_name_fragments(self):
        names = {}
        for family, path in zip(FAMILIES, SOURCES[1:]):
            names[family] = {keyword.value.value for node in registry(path).elts if isinstance(node, ast.Call)
                             for keyword in node.keywords if keyword.arg == "name"}
        self.assertIn("pi0_umi_low_mem_finetune", names["sft"])
        self.assertIn("pi0_fast_libero_low_mem_finetune", names["sft"])
        self.assertIn("pi0_value_umi", names["recap"])
        self.assertIn("pi05_acp_umi_bimanual_wbcd_multi_dataset", names["recap"])
        self.assertIn("pi0_mem_pf_safe_evan_shellgame_joint_260806", names["mem"])
        self.assertIn("pi05_umi_robustness", names["uot"])
        self.assertIn("pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta_uot", names["uot"])
        self.assertEqual(registry(SOURCES[4]).elts, [])
        for family in ("sft", "recap", "mem"):
            for node in registry(CONFIG.parent / "recipes" / (family + ".py")).elts:
                if isinstance(node, ast.Call):
                    self.assertFalse(any(keyword.arg == "uot" for keyword in node.keywords))

    def test_facade_and_each_recipe_module_can_be_imported_first(self):
        for first in ("config", *("recipes." + family for family in FAMILIES)):
            with self.subTest(first=first):
                result = subprocess.run([sys.executable, str(Path(__file__)), "--probe", first],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                evidence = json.loads(result.stdout)
                self.assertTrue(evidence["all_literal_names_present"])
                self.assertTrue(evidence["shared_objects_preserved"])
                self.assertTrue(evidence["cli_catalog_preserved"])
                self.assertTrue(evidence["duplicate_names_rejected"])
                self.assertEqual(evidence["rlt_recipes"], 0)


class _Symbol:
    def __init__(self, name, args=(), kwargs=None):
        self.name, self.args, self.kwargs = name, args, kwargs or {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Symbol(self.name + "." + name)

    def __call__(self, *args, **kwargs):
        return _Symbol(self.name + "()", args, kwargs)

    def __getitem__(self, _):
        return self

    def __or__(self, _):
        return self

    def __ror__(self, _):
        return self


class _Backend(types.ModuleType):
    def __getattr__(self, name):
        return _Symbol(self.__name__ + "." + name)


def _probe(first):
    def package(name, path=None, backend=False):
        module = (_Backend if backend else types.ModuleType)(name)
        module.__path__ = [str(path)] if path is not None else []
        module.__file__ = str(path / "__init__.py") if path is not None else "<inert-training-backend>"
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
        return module

    package("openpi", ROOT / "src/openpi")
    package("openpi.training", CONFIG.parent)
    for name in ("openpi.models", "openpi.policies", "openpi.shared", "etils", "flax"):
        package(name, backend=True)
    for name in ("etils.epath", "flax.nnx", "tyro", "openpi.transforms"):
        package(name, backend=True)
    tree = ast.parse(CONFIG.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            name = alias.name
            if name.startswith(("openpi.models.", "openpi.policies.", "openpi.shared.")) or name in {
                    "openpi.training.droid_rlds_dataset", "openpi.training.optimizer", "openpi.training.weight_loaders"}:
                package(name, backend=True)
    sys.modules["openpi.transforms"].make_bool_mask = lambda *runs: tuple(
        bit for length in runs for bit in [length > 0] * abs(length))
    package("openpi.training.mem", backend=True)
    package("openpi.training.mem.recipes", backend=True)
    factory_names = {
        "shellgame_semantic_action": "pi0_mem_semantic_action_shellgame_eef7",
        "shellgame_real_wrist_stage2": "pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2",
        "shellgame_real_memory_adapt": "pi0_mem_shellgame_real_relation_adapt_new75_old25",
    }
    for module_name, config_name in factory_names.items():
        module = package("openpi.training.mem.recipes." + module_name, backend=True)
        def make_config(name=config_name):
            from openpi.training.config import TrainConfig
            return TrainConfig(name=name)
        module.make_train_config = make_config
        module.data_config_type = lambda: _Symbol("existing_task_data_factory")
    importlib.import_module("openpi.training." + first)
    config = importlib.import_module("openpi.training.config")
    expected = json.loads((ROOT / "source_unification.json").read_text())["recipe_split_baseline"]["recipes"]
    all_names = set(config._CONFIGS_DICT)
    assert set(expected) <= all_names
    assert set(factory_names.values()) <= all_names
    assert "paligemma_binning_droid" in all_names
    assert config.TrainConfig.__module__ == "openpi.training.config"
    assert config.DataConfigFactory.__module__ == "openpi.training.config"
    assert issubclass(config.ComposableDataConfig, config.DataConfigFactory)
    assert config.ComposableDataConfig.COMPOSITION_SCHEMA_VERSION == 1
    assert callable(config.ComposableDataConfig.processing_contract)
    assert "COMPOSITION_SCHEMA_VERSION" not in {field.name for field in __import__("dataclasses").fields(config.ComposableDataConfig)}
    pf = config.get_config("pi0_mem_pf_safe_evan_shellgame_joint_260806")
    assert pf.model is config._pi0_mem_pf_safe_shellgame_joint_model
    first_query = config.get_config("pi0_mem_fixed_grid_query_action_shellgame_joint_260810")
    second_query = config.get_config("pi0_mem_fixed_grid_query_action_decision_weighted_shellgame_joint_260810")
    assert first_query.model is second_query.model is config._shellgame_joint_fixed_grid_query_action_model
    cli = config.cli()
    assert set(cli.args[0]) == all_names
    try:
        config.get_config("nonexistent_recipe")
    except ValueError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("Missing recipe was accepted")
    # Re-execute the real aggregation statements with an actual duplicate list
    # entry. The normal facade must continue to reject duplicate registrations.
    config._SFT_CONFIGS.append(config._RECAP_CONFIGS[0])
    aggregate = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "_CONFIGS" for target in node.targets))
    try:
        exec(compile(ast.Module(body=tree.body[aggregate:], type_ignores=[]), str(CONFIG), "exec"), config.__dict__)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Duplicate recipe names were accepted")
    print(json.dumps({"all_literal_names_present": True, "shared_objects_preserved": True,
                      "cli_catalog_preserved": True, "duplicate_names_rejected": True,
                      "rlt_recipes": len(config._RLT_CONFIGS)}))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        _probe(sys.argv[2])
    else:
        unittest.main()
