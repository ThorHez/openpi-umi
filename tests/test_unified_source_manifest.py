"""Standard-library regression tests for the source-unification audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_unified_sources", ROOT / "scripts/check_unified_sources.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
PRODUCTION = '''_CONFIGS = [TrainConfig(
    name="production_sft",
    data=DataConfig(repo_id="/datasets/production"),
    weight_loader=Weights("/models/production"),
    num_train_steps=60000,
)]
'''


class SourceManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "src/openpi/training/config.py"
        self.policy = self.root / "src/openpi/policies/umi_policy.py"
        self.config.parent.mkdir(parents=True)
        self.policy.parent.mkdir(parents=True)
        self.config.write_text(PRODUCTION)
        self.policy.write_text("class UmiInputs: pass\n")
        self.manifest = {
            "schema_version": 1,
            "production_recipe_baseline": {
                "config_path": "src/openpi/training/config.py",
                "recipe_count": 1,
                "fingerprint_algorithm": CHECKER.FINGERPRINT_ALGORITHM,
                "recipes": CHECKER.recipe_fingerprints(PRODUCTION),
            },
            "validation": {
                "scoped_paths": ["src/openpi/training/config.py", "src/openpi/policies/umi_policy.py"],
                "syntax_paths": ["src/openpi/training/config.py"],
                "conflict_marker_globs": ["src/**/*.py"],
            },
        }

    def test_whitespace_and_comments_do_not_change_recipe_fingerprints(self):
        reformatted = PRODUCTION.replace("num_train_steps=60000,", "num_train_steps = 60_000,  # preserve steps")
        self.assertEqual(CHECKER.recipe_fingerprints(PRODUCTION), CHECKER.recipe_fingerprints(reformatted))

    def test_dataset_weight_and_training_step_drift_are_rejected(self):
        for before, after in (("/datasets/production", "/datasets/other"),
                              ("/models/production", "/models/other"),
                              ("60000", "30000")):
            with self.subTest(changed=before):
                self.config.write_text(PRODUCTION.replace(before, after))
                result = CHECKER.validate(self.root, self.manifest)
                self.assertFalse(result["ok"])
                self.assertIn("production recipe drift: production_sft", result["errors"])

    def test_new_recipe_is_allowed_but_missing_production_recipe_is_rejected(self):
        self.config.write_text(PRODUCTION + '\nextra = TrainConfig(name="new_uot", num_train_steps=20)\n')
        self.assertTrue(CHECKER.validate(self.root, self.manifest)["ok"])
        self.config.write_text('extra = TrainConfig(name="new_uot", num_train_steps=20)\n')
        result = CHECKER.validate(self.root, self.manifest)
        self.assertIn("missing production recipe: production_sft", result["errors"])

    def test_duplicate_recipe_name_is_rejected(self):
        self.config.write_text(PRODUCTION + '\nextra = TrainConfig(name="production_sft")\n')
        result = CHECKER.validate(self.root, self.manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any("duplicate recipe name: production_sft" in message for message in result["errors"]))

    def test_undefined_adapter_reference_is_rejected(self):
        self.config.write_text(PRODUCTION + '\nprocessor = umi_policy.MissingInputs()\n')
        result = CHECKER.validate(self.root, self.manifest)
        self.assertIn("undefined umi_policy reference: MissingInputs", result["errors"])

    def split_sources(self):
        part = self.config.parent / "recipes/sft.py"
        part.parent.mkdir()
        part.write_text(PRODUCTION)
        self.config.write_text("from openpi.training.recipes.sft import _CONFIGS\n")
        self.manifest["production_recipe_baseline"]["config_paths"] = [
            "src/openpi/training/config.py", "src/openpi/training/recipes/sft.py"]
        return part

    def test_split_source_preserves_full_recipe_fingerprints(self):
        part = self.split_sources()
        self.assertTrue(CHECKER.validate(self.root, self.manifest)["ok"])
        part.write_text(PRODUCTION.replace("60000", "30000"))
        self.assertIn("production recipe drift: production_sft", CHECKER.validate(self.root, self.manifest)["errors"])

    def test_split_sources_reject_duplicate_names_and_undefined_adapters(self):
        part = self.split_sources()
        self.config.write_text(PRODUCTION)
        result = CHECKER.validate(self.root, self.manifest)
        self.assertTrue(any("duplicate recipe name across" in error for error in result["errors"]))
        self.config.write_text("_CONFIGS = []\n")
        part.write_text(PRODUCTION + "\nprocessor = umi_policy.MissingInPart()\n")
        self.assertIn("undefined umi_policy reference: MissingInPart", CHECKER.validate(self.root, self.manifest)["errors"])

    def test_complete_split_baseline_protects_nonproduction_experiment_recipes(self):
        part = self.split_sources()
        extra = "experiment = TrainConfig(name='experiment', num_train_steps=100)\n"
        part.write_text(PRODUCTION + extra)
        self.manifest["recipe_split_baseline"] = {
            "recipe_count": 2, "fingerprint_algorithm": CHECKER.FINGERPRINT_ALGORITHM,
            "recipes": CHECKER.recipe_fingerprints(PRODUCTION + extra)}
        self.assertTrue(CHECKER.validate(self.root, self.manifest)["ok"])
        part.write_text(PRODUCTION + extra.replace("100", "200"))
        self.assertIn("split-baseline recipe drift: experiment", CHECKER.validate(self.root, self.manifest)["errors"])

    def test_split_source_paths_cannot_escape_repository(self):
        self.manifest["production_recipe_baseline"]["config_paths"] = ["../outside.py"]
        self.assertFalse(CHECKER.validate(self.root, self.manifest)["ok"])

    def test_conflict_markers_are_rejected_without_parsing_old_experiments(self):
        experiment = self.root / "src/old_experiment.py"
        experiment.write_text('this is an old syntax error ???\n')
        self.assertTrue(CHECKER.validate(self.root, self.manifest)["ok"])
        experiment.write_text('<<<<<<< old\nvalue = 1\n=======\nvalue = 2\n>>>>>>> new\n')
        result = CHECKER.validate(self.root, self.manifest)
        self.assertIn("unresolved merge conflict marker: src/old_experiment.py", result["errors"])


if __name__ == "__main__":
    unittest.main()
