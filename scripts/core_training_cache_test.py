"""Exercise core entry-point cache bootstraps without importing training stacks."""

import ast
import os
from pathlib import Path

import pytest


ENTRYPOINTS = (
    "scripts/mem/train_semantic_memory.py",
    "scripts/mem/train_pi0_mem.py",
    "scripts/mem/train_pi0_mem_compress.py",
    "scripts/pf/train_pi0_mem_pf.py",
    "scripts/pf/train_pi0_mem_pf_safe.py",
)
CACHE_KEYS = (
    "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE", "OPENPI_DATA_HOME", "JAX_COMPILATION_CACHE_DIR",
    "TMPDIR", "TEMP", "TMP",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("explicit_overrides", [False, True])
def test_core_bootstrap_uses_portable_cache_defaults_and_preserves_overrides(
    monkeypatch, tmp_path, entrypoint, explicit_overrides
):
    for key in CACHE_KEYS:
        monkeypatch.delenv(key, raising=False)
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    expected = {
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "OPENPI_DATA_HOME": str(cache / "openpi"),
        "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
        "TMPDIR": str(cache / "tmp"),
    }
    if explicit_overrides:
        expected = {key: str(tmp_path / key.lower()) for key in expected}
        for key, value in expected.items():
            monkeypatch.setenv(key, value)
    source = Path(__file__).resolve().parents[1] / entrypoint
    tree = ast.parse(source.read_text())
    prefix = []
    for node in tree.body:
        prefix.append(node)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "configure_training_cache"):
            break
    else:
        pytest.fail(f"{entrypoint} does not configure caches before training imports")
    # Execute the real bootstrap only, stopping before imports of JAX/models/data.
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(source), "exec"), {"__file__": str(source)})
    assert {key: os.environ[key] for key in expected} == expected
    assert os.environ["TEMP"] == os.environ["TMP"] == expected["TMPDIR"]
