"""Verify cache overrides without importing a training runtime."""

from pathlib import Path

from openpi.shared.runtime_paths import configure_training_cache


def test_training_cache_respects_explicit_paths_and_supports_dataset_locks(monkeypatch, tmp_path):
    for key in ("XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                "TMPDIR", "TEMP", "TMP", "JAX_COMPILATION_CACHE_DIR", "OPENPI_DATA_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "temporary"))
    configure_training_cache()
    import os

    assert os.environ["HF_DATASETS_CACHE"] == str(tmp_path / "datasets")
    assert os.environ["TMPDIR"] == str(tmp_path / "temporary")
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "cache/jax")
    lock = Path(os.environ["HF_DATASETS_CACHE"]) / "test.lock"
    with lock.open("x") as stream:
        stream.write("lock probe")
    lock.unlink()
    assert Path(os.environ["HF_HUB_CACHE"]).is_dir()
