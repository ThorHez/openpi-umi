"""Configure training caches before importing libraries that cache their paths."""

import os
from pathlib import Path


def configure_training_cache() -> None:
    """Keep default caches on the repository's disk, respecting explicit overrides."""
    repo_cache = Path(__file__).resolve().parents[3] / ".cache"
    cache_root = Path(os.environ.setdefault("XDG_CACHE_HOME", str(repo_cache))).expanduser()
    hf_home = Path(os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))).expanduser()
    temp_dir = next(
        (os.environ[key] for key in ("TMPDIR", "TEMP", "TMP") if os.environ.get(key)),
        str(cache_root / "tmp"),
    )
    defaults = {
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")),
        "TMPDIR": temp_dir,
        "JAX_COMPILATION_CACHE_DIR": str(cache_root / "jax"),
        "OPENPI_DATA_HOME": str(cache_root / "openpi"),
    }
    for key, default in defaults.items():
        Path(os.environ.setdefault(key, default)).expanduser().mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP"):
        os.environ.setdefault(key, os.environ["TMPDIR"])
