"""Smoke tests for Phase 1 — project foundation.

Verifies:
1. All package modules are importable.
2. Configuration loads from defaults.
3. Configuration loads from YAML.
4. Logging setup works.
5. Reproducibility seed control works.
6. Project directories can be created.
"""

import os
import tempfile
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# Test 1: Package imports
# ------------------------------------------------------------------

def test_import_config():
    from config import settings
    assert hasattr(settings, "Config")
    assert hasattr(settings, "cfg")


def test_import_utils_logging():
    from utils.logging_setup import get_logger, setup_logging
    assert callable(get_logger)
    assert callable(setup_logging)


def test_import_utils_reproducibility():
    from utils.reproducibility import set_seed, get_device
    assert callable(set_seed)
    assert callable(get_device)


def test_import_module_packages():
    """All module packages should be importable (even if empty)."""
    import retrieval
    import verification
    import attribution
    import intervention
    import fuzzy
    import conflict
    import stability
    import evaluation
    import app


# ------------------------------------------------------------------
# Test 2: Configuration defaults
# ------------------------------------------------------------------

def test_config_defaults():
    from config.settings import Config
    cfg = Config()
    assert cfg.dataset_name == "fever"
    assert cfg.retrieval_mode == "hybrid"
    assert cfg.top_k == 5
    assert cfg.random_seed == 42
    assert cfg.device == "cpu"
    assert cfg.debug is False


# ------------------------------------------------------------------
# Test 3: Configuration from YAML
# ------------------------------------------------------------------

def test_config_from_yaml(tmp_path):
    from config.settings import Config
    yaml_content = "dataset_name: scifact\ntop_k: 10\ndebug: true\n"
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = Config.from_yaml(str(yaml_file))
    assert cfg.dataset_name == "scifact"
    assert cfg.top_k == 10
    assert cfg.debug is True
    # Unspecified fields keep defaults
    assert cfg.random_seed == 42


def test_config_to_yaml(tmp_path):
    from config.settings import Config
    cfg = Config(dataset_name="test_ds", top_k=3)
    out_path = str(tmp_path / "out_config.yaml")
    cfg.to_yaml(out_path)
    assert Path(out_path).exists()

    loaded = Config.from_yaml(out_path)
    assert loaded.dataset_name == "test_ds"
    assert loaded.top_k == 3


# ------------------------------------------------------------------
# Test 4: Logging
# ------------------------------------------------------------------

def test_logging_produces_logger():
    from utils.logging_setup import get_logger
    logger = get_logger("test_smoke")
    assert logger.name == "test_smoke"
    # Should not raise
    logger.info("Smoke test log message.")


# ------------------------------------------------------------------
# Test 5: Reproducibility
# ------------------------------------------------------------------

def test_set_seed_deterministic():
    import random as rng
    from utils.reproducibility import set_seed

    set_seed(123)
    a = rng.random()
    set_seed(123)
    b = rng.random()
    assert a == b


def test_get_device_fallback():
    from utils.reproducibility import get_device
    # CPU should always be available
    assert get_device("cpu") == "cpu"
    # Invalid device falls back to CPU
    assert get_device("nonexistent") == "cpu"


# ------------------------------------------------------------------
# Test 6: Directory creation
# ------------------------------------------------------------------

def test_ensure_dirs(tmp_path):
    from config.settings import Config
    cfg = Config(
        project_root=str(tmp_path),
        data_dir="data",
        cache_dir="cache",
        output_dir="outputs",
    )
    cfg.ensure_dirs()
    assert Path(cfg.data_dir).exists()
    assert Path(cfg.cache_dir).exists()
    assert Path(cfg.output_dir).exists()
