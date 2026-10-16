"""
Smoke tests verifying environment, configuration, and imports.
"""

from pathlib import Path
import yaml


def test_config_exists_and_loads():
    """Verify config.yaml exists and contains required keys."""
    config_path = Path("config.yaml")
    assert config_path.exists(), "config.yaml must exist at repository root"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "location" in cfg
    assert "data" in cfg
    assert "splits" in cfg
    assert "climatology" in cfg
    assert cfg["location"]["target_latitude"] == 21.1458
    assert cfg["location"]["target_longitude"] == 79.0882
    assert cfg["data"]["years"]["start"] == 1981
    assert cfg["data"]["years"]["end"] == 2026


def test_pipeline_import():
    """Verify src.pipeline can be imported."""
    from src import pipeline
    assert hasattr(pipeline, "main")
    assert hasattr(pipeline, "load_config")
