"""Validated module configuration schema tests (#157)."""

from __future__ import annotations

import pytest

from visual_perception.config import ModuleConfig, QualityProfile, TilingConfig


def test_minimal_config_validates() -> None:
    config = ModuleConfig()
    assert config.quality_profile is QualityProfile.RESEARCH_QUALITY


def test_complete_config_validates() -> None:
    tiling = TilingConfig(multi_scale_enabled=True, tile_grid="2x2")
    config = ModuleConfig(gpu_memory_budget_gb=12.0, tiling=tiling)
    assert config.tiling.multi_scale_enabled


def test_rejects_non_positive_memory_budget() -> None:
    with pytest.raises(ValueError):
        ModuleConfig(gpu_memory_budget_gb=0)


def test_rejects_incompatible_reduced_cost_multi_scale() -> None:
    tiling = TilingConfig(multi_scale_enabled=True)
    with pytest.raises(ValueError):
        ModuleConfig(quality_profile=QualityProfile.REDUCED_COST, tiling=tiling)


def test_rejects_invalid_tile_grid() -> None:
    with pytest.raises(ValueError):
        TilingConfig(tile_grid="not-a-grid")


def test_config_round_trips_through_dict() -> None:
    config = ModuleConfig(tiling=TilingConfig(multi_scale_enabled=True, tile_grid="2x2"))
    restored = ModuleConfig.from_dict(config.to_dict())
    assert restored == config


def test_fingerprint_is_stable_for_equal_configs() -> None:
    assert ModuleConfig().fingerprint() == ModuleConfig().fingerprint()


def test_fingerprint_changes_when_config_changes() -> None:
    assert ModuleConfig().fingerprint() != ModuleConfig(gpu_memory_budget_gb=16.0).fingerprint()
