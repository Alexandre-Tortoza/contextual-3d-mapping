"""Testes do schema de configuração de módulo validada (#157)."""

from __future__ import annotations

import pytest

from visual_perception.config import ModuleConfig, QualityProfile, TilingConfig


# Confirma que a configuração default (sem nenhum campo explícito) é válida e usa o
# quality profile padrão esperado.
def test_minimal_config_validates() -> None:
    config = ModuleConfig()
    assert config.quality_profile is QualityProfile.RESEARCH_QUALITY


# Confirma que uma configuração com campos explícitos (orçamento de GPU, tiling
# multi-scale) também valida corretamente.
def test_complete_config_validates() -> None:
    tiling = TilingConfig(multi_scale_enabled=True, tile_grid="2x2")
    config = ModuleConfig(gpu_memory_budget_gb=12.0, tiling=tiling)
    assert config.tiling.multi_scale_enabled


# Protege o invariante de que o orçamento de memória de GPU deve ser positivo.
def test_rejects_non_positive_memory_budget() -> None:
    with pytest.raises(ValueError):
        ModuleConfig(gpu_memory_budget_gb=0)


# Protege a regra de negócio: o profile reduced_cost não pode ser combinado com tiling
# multi-scale (documentada em docs/architecture.md).
def test_rejects_incompatible_reduced_cost_multi_scale() -> None:
    tiling = TilingConfig(multi_scale_enabled=True)
    with pytest.raises(ValueError):
        ModuleConfig(quality_profile=QualityProfile.REDUCED_COST, tiling=tiling)


# Garante que um tile_grid mal formado é rejeitado cedo, na construção da config.
def test_rejects_invalid_tile_grid() -> None:
    with pytest.raises(ValueError):
        TilingConfig(tile_grid="not-a-grid")


# Confirma que a config sobrevive a um round-trip via to_dict/from_dict sem perda de
# informação — necessário para persistência e para o fingerprint do cache de estágio.
def test_config_round_trips_through_dict() -> None:
    config = ModuleConfig(tiling=TilingConfig(multi_scale_enabled=True, tile_grid="2x2"))
    restored = ModuleConfig.from_dict(config.to_dict())
    assert restored == config


# Garante que o fingerprint é determinístico: duas configs iguais produzem o mesmo
# fingerprint, condição necessária para o cache de estágio funcionar.
def test_fingerprint_is_stable_for_equal_configs() -> None:
    assert ModuleConfig().fingerprint() == ModuleConfig().fingerprint()


# Garante que configs diferentes produzem fingerprints diferentes, para que uma
# mudança de configuração invalide corretamente o cache.
def test_fingerprint_changes_when_config_changes() -> None:
    assert ModuleConfig().fingerprint() != ModuleConfig(gpu_memory_budget_gb=16.0).fingerprint()
