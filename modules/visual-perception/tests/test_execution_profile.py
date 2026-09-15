"""Testes do execution profile quality-first (#181)."""

from __future__ import annotations

import pytest

from visual_perception.application.execution_profile import (
    BackendCandidate,
    research_quality_config,
    select_research_quality_backend,
)
from visual_perception.config import QualityProfile


# Confirma a regra central da política quality-first: entre candidatos que cabem no
# orçamento, o de maior qualidade vence, mesmo sendo mais lento.
def test_selects_highest_quality_candidate_within_budget() -> None:
    candidates = (
        BackendCandidate("slow_high_quality", quality_score=0.95, peak_vram_gb=7.0, latency_s=5.0),
        BackendCandidate("fast_low_quality", quality_score=0.7, peak_vram_gb=1.0, latency_s=0.1),
    )
    selected = select_research_quality_backend(candidates, memory_budget_gb=8.0)
    assert selected.name == "slow_high_quality"


# Garante que latência nunca é usada como critério de exclusão, só memória — mesmo um
# candidato extremamente lento é selecionado se couber no orçamento e tiver qualidade.
def test_latency_never_excludes_a_candidate_that_fits_memory() -> None:
    candidates = (BackendCandidate("very_slow", quality_score=0.99, peak_vram_gb=7.9, latency_s=1000.0),)
    selected = select_research_quality_backend(candidates, memory_budget_gb=8.0)
    assert selected.name == "very_slow"


# Garante uma falha explícita quando nenhum candidato cabe no orçamento de memória, em
# vez de selecionar silenciosamente algo que estouraria a GPU.
def test_no_candidate_fitting_budget_raises() -> None:
    candidates = (BackendCandidate("too_big", quality_score=0.99, peak_vram_gb=16.0, latency_s=1.0),)
    with pytest.raises(ValueError):
        select_research_quality_backend(candidates, memory_budget_gb=8.0)


# Confirma que o perfil research-quality usa o fluxo híbrido por default, mas
# conserva o braço full-only para ablações com geometria de regiões fixa.
def test_research_quality_config_enables_multi_scale_by_default_and_allows_ablation() -> None:
    default = research_quality_config()
    full_only = research_quality_config(multi_scale_enabled=False)

    assert default.tiling.multi_scale_enabled
    assert default.tiling.tile_grid == "2x2"
    assert default.tiling.overlap_ratio == 0.2
    assert not full_only.tiling.multi_scale_enabled
    assert default.quality_profile is QualityProfile.RESEARCH_QUALITY


# Confirma que o perfil de referência real declara os quatro slots em vez
# de herdar defaults que mantinham contexto e cena desabilitados (#209).
def test_real_research_profile_enables_every_multi_context_slot_explicitly() -> None:
    """Habilita foreground, crop justo, crop contextual e cena no perfil real."""
    config = research_quality_config(real_backends=True)

    assert config.multi_context.foreground_enabled
    assert config.multi_context.tight_crop_enabled
    assert config.multi_context.contextual_crop_enabled
    assert config.multi_context.scene_conditioned_enabled
    assert config.feature_extraction.input_resolution == 448
    assert config.region_discovery.backend == "sam3"
    assert config.region_discovery.checkpoint == "facebook/sam3"
    assert config.region_discovery.prompt == "all visible objects"
