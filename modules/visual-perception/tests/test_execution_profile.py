"""Testes do execution profile quality-first (#181)."""

from __future__ import annotations

import pytest

from visual_perception.application.execution_profile import (
    BackendCandidate,
    additional_compute_is_justified,
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


# Verifica o threshold de melhoria mínima que justifica compute adicional (ex: habilitar
# tiling multi-scale) — só compensa se o ganho de qualidade for mensurável.
def test_additional_compute_requires_measured_improvement() -> None:
    assert additional_compute_is_justified(0.70, 0.85)
    assert not additional_compute_is_justified(0.70, 0.71, minimum_improvement=0.05)


# Confirma que a config de research-quality só habilita tiling multi-scale quando o
# chamador já determinou (via benchmark) que o ganho de qualidade justifica o custo.
def test_research_quality_config_only_enables_multi_scale_when_justified() -> None:
    justified = research_quality_config(multi_scale_justified=True)
    unjustified = research_quality_config(multi_scale_justified=False)
    assert justified.tiling.multi_scale_enabled
    assert not unjustified.tiling.multi_scale_enabled
    assert justified.quality_profile is QualityProfile.RESEARCH_QUALITY
