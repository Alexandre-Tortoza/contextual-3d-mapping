"""Quality-first execution profile.

Issue: #181.

Defines the research-reference selection policy: quality is the primary
optimization target *subject to* the configured GPU memory budget. Latency
and throughput are measured but never used to reject a candidate that fits
the memory budget. Additional compute (multi-scale passes, larger models,
selective reprocessing) is only turned on in the reference config when
benchmark evidence (#174, #175) shows it actually improves quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.config import ModuleConfig, QualityProfile, TilingConfig


@dataclass(frozen=True)
class BackendCandidate:
    """One benchmarked backend option for a given stage (see #174)."""

    name: str
    quality_score: float
    peak_vram_gb: float
    latency_s: float

    def __post_init__(self) -> None:
        if self.peak_vram_gb < 0 or self.latency_s < 0:
            raise ValueError("peak_vram_gb and latency_s must be non-negative.")


def select_research_quality_backend(
    candidates: tuple[BackendCandidate, ...], memory_budget_gb: float
) -> BackendCandidate:
    """Pick the highest-quality candidate that fits the memory budget.

    Latency never excludes a candidate; only ``peak_vram_gb`` does.
    """
    affordable = tuple(c for c in candidates if c.peak_vram_gb <= memory_budget_gb)
    if not affordable:
        raise ValueError(
            f"No candidate fits the {memory_budget_gb} GB memory budget: "
            f"{[c.name for c in candidates]}."
        )
    return max(affordable, key=lambda c: (c.quality_score, -c.peak_vram_gb, c.name))


def additional_compute_is_justified(
    baseline_quality: float, enhanced_quality: float, minimum_improvement: float = 0.0
) -> bool:
    """Whether measured evidence justifies enabling extra compute (e.g. multi-scale, #159).

    ``minimum_improvement`` guards against enabling costly compute for noise-level gains.
    """
    return enhanced_quality > baseline_quality + minimum_improvement


def research_quality_config(
    *, multi_scale_justified: bool, gpu_memory_budget_gb: float = 8.0
) -> ModuleConfig:
    """Build the research-quality reference configuration.

    ``multi_scale_justified`` must come from a benchmark comparison (see
    :func:`additional_compute_is_justified`); this function does not decide
    it on its own.
    """
    return ModuleConfig(
        quality_profile=QualityProfile.RESEARCH_QUALITY,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        tiling=TilingConfig(multi_scale_enabled=multi_scale_justified, tile_grid="2x2"),
    )
