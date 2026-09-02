"""Validated module configuration schema.

Issue: #157.

This module owns the parameters that control its own algorithms (backend
identifiers, checkpoints, resolutions, thresholds, quality profile). Dataset
source, synchronization, and application composition are owned by the
consuming application, not by this schema (see ``docs/engineering-principles.md``
"Configuration ownership").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class QualityProfile(StrEnum):
    """Which execution profile the module should optimize for.

    Issue: #181 defines the ``RESEARCH_QUALITY`` profile in detail.
    """

    RESEARCH_QUALITY = "research_quality"
    REDUCED_COST = "reduced_cost"


@dataclass(frozen=True)
class RegionDiscoveryConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    score_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("region_discovery.score_threshold must be in [0, 1].")


@dataclass(frozen=True)
class TilingConfig:
    multi_scale_enabled: bool = False
    tile_grid: str = "1x1"
    overlap_ratio: float = 0.2

    def __post_init__(self) -> None:
        if "x" not in self.tile_grid:
            raise ValueError("tiling.tile_grid must look like '<rows>x<cols>', e.g. '2x2'.")
        rows, _, cols = self.tile_grid.partition("x")
        if not (rows.isdigit() and cols.isdigit() and int(rows) > 0 and int(cols) > 0):
            raise ValueError(f"tiling.tile_grid is not a valid grid: {self.tile_grid!r}.")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("tiling.overlap_ratio must be in [0, 1).")


@dataclass(frozen=True)
class RegionMergeConfig:
    iou_merge_threshold: float = 0.85
    containment_merge_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_merge_threshold <= 1.0:
            raise ValueError("merge.iou_merge_threshold must be in [0, 1].")
        if not 0.0 <= self.containment_merge_threshold <= 1.0:
            raise ValueError("merge.containment_merge_threshold must be in [0, 1].")


@dataclass(frozen=True)
class FeatureExtractionConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    feature_resolution: int = 16

    def __post_init__(self) -> None:
        if self.feature_resolution <= 0:
            raise ValueError("feature_extraction.feature_resolution must be positive.")


@dataclass(frozen=True)
class LanguageEmbeddingConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    dimension: int = 512

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("language_embedding.dimension must be positive.")


@dataclass(frozen=True)
class MultimodalReasoningConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.prompt_version:
            raise ValueError("multimodal_reasoning.prompt_version must not be empty.")


@dataclass(frozen=True)
class ModuleConfig:
    """The complete, reproducible configuration owned by visual-perception."""

    quality_profile: QualityProfile = QualityProfile.RESEARCH_QUALITY
    gpu_memory_budget_gb: float = 8.0
    region_discovery: RegionDiscoveryConfig = field(default_factory=RegionDiscoveryConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    merge: RegionMergeConfig = field(default_factory=RegionMergeConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    language_embedding: LanguageEmbeddingConfig = field(default_factory=LanguageEmbeddingConfig)
    multimodal_reasoning: MultimodalReasoningConfig = field(
        default_factory=MultimodalReasoningConfig
    )

    def __post_init__(self) -> None:
        if self.gpu_memory_budget_gb <= 0:
            raise ValueError("gpu_memory_budget_gb must be positive.")
        if (
            self.quality_profile is QualityProfile.REDUCED_COST
            and self.tiling.multi_scale_enabled
        ):
            raise ValueError(
                "Incompatible configuration: 'reduced_cost' quality profile does not support "
                "multi-scale tiling (see issue #181)."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_profile"] = self.quality_profile.value
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ModuleConfig:
        payload = dict(payload)
        payload["quality_profile"] = QualityProfile(
            payload.get("quality_profile", QualityProfile.RESEARCH_QUALITY.value)
        )
        for key, config_type in (
            ("region_discovery", RegionDiscoveryConfig),
            ("tiling", TilingConfig),
            ("merge", RegionMergeConfig),
            ("feature_extraction", FeatureExtractionConfig),
            ("language_embedding", LanguageEmbeddingConfig),
            ("multimodal_reasoning", MultimodalReasoningConfig),
        ):
            if key in payload and isinstance(payload[key], dict):
                payload[key] = config_type(**payload[key])
        return ModuleConfig(**payload)

    def fingerprint(self) -> str:
        """A stable hash of the full configuration, used for caching (#170)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
