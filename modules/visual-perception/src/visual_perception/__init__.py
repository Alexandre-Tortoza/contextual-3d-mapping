"""Structured visual and semantic observations from RGB images.

Public boundary (issue #152): consumers should only need these imports.
Everything else under ``visual_perception`` is an implementation detail.
"""

from __future__ import annotations

from contextual_mapping_contracts import ObservationReference, SourceArtifactReference

from visual_perception.application.pipeline import PerceptionPorts, PipelineResult, run_canonical_pipeline
from visual_perception.config import ModuleConfig, QualityProfile
from visual_perception.domain.audit import AuditResult
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.visual_observation import VisualObservation

__all__ = [
    "AuditResult",
    "ImageObservation",
    "ImagePayload",
    "ModelProvenance",
    "ModuleConfig",
    "ObservationReference",
    "PerceptionPorts",
    "PipelineResult",
    "QualityProfile",
    "SourceArtifactReference",
    "VisualObservation",
    "run_canonical_pipeline",
]
