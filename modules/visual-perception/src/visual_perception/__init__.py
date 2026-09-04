"""Observações visuais e semânticas estruturadas a partir de imagens RGB.

Fronteira pública (issue #152): consumidores só devem precisar desses
imports. Tudo o mais dentro de ``visual_perception`` é detalhe de
implementação.
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
from visual_perception.infrastructure.adapters.factory import create_perception_ports

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
    "create_perception_ports",
    "run_canonical_pipeline",
]
