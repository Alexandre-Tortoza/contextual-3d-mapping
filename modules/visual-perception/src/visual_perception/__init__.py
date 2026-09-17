"""Observações visuais e semânticas estruturadas a partir de imagens RGB.

Fronteira pública (issue #152): consumidores só devem precisar desses
imports. Tudo o mais dentro de ``visual_perception`` é detalhe de
implementação.
"""

from __future__ import annotations

from contextual_mapping_contracts import ObservationReference, SourceArtifactReference

from visual_perception.application.pipeline import PerceptionPorts, PipelineResult, run_canonical_pipeline
from visual_perception.application.semantic_grounding import build_spatial_footprints, ground_regions
from visual_perception.application.temporal_prior import prior_from
from visual_perception.config import (
    LanguageEmbeddingConfig,
    ModuleConfig,
    QualityProfile,
    SemanticGroundingConfig,
)
from visual_perception.debug_artifacts import FrameInputs, StageDebugImages, write_stage_debug_images
from visual_perception.domain.audit import AuditResult
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.grounding import (
    GroundingPrediction,
    GroundingRequest,
    GroundingStatus,
    SemanticGrounding,
    SpatialRegionFootprint,
)
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_reasoning import ScenePrior
from visual_perception.domain.semantics import RegionKind
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.infrastructure.adapters.factory import create_perception_ports
from visual_perception.infrastructure.debug_recorder import DebugRecorder
from visual_perception.infrastructure.embedding_archive import (
    UnresolvableEmbeddingRefError,
    resolve_embedding_vector,
    write_embedding_archive,
)
from visual_perception.infrastructure.serialization import deserialize_observation, serialize_observation
from visual_perception.infrastructure.serialization import mask_to_dict as encode_mask
from visual_perception.ports.semantic_grounding import SemanticGrounder

__all__ = [
    "AuditResult",
    "BoundingBox",
    "ImageAreaMasks",
    "ImageObservation",
    "ImagePayload",
    "LanguageEmbeddingConfig",
    "ModelProvenance",
    "ModuleConfig",
    "ObservationReference",
    "PerceptionPorts",
    "PipelineResult",
    "QualityProfile",
    "ScenePrior",
    "SourceArtifactReference",
    "VisualObservation",
    "create_perception_ports",
    "prior_from",
    "run_canonical_pipeline",
    "GroundingPrediction",
    "GroundingRequest",
    "GroundingStatus",
    "Mask",
    "SemanticGrounding",
    "SemanticGroundingConfig",
    "SemanticGrounder",
    "SpatialRegionFootprint",
    "RegionKind",
    "build_spatial_footprints",
    "deserialize_observation",
    "ground_regions",
    "serialize_observation",
    "encode_mask",
    "DebugRecorder",
    "UnresolvableEmbeddingRefError",
    "resolve_embedding_vector",
    "write_embedding_archive",
    "FrameInputs",
    "StageDebugImages",
    "write_stage_debug_images",
]
