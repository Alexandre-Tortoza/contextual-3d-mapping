"""Canonical visual observation serialization boundary.

Issue: #172.

Masks are embedded as compact run-length-encoded booleans (small, exact, and
self-contained) rather than through an external artifact store, since a
region's own geometry is required to interpret the observation at all.
Visual/language embeddings and raw model evidence are referenced by id only
(``visual_embedding_ref`` / ``language_embedding_ref`` / ``Evidence.artifact``),
consistent with #154's "reference large payloads" rule.
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import CandidateRelation, RelationSource
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext, VisualObservation

#: The only schema version this module can read. Bump together with a
#: migration or an explicit incompatibility error, never silently.
SUPPORTED_SCHEMA_VERSION = 1


class UnsupportedSchemaVersionError(ValueError):
    """Raised when deserializing a payload from an incompatible schema version."""


def serialize_observation(observation: VisualObservation) -> dict[str, Any]:
    """Serialize a canonical VisualObservation to a plain JSON-able dict."""
    if observation.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Cannot serialize schema_version={observation.schema_version}; "
            f"this module writes schema_version={SUPPORTED_SCHEMA_VERSION}."
        )
    return {
        "schema_version": observation.schema_version,
        "coordinate_convention": observation.coordinate_convention,
        "source": _source_to_dict(observation.source),
        "image_width": observation.image_width,
        "image_height": observation.image_height,
        "scene_context": {"claims": [_claim_to_dict(claim) for claim in observation.scene_context.claims]},
        "regions": [_region_to_dict(region) for region in observation.regions],
        "relations": [_relation_to_dict(relation) for relation in observation.relations],
    }


def deserialize_observation(payload: dict[str, Any]) -> VisualObservation:
    """Reconstruct a canonical VisualObservation, round-tripping without information loss."""
    schema_version = payload.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Cannot deserialize schema_version={schema_version!r}; "
            f"this module reads schema_version={SUPPORTED_SCHEMA_VERSION}."
        )
    return VisualObservation(
        source=_source_from_dict(payload["source"]),
        image_width=payload["image_width"],
        image_height=payload["image_height"],
        scene_context=SceneContext(
            claims=tuple(_claim_from_dict(claim) for claim in payload["scene_context"]["claims"])
        ),
        regions=tuple(_region_from_dict(region) for region in payload["regions"]),
        relations=tuple(_relation_from_dict(relation) for relation in payload["relations"]),
        schema_version=schema_version,
        coordinate_convention=payload["coordinate_convention"],
    )


def _source_to_dict(source: ObservationReference) -> dict[str, Any]:
    return {
        "observation_id": source.observation_id,
        "dataset_id": source.dataset_id,
        "sequence_id": source.sequence_id,
        "sensor_id": source.sensor_id,
        "sequence_index": source.sequence_index,
        "timestamp": {"nanoseconds": source.timestamp.nanoseconds, "clock_id": source.timestamp.clock_id},
        "frame_id": source.frame_id.value,
        "calibration_id": source.calibration_id,
    }


def _source_from_dict(payload: dict[str, Any]) -> ObservationReference:
    payload = dict(payload)
    payload["timestamp"] = Timestamp(**payload["timestamp"])
    payload["frame_id"] = FrameId(payload["frame_id"])
    return ObservationReference(**payload)


def _mask_to_dict(mask: Mask) -> dict[str, Any]:
    flat = mask.data.reshape(-1)
    runs = [len(list(group)) for _, group in groupby(flat.tolist())]
    if flat.size > 0 and bool(flat[0]):
        runs = [0, *runs]  # RLE always starts with a False run count, even if zero.
    return {"width": mask.image_width, "height": mask.image_height, "rle": runs}


def _mask_from_dict(payload: dict[str, Any]) -> Mask:
    width, height, runs = payload["width"], payload["height"], payload["rle"]
    flat = np.zeros(width * height, dtype=np.bool_)
    cursor = 0
    value = False
    for run_length in runs:
        if value:
            flat[cursor : cursor + run_length] = True
        cursor += run_length
        value = not value
    return Mask(flat.reshape(height, width), width, height)


def _box_to_dict(box: BoundingBox) -> dict[str, float]:
    return {"x_min": box.x_min, "y_min": box.y_min, "x_max": box.x_max, "y_max": box.y_max}


def _box_from_dict(payload: dict[str, float]) -> BoundingBox:
    return BoundingBox(**payload)


def _provenance_to_dict(provenance: ModelProvenance) -> dict[str, Any]:
    return {
        "stage": provenance.stage,
        "producer": provenance.producer,
        "config_fingerprint": provenance.config_fingerprint,
        "model_id": provenance.model_id,
        "checkpoint": provenance.checkpoint,
        "prompt_version": provenance.prompt_version,
    }


def _provenance_from_dict(payload: dict[str, Any]) -> ModelProvenance:
    return ModelProvenance(**payload)


def _evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    artifact = None if evidence.artifact is None else vars(evidence.artifact)
    return {"description": evidence.description, "artifact": artifact}


def _evidence_from_dict(payload: dict[str, Any]) -> Evidence:
    artifact = None if payload.get("artifact") is None else SourceArtifactReference(**payload["artifact"])
    return Evidence(description=payload["description"], artifact=artifact)


def _claim_to_dict(claim: SemanticClaim) -> dict[str, Any]:
    return {
        "kind": claim.kind.value,
        "value": claim.value,
        "confidence": {"value": claim.confidence.value, "source": claim.confidence.source},
        "evidence": [_evidence_to_dict(item) for item in claim.evidence],
        "provenance": _provenance_to_dict(claim.provenance),
    }


def _claim_from_dict(payload: dict[str, Any]) -> SemanticClaim:
    return SemanticClaim(
        kind=ClaimKind(payload["kind"]),
        value=payload["value"],
        confidence=ConfidenceScore(**payload["confidence"]),
        evidence=tuple(_evidence_from_dict(item) for item in payload["evidence"]),
        provenance=_provenance_from_dict(payload["provenance"]),
    )


def _region_to_dict(region: ObservedRegion) -> dict[str, Any]:
    return {
        "region_id": region.region_id,
        "mask": _mask_to_dict(region.mask),
        "box": _box_to_dict(region.box),
        "geometric_confidence": region.geometric_confidence,
        "contributing_proposal_ids": list(region.contributing_proposal_ids),
        "claims": [_claim_to_dict(claim) for claim in region.claims],
        "visual_embedding_ref": region.visual_embedding_ref,
        "language_embedding_ref": region.language_embedding_ref,
    }


def _region_from_dict(payload: dict[str, Any]) -> ObservedRegion:
    validate_identifier(payload["region_id"], field="region_id")
    return ObservedRegion(
        region_id=payload["region_id"],
        mask=_mask_from_dict(payload["mask"]),
        box=_box_from_dict(payload["box"]),
        geometric_confidence=payload["geometric_confidence"],
        contributing_proposal_ids=tuple(payload["contributing_proposal_ids"]),
        claims=tuple(_claim_from_dict(claim) for claim in payload["claims"]),
        visual_embedding_ref=payload["visual_embedding_ref"],
        language_embedding_ref=payload["language_embedding_ref"],
    )


def _relation_to_dict(relation: CandidateRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "subject_region_id": relation.subject_region_id,
        "predicate": relation.predicate,
        "object_region_id": relation.object_region_id,
        "confidence": {"value": relation.confidence.value, "source": relation.confidence.source},
        "source": relation.source.value,
        "evidence": [_evidence_to_dict(item) for item in relation.evidence],
        "provenance": _provenance_to_dict(relation.provenance),
    }


def _relation_from_dict(payload: dict[str, Any]) -> CandidateRelation:
    return CandidateRelation(
        relation_id=payload["relation_id"],
        subject_region_id=payload["subject_region_id"],
        predicate=payload["predicate"],
        object_region_id=payload["object_region_id"],
        confidence=ConfidenceScore(**payload["confidence"]),
        source=RelationSource(payload["source"]),
        evidence=tuple(_evidence_from_dict(item) for item in payload["evidence"]),
        provenance=_provenance_from_dict(payload["provenance"]),
    )
