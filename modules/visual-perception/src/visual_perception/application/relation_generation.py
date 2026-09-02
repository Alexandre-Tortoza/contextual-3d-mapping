"""Image-level relation generation.

Issue: #167.

Generates 2D-geometric relations directly from merged region geometry, and
optionally accepts model-inferred relations that a multimodal stage has
already resolved to canonical region ids (subject/object resolution from raw
model text is not this function's responsibility). Both kinds keep their
provenance separately, and both are still explicitly *candidate*, unverified
relations (#166): no 3D validation happens here.
"""

from __future__ import annotations

from typing import Any

from visual_perception.application.support import fingerprint_of
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import (
    CandidateRelation,
    RelationSource,
    validate_relation_references,
)
from visual_perception.domain.semantics import ConfidenceScore, Evidence

_CONTAINMENT_THRESHOLD = 0.9
_ADJACENCY_MARGIN_PX = 5.0


def generate_relations(
    regions: tuple[ObservedRegion, ...],
    config: RegionMergeConfig,
    inferred_relations: tuple[dict[str, Any], ...] = (),
) -> tuple[CandidateRelation, ...]:
    """Generate candidate relations, referencing only the given canonical regions."""
    geometric_provenance = ModelProvenance(
        stage="relation_generation", producer="geometric_2d", config_fingerprint=fingerprint_of(config)
    )
    relations: list[CandidateRelation] = []
    for i, subject in enumerate(regions):
        for target in regions[i + 1 :]:
            relations.extend(
                _geometric_relations(subject, target, geometric_provenance)
            )

    for index, raw in enumerate(inferred_relations):
        relations.append(_model_inferred_relation(raw, index))

    known_ids = frozenset(region.region_id for region in regions)
    result = tuple(relations)
    validate_relation_references(result, known_ids)
    return result


def _geometric_relations(
    subject: ObservedRegion, target: ObservedRegion, provenance: ModelProvenance
) -> list[CandidateRelation]:
    relations: list[CandidateRelation] = []
    confidence = ConfidenceScore(1.0, source="geometric_2d")
    iou = subject.mask.iou(target.mask)
    evidence = (Evidence(description=f"mask overlap between {subject.region_id} and {target.region_id}"),)

    if iou > 0.0:
        relations.append(
            CandidateRelation(
                relation_id=f"rel-overlaps-{subject.region_id}-{target.region_id}",
                subject_region_id=subject.region_id,
                predicate="overlaps",
                object_region_id=target.region_id,
                confidence=confidence,
                source=RelationSource.GEOMETRIC_2D,
                evidence=evidence,
                provenance=provenance,
            )
        )

    forward = subject.mask.containment_ratio(target.mask)
    backward = target.mask.containment_ratio(subject.mask)
    if forward >= _CONTAINMENT_THRESHOLD and forward > backward:
        relations.append(_contains(subject, target, confidence, provenance))
    elif backward >= _CONTAINMENT_THRESHOLD and backward > forward:
        relations.append(_contains(target, subject, confidence, provenance))

    if iou == 0.0 and _boxes_are_near(subject, target):
        relations.append(
            CandidateRelation(
                relation_id=f"rel-near-{subject.region_id}-{target.region_id}",
                subject_region_id=subject.region_id,
                predicate="near",
                object_region_id=target.region_id,
                confidence=confidence,
                source=RelationSource.GEOMETRIC_2D,
                evidence=evidence,
                provenance=provenance,
            )
        )
    return relations


def _contains(
    container: ObservedRegion, part: ObservedRegion, confidence: ConfidenceScore, provenance: ModelProvenance
) -> CandidateRelation:
    return CandidateRelation(
        relation_id=f"rel-contains-{container.region_id}-{part.region_id}",
        subject_region_id=container.region_id,
        predicate="contains",
        object_region_id=part.region_id,
        confidence=confidence,
        source=RelationSource.GEOMETRIC_2D,
        evidence=(Evidence(description=f"{container.region_id} contains most of {part.region_id}"),),
        provenance=provenance,
    )


def _boxes_are_near(subject: ObservedRegion, target: ObservedRegion) -> bool:
    a, b = subject.box, target.box
    gap_x = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    gap_y = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    return gap_x <= _ADJACENCY_MARGIN_PX and gap_y <= _ADJACENCY_MARGIN_PX


def _model_inferred_relation(raw: dict[str, Any], index: int) -> CandidateRelation:
    try:
        subject_id = str(raw["subject_region_id"])
        predicate = str(raw["predicate"])
        object_id = str(raw["object_region_id"])
    except KeyError as error:
        raise ValueError(f"Malformed inferred relation at index {index}: missing {error}.") from error
    validate_identifier(subject_id, field="subject_region_id")
    validate_identifier(object_id, field="object_region_id")
    provenance = ModelProvenance(
        stage="relation_generation",
        producer=str(raw.get("producer", "multimodal_reasoner")),
        config_fingerprint=str(raw.get("config_fingerprint", "unknown")),
    )
    return CandidateRelation(
        relation_id=f"rel-inferred-{index}-{subject_id}-{object_id}",
        subject_region_id=subject_id,
        predicate=predicate,
        object_region_id=object_id,
        confidence=ConfidenceScore(float(raw.get("confidence", 1.0)), source=provenance.producer),
        source=RelationSource.MODEL_INFERRED,
        evidence=(Evidence(description="model-inferred relation from contextual analysis"),),
        provenance=provenance,
    )
