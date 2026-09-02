"""Structured candidate relation contract.

Issue: #166.

Relations are 2D image-level candidates only. They are never treated as
verified 3D relations: that validation is owned downstream (scene-graph /
context-reasoning), once geometry has been associated across sensors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import ConfidenceScore, Evidence

_PREDICATE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


class RelationSource(StrEnum):
    """Whether a relation was derived from 2D geometry or inferred by a model.

    Issue: #167 keeps these distinguishable rather than merging provenance.
    """

    GEOMETRIC_2D = "geometric_2d"
    MODEL_INFERRED = "model_inferred"


@dataclass(frozen=True)
class CandidateRelation:
    """One unverified, image-level relation between two canonical regions."""

    relation_id: str
    subject_region_id: str
    predicate: str
    object_region_id: str
    confidence: ConfidenceScore
    source: RelationSource
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance

    def __post_init__(self) -> None:
        validate_identifier(self.relation_id, field="relation_id")
        validate_identifier(self.subject_region_id, field="subject_region_id")
        validate_identifier(self.object_region_id, field="object_region_id")
        if self.subject_region_id == self.object_region_id:
            raise ValueError(
                f"CandidateRelation({self.relation_id!r}) is a self-relation, which is not "
                "supported: subject_region_id and object_region_id must differ."
            )
        if not _PREDICATE_PATTERN.match(self.predicate):
            raise ValueError(
                f"predicate must be normalized snake_case matching {_PREDICATE_PATTERN.pattern!r}, "
                f"got {self.predicate!r}."
            )
        if not self.evidence:
            raise ValueError(
                f"CandidateRelation({self.relation_id!r}) must reference at least one Evidence."
            )


def validate_relation_references(
    relations: tuple[CandidateRelation, ...], known_region_ids: frozenset[str]
) -> None:
    """Reject relations that reference regions outside ``known_region_ids``.

    A dangling reference (e.g. to a region discarded during merge/refinement)
    fails explicitly rather than being silently dropped.
    """
    for relation in relations:
        for region_id, role in (
            (relation.subject_region_id, "subject_region_id"),
            (relation.object_region_id, "object_region_id"),
        ):
            if region_id not in known_region_ids:
                raise ValueError(
                    f"CandidateRelation({relation.relation_id!r}).{role} references unknown "
                    f"region {region_id!r}."
                )
