"""Structured candidate relation contract tests (#166)."""

from __future__ import annotations

import pytest

from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.relations import (
    CandidateRelation,
    RelationSource,
    validate_relation_references,
)
from visual_perception.domain.semantics import ConfidenceScore, Evidence


def _relation(subject: str, predicate: str, obj: str) -> CandidateRelation:
    return CandidateRelation(
        relation_id=f"rel-{subject}-{obj}",
        subject_region_id=subject,
        predicate=predicate,
        object_region_id=obj,
        confidence=ConfidenceScore(0.8, source="fake"),
        source=RelationSource.GEOMETRIC_2D,
        evidence=(Evidence("overlap"),),
        provenance=ModelProvenance(stage="test", producer="fake", config_fingerprint="abc"),
    )


def test_valid_relation() -> None:
    relation = _relation("region-a", "overlaps", "region-b")
    assert relation.predicate == "overlaps"


def test_rejects_self_relation() -> None:
    with pytest.raises(ValueError):
        _relation("region-a", "overlaps", "region-a")


def test_rejects_non_normalized_predicate() -> None:
    with pytest.raises(ValueError):
        _relation("region-a", "Overlaps With", "region-b")


def test_rejects_relation_without_evidence() -> None:
    with pytest.raises(ValueError):
        CandidateRelation(
            "rel-1",
            "region-a",
            "overlaps",
            "region-b",
            ConfidenceScore(0.8, source="fake"),
            RelationSource.GEOMETRIC_2D,
            (),
            ModelProvenance(stage="test", producer="fake", config_fingerprint="abc"),
        )


def test_dangling_reference_is_rejected() -> None:
    relations = (_relation("region-a", "overlaps", "region-missing"),)
    with pytest.raises(ValueError):
        validate_relation_references(relations, frozenset({"region-a"}))


def test_valid_reference_passes() -> None:
    relations = (_relation("region-a", "overlaps", "region-b"),)
    validate_relation_references(relations, frozenset({"region-a", "region-b"}))
