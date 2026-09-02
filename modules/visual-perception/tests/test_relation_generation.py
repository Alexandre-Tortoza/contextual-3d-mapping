"""Image-level relation generation tests (#167)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.application.relation_generation import generate_relations
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import RelationSource


def _region(region_id: str, box: tuple[int, int, int, int], size: int = 32) -> ObservedRegion:
    data = np.zeros((size, size), dtype=np.bool_)
    x0, y0, x1, y1 = box
    data[y0:y1, x0:x1] = True
    mask = Mask(data, size, size)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",))


def test_overlapping_regions_generate_overlap_relation() -> None:
    a = _region("a", (0, 0, 10, 10))
    b = _region("b", (5, 5, 15, 15))
    relations = generate_relations((a, b), RegionMergeConfig())
    assert any(r.predicate == "overlaps" for r in relations)


def test_containment_generates_contains_relation() -> None:
    container = _region("container", (0, 0, 20, 20))
    part = _region("part", (2, 2, 4, 4))
    relations = generate_relations((container, part), RegionMergeConfig())
    contains = [r for r in relations if r.predicate == "contains"]
    assert len(contains) == 1
    assert contains[0].subject_region_id == "container"
    assert contains[0].object_region_id == "part"


def test_adjacent_disjoint_regions_generate_near_relation() -> None:
    a = _region("a", (0, 0, 4, 4))
    b = _region("b", (5, 0, 9, 4))
    relations = generate_relations((a, b), RegionMergeConfig())
    assert any(r.predicate == "near" for r in relations)


def test_disjoint_far_regions_generate_no_relation() -> None:
    a = _region("a", (0, 0, 2, 2))
    b = _region("b", (28, 28, 30, 30))
    relations = generate_relations((a, b), RegionMergeConfig())
    assert relations == ()


def test_inferred_relations_are_included_and_tagged() -> None:
    a = _region("a", (0, 0, 2, 2))
    b = _region("b", (28, 28, 30, 30))
    inferred = (
        {"subject_region_id": "a", "predicate": "next_to", "object_region_id": "b", "confidence": 0.7},
    )
    relations = generate_relations((a, b), RegionMergeConfig(), inferred_relations=inferred)
    inferred_relation = next(r for r in relations if r.predicate == "next_to")
    assert inferred_relation.source is RelationSource.MODEL_INFERRED


def test_relations_referencing_unknown_region_are_rejected() -> None:
    a = _region("a", (0, 0, 2, 2))
    inferred = ({"subject_region_id": "a", "predicate": "next_to", "object_region_id": "ghost"},)
    with pytest.raises(ValueError):
        generate_relations((a,), RegionMergeConfig(), inferred_relations=inferred)
