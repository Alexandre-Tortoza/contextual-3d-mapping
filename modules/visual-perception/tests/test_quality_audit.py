"""Visual observation quality auditor tests (#168)."""

from __future__ import annotations

import dataclasses

import numpy as np
from contextual_mapping_contracts import FrameId, ObservationReference, Timestamp

from visual_perception.application.quality_audit import audit_observation
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext, VisualObservation


def _mask(width: int = 8, height: int = 8) -> Mask:
    data = np.zeros((height, width), dtype=np.bool_)
    data[0:2, 0:2] = True
    return Mask(data, width, height)


def _claim(value: str) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.8, source="fake"),
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
    )


def _region(region_id: str, claims: tuple[SemanticClaim, ...] = ()) -> ObservedRegion:
    mask = _mask()
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",), claims=claims)


def _source() -> ObservationReference:
    return ObservationReference(
        observation_id="obs-1",
        dataset_id="ds",
        sequence_id="seq",
        sensor_id="cam",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=0, clock_id="rosbag"),
        frame_id=FrameId("cam_optical_frame"),
    )


def _observation(regions: tuple[ObservedRegion, ...]) -> VisualObservation:
    return VisualObservation(_source(), 8, 8, SceneContext(), regions, ())


def test_valid_observation_passes_without_modification() -> None:
    observation = _observation((_region("a", (_claim("box"),)),))
    result = audit_observation(observation)
    assert result.passed
    assert result.issues == ()


def test_contradictory_claims_are_flagged_but_observation_still_passes() -> None:
    observation = _observation((_region("a", (_claim("box"), _claim("crate"))),))
    result = audit_observation(observation)
    assert result.passed
    assert any(issue.code == "contradictory_claims" for issue in result.warnings)


def test_box_mask_mismatch_is_an_error() -> None:
    region = _region("a")
    tampered = dataclasses.replace(region, box=type(region.box)(0, 0, 100, 100))
    result = audit_observation(_observation((tampered,)))
    assert not result.passed
    assert any(issue.code == "box_mask_mismatch" for issue in result.errors)


def test_audit_is_deterministic() -> None:
    observation = _observation((_region("a", (_claim("box"), _claim("crate"))),))
    first = audit_observation(observation)
    second = audit_observation(observation)
    assert first == second
