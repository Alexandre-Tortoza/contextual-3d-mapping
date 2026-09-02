"""Canonical image observation input contract tests (#153)."""

from __future__ import annotations

import pytest
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from fixtures import image_observation
from visual_perception.domain.image_observation import ImageObservation


def _reference() -> tuple[SourceArtifactReference, ObservationReference]:
    return (
        SourceArtifactReference(uri="mem://1", media_type="image/png"),
        ObservationReference(
            observation_id="obs-1",
            dataset_id="ds",
            sequence_id="seq",
            sensor_id="camera_1",
            sequence_index=0,
            timestamp=Timestamp(nanoseconds=0, clock_id="rosbag"),
            frame_id=FrameId("camera_1_optical_frame"),
        ),
    )


def test_minimal_valid_observation() -> None:
    observation = image_observation()
    assert observation.observation_id == "frame-0001"


def test_rejects_non_positive_dimensions() -> None:
    image, source = _reference()
    with pytest.raises(ValueError):
        ImageObservation(0, 0, "rgb8", image, source)


def test_rejects_unsupported_encoding() -> None:
    image, source = _reference()
    with pytest.raises(ValueError):
        ImageObservation(10, 10, "yuv420", image, source)


def test_preserves_source_observation_reference() -> None:
    observation = image_observation()
    assert observation.source.sensor_id == "camera_1"
