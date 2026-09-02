"""Canonical visual observation output contract tests (#154)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId, ObservationReference, Timestamp

from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.visual_observation import SceneContext, VisualObservation


def _mask(width: int = 4, height: int = 4) -> Mask:
    data = np.zeros((height, width), dtype=np.bool_)
    data[0, 0] = True
    return Mask(data, width, height)


def _region(region_id: str, width: int = 4, height: int = 4) -> ObservedRegion:
    mask = _mask(width, height)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-proposal",))


def _source(observation_id: str = "obs-1") -> ObservationReference:
    return ObservationReference(
        observation_id=observation_id,
        dataset_id="ds",
        sequence_id="seq",
        sensor_id="camera_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=0, clock_id="rosbag"),
        frame_id=FrameId("camera_1_optical_frame"),
    )


def test_minimal_observation_has_no_regions() -> None:
    observation = VisualObservation(_source(), 4, 4, SceneContext(), (), ())
    assert observation.regions == ()


def test_observation_id_comes_from_source() -> None:
    observation = VisualObservation(_source("obs-42"), 4, 4, SceneContext(), (), ())
    assert observation.observation_id == "obs-42"


def test_complete_observation_round_trips_region_lookup() -> None:
    region = _region("region-a")
    observation = VisualObservation(_source(), 4, 4, SceneContext(), (region,), ())
    assert observation.region_by_id("region-a") is region


def test_rejects_duplicate_region_ids() -> None:
    region = _region("region-a")
    with pytest.raises(ValueError):
        VisualObservation(_source(), 4, 4, SceneContext(), (region, region), ())


def test_rejects_region_mask_resolution_mismatch() -> None:
    region = _region("region-a", width=8, height=8)
    with pytest.raises(ValueError):
        VisualObservation(_source(), 4, 4, SceneContext(), (region,), ())


def test_unknown_region_lookup_raises_key_error() -> None:
    observation = VisualObservation(_source(), 4, 4, SceneContext(), (), ())
    with pytest.raises(KeyError):
        observation.region_by_id("missing")
