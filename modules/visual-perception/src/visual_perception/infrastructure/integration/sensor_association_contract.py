"""Verify output contract compatibility with sensor-association.

Issue: #178.

`sensor-association` (#137) needs, for every region: a stable identity,
image-space geometry (mask/box) under the documented coordinate convention,
the observation's timing and frame identity, and evidence/provenance. This
module extracts exactly that slice through public contracts only, so a
missing field fails integration tests instead of a silent reinterpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextual_mapping_contracts import FrameId, Timestamp

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.visual_observation import VisualObservation


@dataclass(frozen=True)
class RegionAssociationFixture:
    """The minimal slice of one region that sensor-association can consume."""

    observation_id: str
    region_id: str
    mask: Mask
    box: BoundingBox
    timestamp: Timestamp
    frame_id: FrameId
    coordinate_convention: str


def build_association_fixtures(observation: VisualObservation) -> tuple[RegionAssociationFixture, ...]:
    """Build one fixture per region, using only ``VisualObservation``'s public fields."""
    return tuple(
        RegionAssociationFixture(
            observation_id=observation.observation_id,
            region_id=region.region_id,
            mask=region.mask,
            box=region.box,
            timestamp=observation.source.timestamp,
            frame_id=observation.source.frame_id,
            coordinate_convention=observation.coordinate_convention,
        )
        for region in observation.regions
    )
