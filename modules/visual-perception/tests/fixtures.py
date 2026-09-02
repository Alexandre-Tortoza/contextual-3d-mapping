"""Shared fixtures for canonical image and visual observations (#173)."""

from __future__ import annotations

import numpy as np
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from visual_perception.config import ModuleConfig
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload


def blank_payload(width: int = 32, height: int = 32) -> ImagePayload:
    """An all-white image: the fake region discoverer finds zero regions in it."""
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    return ImagePayload(pixels, width=width, height=height)


def payload_with_blobs(
    width: int = 32, height: int = 32, blobs: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...] = (
        (4, 4, 10, 10, (200, 30, 30)),
    ),
) -> ImagePayload:
    """A white image with one or more solid-color rectangular blobs.

    Each blob is ``(x_min, y_min, x_max, y_max, rgb)``.
    """
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for x_min, y_min, x_max, y_max, rgb in blobs:
        pixels[y_min:y_max, x_min:x_max] = rgb
    return ImagePayload(pixels, width=width, height=height)


def image_observation(
    observation_id: str = "frame-0001", width: int = 32, height: int = 32
) -> ImageObservation:
    image = SourceArtifactReference(uri="mem://frame-0001", media_type="image/png")
    source = ObservationReference(
        observation_id=observation_id,
        dataset_id="corridor02",
        sequence_id="seq-0",
        sensor_id="camera_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=1_000_000, clock_id="rosbag"),
        frame_id=FrameId("camera_1_optical_frame"),
    )
    return ImageObservation(width=width, height=height, encoding="rgb8", image=image, source=source)


def default_config() -> ModuleConfig:
    return ModuleConfig()
