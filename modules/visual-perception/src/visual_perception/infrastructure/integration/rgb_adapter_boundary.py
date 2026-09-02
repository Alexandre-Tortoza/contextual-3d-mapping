"""Integrate canonical RGB observations from repository adapters.

Issue: #177.

`[adapters]` #103's ``CanonicalObservation`` boundary now exists
(`contextual_mapping_adapters`). This module adapts one RGB-kind
``CanonicalObservation`` (plus its already-resolved pixel array — decoding
the referenced artifact URI is dataset/transport-specific and stays outside
visual-perception, per #151) into the module's own canonical input.
"""

from __future__ import annotations

import numpy as np
from contextual_mapping_adapters import CanonicalObservation

from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload


def to_canonical_input(
    observation: CanonicalObservation, pixels: np.ndarray, encoding: str = "rgb8"
) -> tuple[ImageObservation, ImagePayload]:
    """Adapt one resolved RGB ``CanonicalObservation`` into visual-perception's input.

    Fails before inference when required metadata is missing, per #177's
    acceptance criteria, by relying on :class:`ImageObservation`'s own
    validation.
    """
    if observation.kind != "rgb":
        raise ValueError(f"Expected an rgb CanonicalObservation, got kind={observation.kind!r}.")
    height, width = pixels.shape[:2]
    image_observation = ImageObservation(
        width=width,
        height=height,
        encoding=encoding,
        image=observation.artifact,
        source=observation.reference,
    )
    payload = ImagePayload(pixels, width=width, height=height)
    return image_observation, payload
