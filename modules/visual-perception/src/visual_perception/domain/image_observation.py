"""Canonical image observation input contract.

Issue: #153.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.domain.references import ObservationReference, SourceArtifactReference

_SUPPORTED_ENCODINGS = frozenset({"rgb8", "bgr8", "rgba8"})


@dataclass(frozen=True)
class ImageObservation:
    """One RGB(-like) frame consumed by visual perception.

    Implementation-agnostic: it carries a :class:`SourceArtifactReference` to
    the pixel payload rather than a concrete tensor/array type, so this
    contract is serializable and does not depend on any imaging library.
    Identity, timing, and frame provenance live on ``source`` (the shared
    :class:`ObservationReference`), not duplicated here.
    """

    width: int
    height: int
    encoding: str
    image: SourceArtifactReference
    source: ObservationReference

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if self.encoding not in _SUPPORTED_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {sorted(_SUPPORTED_ENCODINGS)}, got {self.encoding!r}."
            )

    @property
    def observation_id(self) -> str:
        return str(self.source.observation_id)
