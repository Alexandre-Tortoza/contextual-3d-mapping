"""Implementation-agnostic source observation and provenance references."""

from __future__ import annotations

from dataclasses import dataclass

from .spatial import FrameId
from .temporal import Timestamp


def _require(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty.")


@dataclass(frozen=True)
class SourceArtifactReference:
    """Logical reference to source data kept outside the repository."""

    uri: str
    media_type: str
    digest: str | None = None

    def __post_init__(self) -> None:
        _require(self.uri, "uri")
        _require(self.media_type, "media_type")
        if self.digest is not None:
            _require(self.digest, "digest")


@dataclass(frozen=True)
class ObservationReference:
    """Stable identity and interpretation metadata for a source observation."""

    observation_id: str
    dataset_id: str
    sequence_id: str
    sensor_id: str
    sequence_index: int
    timestamp: Timestamp
    frame_id: FrameId
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for field in ("observation_id", "dataset_id", "sequence_id", "sensor_id"):
            _require(getattr(self, field), field)
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative.")
        if self.calibration_id is not None:
            _require(self.calibration_id, "calibration_id")


@dataclass(frozen=True)
class Provenance:
    """A derived item's complete, non-overwriting set of source contributors."""

    producer: str
    observations: tuple[ObservationReference, ...]
    source_artifacts: tuple[SourceArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        _require(self.producer, "producer")
        if not self.observations:
            raise ValueError("provenance must contain at least one observation.")
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("provenance observation ids must be unique.")
