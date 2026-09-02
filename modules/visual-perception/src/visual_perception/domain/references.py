"""Repository-wide references this module depends on, plus its own model provenance.

``ObservationReference`` and ``SourceArtifactReference`` come from
`contextual_mapping_contracts` (#99/#100): stable identity, timing, frame,
and source-artifact references are a repository-wide concept, not owned by
visual-perception.

``ModelProvenance`` is deliberately kept local and distinct from
``contextual_mapping_contracts.Provenance``: the shared type links a derived
item back to the *source observations* it came from, while this type
records *which model/stage/configuration* produced one semantic claim or
relation (checkpoint, prompt version, config fingerprint). They answer
different questions and must not be confused.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextual_mapping_contracts import ObservationReference, SourceArtifactReference

__all__ = ["ModelProvenance", "ObservationReference", "SourceArtifactReference"]


@dataclass(frozen=True)
class ModelProvenance:
    """Which model/stage/configuration produced one derived claim, embedding, or relation."""

    stage: str
    producer: str
    config_fingerprint: str
    model_id: str | None = None
    checkpoint: str | None = None
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must not be empty.")
        if not self.producer:
            raise ValueError("producer must not be empty.")
        if not self.config_fingerprint:
            raise ValueError("config_fingerprint must not be empty.")
