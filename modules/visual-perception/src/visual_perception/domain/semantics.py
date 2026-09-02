"""Semantic claims, confidence, evidence, and provenance contracts.

Issue: #156.

Semantic interpretation is represented as a set of auditable *claims* rather
than one label and one score. Multiple, even contradictory, claims of the
same kind may coexist (e.g. two label hypotheses for an ambiguous region);
nothing is silently overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.references import ModelProvenance, SourceArtifactReference


class ClaimKind(StrEnum):
    """The category of a semantic claim.

    Geometric confidence (mask/box quality) is a separate concept, tracked on
    ``ObservedRegion.geometric_confidence`` (see ``domain/regions.py``), never
    mixed with a claim's semantic confidence.
    """

    LABEL = "label"
    ATTRIBUTE = "attribute"
    CONDITION = "condition"
    MATERIAL = "material"
    HAZARD = "hazard"
    SCENE_TYPE = "scene_type"
    SCENE_DESCRIPTION = "scene_description"


@dataclass(frozen=True)
class ConfidenceScore:
    """A confidence value in ``[0, 1]`` attributed to one source."""

    value: float
    source: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"ConfidenceScore.value must be in [0, 1], got {self.value}.")
        if not self.source:
            raise ValueError("ConfidenceScore.source must not be empty.")


@dataclass(frozen=True)
class Evidence:
    """A pointer to the raw evidence that supports a claim (crop, prompt/response, etc.).

    Reuses the shared ``SourceArtifactReference`` shape (uri/media_type/digest)
    for this module's own derived evidence artifacts too, rather than
    introducing a near-identical local type.
    """

    description: str
    artifact: SourceArtifactReference | None = None

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("Evidence.description must not be empty.")


@dataclass(frozen=True)
class SemanticClaim:
    """One auditable unit of semantic interpretation."""

    kind: ClaimKind
    value: str
    confidence: ConfidenceScore
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("SemanticClaim.value must not be empty.")
        if not self.evidence:
            raise ValueError(
                f"SemanticClaim({self.kind.value!r}) must reference at least one Evidence."
            )


def contradicting_claims(claims: tuple[SemanticClaim, ...], kind: ClaimKind) -> tuple[SemanticClaim, ...]:
    """Return the claims of ``kind`` that disagree with each other (more than one distinct value).

    Used by the quality auditor (#168) to flag, not silently resolve,
    contradictions.
    """
    matching = tuple(claim for claim in claims if claim.kind is kind)
    distinct_values = {claim.value for claim in matching}
    return matching if len(distinct_values) > 1 else ()
