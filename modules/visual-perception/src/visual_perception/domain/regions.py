"""Region proposal and canonical observed-region contracts.

Issues: #158 (region discovery boundary output), #154/#160 (canonical region).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.semantics import SemanticClaim


@dataclass(frozen=True)
class TileProvenance:
    """Which scale/tile of a (possibly tiled, possibly multi-scale) pass produced a proposal.

    Issue: #159.
    """

    scale_id: str
    tile_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.scale_id, field="scale_id")
        validate_identifier(self.tile_id, field="tile_id")


@dataclass(frozen=True)
class LocalRegionProposal:
    """One candidate region in tile-local pixel coordinates, as produced
    directly by a :class:`~visual_perception.ports.region_discovery.RegionDiscoverer`
    before remapping to the original image (see #159).
    """

    local_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    source: str

    def __post_init__(self) -> None:
        validate_identifier(self.local_id, field="local_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.source:
            raise ValueError("source must not be empty.")
        if self.mask.is_empty:
            raise ValueError(f"LocalRegionProposal({self.local_id!r}) has an empty mask.")


@dataclass(frozen=True)
class RegionProposal:
    """One candidate region as produced by a region discovery backend, already
    remapped to original image coordinates (see #159's tiling boundary).
    """

    proposal_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    source: str
    tile: TileProvenance

    def __post_init__(self) -> None:
        validate_identifier(self.proposal_id, field="proposal_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.source:
            raise ValueError("source must not be empty.")
        if self.mask.is_empty:
            raise ValueError(f"RegionProposal({self.proposal_id!r}) has an empty mask.")


@dataclass(frozen=True)
class ObservedRegion:
    """One canonical, final region inside a :class:`VisualObservation`.

    ``geometric_confidence`` is distinct from any semantic claim's confidence
    (see #156): it describes only how trustworthy the mask/box geometry is.
    """

    region_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    contributing_proposal_ids: tuple[str, ...]
    claims: tuple[SemanticClaim, ...] = field(default_factory=tuple)
    visual_embedding_ref: str | None = None
    language_embedding_ref: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.region_id, field="region_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.contributing_proposal_ids:
            raise ValueError("ObservedRegion must preserve at least one contributing proposal id.")
