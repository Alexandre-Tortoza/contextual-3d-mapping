"""Canonical visual observation output contract.

Issue: #154.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visual_perception.domain.references import ObservationReference
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import CandidateRelation, validate_relation_references
from visual_perception.domain.semantics import SemanticClaim

#: Binding image-coordinate convention for every VisualObservation.
#: See ``domain/geometry.py`` for the full definition.
COORDINATE_CONVENTION = "top-left-origin,half-open-xyxy"


@dataclass(frozen=True)
class SceneContext:
    """Scene-level semantic claims that are not the source of region geometry.

    Issue: #164.
    """

    claims: tuple[SemanticClaim, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VisualObservation:
    """The complete, canonical output of visual perception for one image.

    Identity and timing live on ``source`` (the shared ``ObservationReference``,
    #100), not duplicated as separate fields here.
    """

    source: ObservationReference
    image_width: int
    image_height: int
    scene_context: SceneContext
    regions: tuple[ObservedRegion, ...]
    relations: tuple[CandidateRelation, ...]
    schema_version: int = 1
    coordinate_convention: str = COORDINATE_CONVENTION

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive.")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("VisualObservation.regions must have unique region_id values.")
        for region in self.regions:
            if (region.mask.image_width, region.mask.image_height) != (
                self.image_width,
                self.image_height,
            ):
                raise ValueError(
                    f"ObservedRegion({region.region_id!r}) mask resolution does not match "
                    "the observation's image resolution."
                )
        validate_relation_references(self.relations, frozenset(region_ids))

    @property
    def observation_id(self) -> str:
        return str(self.source.observation_id)

    def region_by_id(self, region_id: str) -> ObservedRegion:
        for region in self.regions:
            if region.region_id == region_id:
                return region
        raise KeyError(f"No region with region_id={region_id!r}.")
