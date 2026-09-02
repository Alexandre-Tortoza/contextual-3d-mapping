"""Uncertainty-driven selective refinement.

Issue: #183.

Only regions flagged by structured uncertainty signals are reprocessed.
Previous evidence is never overwritten: refinement appends new claims
(the same append-only behavior region interpretation already has, see
#165) and each iteration is recorded in the returned history. The loop
always terminates: it stops as soon as no region still needs refinement,
and never runs more than ``RefinementConfig.max_iterations`` passes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from visual_perception.application.quality_audit import audit_observation
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner


@dataclass(frozen=True)
class RefinementConfig:
    low_confidence_threshold: float = 0.5
    small_region_area_px: int = 16
    max_iterations: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be in [0, 1].")
        if self.small_region_area_px < 0:
            raise ValueError("small_region_area_px must be non-negative.")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative.")


@dataclass(frozen=True)
class RefinementStep:
    """One refinement iteration's record: which regions were retried and why."""

    iteration: int
    target_region_ids: tuple[str, ...]
    failures: tuple[RegionInterpretationFailure, ...]


def select_refinement_targets(observation: VisualObservation, config: RefinementConfig) -> tuple[str, ...]:
    """Select region ids that need another interpretation pass."""
    audit = audit_observation(observation)
    targets = {
        issue.region_id
        for issue in audit.warnings
        if issue.code == "contradictory_claims" and issue.region_id is not None
    }
    for region in observation.regions:
        too_small = region.mask.area() < config.small_region_area_px
        low_confidence = any(
            claim.confidence.value < config.low_confidence_threshold for claim in region.claims
        )
        if not region.claims or too_small or low_confidence:
            targets.add(region.region_id)
    for relation in observation.relations:
        if relation.confidence.value < config.low_confidence_threshold:
            targets.add(relation.subject_region_id)
            targets.add(relation.object_region_id)
    return tuple(sorted(targets))


def refine_observation(
    observation: VisualObservation,
    image: ImagePayload,
    reasoner: MultimodalReasoner,
    multimodal_config: MultimodalReasoningConfig,
    refinement_config: RefinementConfig,
) -> tuple[VisualObservation, tuple[RefinementStep, ...]]:
    """Selectively reprocess uncertain regions until the loop converges or stops."""
    current = observation
    history: list[RefinementStep] = []

    for iteration in range(refinement_config.max_iterations):
        target_ids = select_refinement_targets(current, refinement_config)
        if not target_ids:
            break

        order = {region.region_id: index for index, region in enumerate(current.regions)}
        targets = tuple(region for region in current.regions if region.region_id in target_ids)
        others = tuple(region for region in current.regions if region.region_id not in target_ids)

        refined_targets, failures = interpret_regions(
            targets, image, current.scene_context, reasoner, multimodal_config
        )
        updated_regions = tuple(
            sorted(others + refined_targets, key=lambda region: order[region.region_id])
        )
        current = dataclasses.replace(current, regions=updated_regions)
        history.append(RefinementStep(iteration=iteration, target_region_ids=target_ids, failures=failures))

    return current, tuple(history)
