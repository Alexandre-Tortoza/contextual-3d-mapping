"""Visual observation quality auditor.

Issue: #168.

Most invariants are already enforced by the frozen domain dataclasses at
construction time. The auditor re-checks them anyway (defense in depth for
observations reconstructed from storage, see #172) and additionally detects
what construction *cannot* enforce: contradictory semantic claims, and
geometry that has drifted out of sync (a region's declared box no longer
matching its mask). Contradictions are reported as warnings, never silently
dropped.
"""

from __future__ import annotations

from visual_perception.domain.audit import AuditIssue, AuditResult, AuditSeverity
from visual_perception.domain.semantics import ClaimKind, contradicting_claims
from visual_perception.domain.visual_observation import VisualObservation


def audit_observation(observation: VisualObservation) -> AuditResult:
    """Deterministically audit one VisualObservation, without modifying it."""
    issues: list[AuditIssue] = []

    region_ids = [region.region_id for region in observation.regions]
    if len(region_ids) != len(set(region_ids)):
        issues.append(
            AuditIssue(AuditSeverity.ERROR, "duplicate_region_id", "Observation has duplicate region ids.")
        )
    known_ids = frozenset(region_ids)

    for region in observation.regions:
        if (region.mask.image_width, region.mask.image_height) != (
            observation.image_width,
            observation.image_height,
        ):
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "mask_resolution_mismatch",
                    f"Region {region.region_id!r} mask resolution does not match the observation.",
                    region_id=region.region_id,
                )
            )
        elif not region.mask.is_empty and region.box != region.mask.bounding_box():
            issues.append(
                AuditIssue(
                    AuditSeverity.ERROR,
                    "box_mask_mismatch",
                    f"Region {region.region_id!r} box does not match its mask's tight bounding box.",
                    region_id=region.region_id,
                )
            )
        for kind in ClaimKind:
            contradiction = contradicting_claims(region.claims, kind)
            if contradiction:
                issues.append(
                    AuditIssue(
                        AuditSeverity.WARNING,
                        "contradictory_claims",
                        f"Region {region.region_id!r} has contradictory {kind.value} claims: "
                        f"{sorted({claim.value for claim in contradiction})}.",
                        region_id=region.region_id,
                    )
                )

    for relation in observation.relations:
        for region_id, role in (
            (relation.subject_region_id, "subject_region_id"),
            (relation.object_region_id, "object_region_id"),
        ):
            if region_id not in known_ids:
                issues.append(
                    AuditIssue(
                        AuditSeverity.ERROR,
                        "dangling_relation_reference",
                        f"Relation {relation.relation_id!r}.{role} references unknown region {region_id!r}.",
                        relation_id=relation.relation_id,
                    )
                )

    for kind in ClaimKind:
        contradiction = contradicting_claims(observation.scene_context.claims, kind)
        if contradiction:
            issues.append(
                AuditIssue(
                    AuditSeverity.WARNING,
                    "contradictory_scene_claims",
                    f"Scene context has contradictory {kind.value} claims: "
                    f"{sorted({claim.value for claim in contradiction})}.",
                )
            )

    return AuditResult(observation_id=observation.observation_id, issues=tuple(issues))
