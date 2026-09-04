"""Auditor de qualidade de visual observation.

Issue: #168.

A maioria dos invariantes já é imposta pelas dataclasses de domínio congeladas
(frozen) no momento da construção. O auditor os reconfere mesmo assim (defesa
em profundidade para observations reconstruídas a partir de armazenamento, ver
#172) e detecta adicionalmente o que a construção *não pode* impor: claims
semânticas contraditórias, e geometria que saiu de sincronia (o box declarado
de uma region não corresponde mais à sua mask). Contradições são reportadas
como warnings, nunca descartadas silenciosamente.
"""

from __future__ import annotations

from visual_perception.domain.audit import AuditIssue, AuditResult, AuditSeverity
from visual_perception.domain.semantics import ClaimKind, contradicting_claims
from visual_perception.domain.visual_observation import VisualObservation


# Audita uma VisualObservation de forma determinística, sem modificá-la.
# Existe como camada de defesa em profundidade: verifica invariantes de
# geometria (mask/box) e detecta claims semânticas contraditórias que as
# dataclasses de domínio congeladas não conseguem impor sozinhas na
# construção. Chamada pelo pipeline de pós-processamento (ver #168/#172).
def audit_observation(observation: VisualObservation) -> AuditResult:
    """Audita uma VisualObservation de forma determinística, sem modificá-la."""
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
