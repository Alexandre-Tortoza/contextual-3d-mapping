"""Auditor de qualidade de visual observation.

Issues: #168 (auditoria), #194 (falha isolada de slot), #196 (falha de
calibração distinta de discordância semântica).

A maioria dos invariantes já é imposta pelas dataclasses de domínio congeladas
(frozen) no momento da construção. O auditor os reconfere mesmo assim (defesa
em profundidade para observations reconstruídas a partir de armazenamento, ver
#172) e detecta adicionalmente o que a construção *não pode* impor: claims
semânticas contraditórias, e geometria que saiu de sincronia (o box declarado
de uma region não corresponde mais à sua mask). Contradições são reportadas
como warnings, nunca descartadas silenciosamente.

Três situações têm códigos próprios e distintos, porque exigem respostas
diferentes de quem lê o relatório:

- ``contradictory_claims``: o modelo discordou de si mesmo — é sinal
  semântico, e a observação continua utilizável;
- ``calibration_failed``: a *regra de calibração* quebrou, então nenhum
  score daquela claim foi verificado — não é discordância semântica;
- ``evidence_slot_failed``: um slot de evidência opcional não pôde ser
  produzido, sem invalidar os slots que deram certo na mesma region.
"""

from __future__ import annotations

from visual_perception.domain.audit import AuditIssue, AuditResult, AuditSeverity
from visual_perception.domain.claim_exclusivity import contradicting_claims
from visual_perception.domain.region_evidence import EvidenceState
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantic_support import SupportState
from visual_perception.domain.semantics import (
    ClaimKind,
    RegionKind,
    SemanticClaim,
    normalize_claim_value,
)
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
        issues.extend(_calibration_issues(region.claims, region_id=region.region_id))
        issues.extend(_region_kind_issues(region))
        issues.extend(_evidence_issues(region))

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
    issues.extend(_calibration_issues(observation.scene_context.claims, region_id=None))

    return AuditResult(observation_id=observation.observation_id, issues=tuple(issues))


# Reporta claims cuja *regra de calibração* falhou, com um código distinto
# do de contradição semântica. Existe porque as duas situações pedem ações
# diferentes: uma indica um modelo indeciso, a outra indica que a etapa de
# calibração não rodou e nenhum score foi verificado (#196).
def _calibration_issues(
    claims: tuple[SemanticClaim, ...], *, region_id: str | None
) -> list[AuditIssue]:
    """Reporta falhas da regra de calibração, sem confundi-las com discordância."""
    owner = "Scene context" if region_id is None else f"Region {region_id!r}"
    return [
        AuditIssue(
            AuditSeverity.WARNING,
            "calibration_failed",
            f"{owner} claim {claim.value!r} ({claim.kind.value}) has no verified score: "
            f"{claim.support.reason}.",
            region_id=region_id,
        )
        for claim in claims
        if claim.support is not None and claim.support.state is SupportState.FAILED
    ]


# Reporta slots de evidência que falharam, preservando o fato de que os
# demais slots da mesma region continuam válidos. Existe porque a #194 exige
# que uma falha isolada de slot seja auditável sem invalidar a region.
def _evidence_issues(region: ObservedRegion) -> list[AuditIssue]:
    """Reporta slots de evidência que falharam nesta region."""
    return [
        AuditIssue(
            AuditSeverity.WARNING,
            "evidence_slot_failed",
            f"Region {region.region_id!r} could not produce {slot.slot.value!r} evidence: {slot.reason}.",
            region_id=region.region_id,
        )
        for slot in region.evidence
        if slot.state is EvidenceState.FAILED
    ]


#: Categorias cuja natureza é inequívoca: uma parede, um piso ou um teto são
#: *stuff* — matéria contínua e não contável — em qualquer cena. A lista é
#: deliberadamente curta. Uma tabela completa de ``label -> kind`` corrigiria a
#: saída do reasoner em vez de expor o erro dele, e é exatamente o que a #202
#: proíbe nesta etapa: primeiro melhora-se a evidência visual, depois se mede se
#: o modelo passou a acertar.
_INHERENTLY_STUFF_CATEGORIES = frozenset(
    {"wall", "floor", "flooring", "ceiling", "ground", "sky"}
)


# Sinaliza incoerências óbvias entre a categoria e a natureza declaradas para
# uma região. **Nunca reescreve** a saída do modelo: o audit reporta, e a
# decisão de corrigir pertence a quem lê o relatório. Chamada por
# audit_observation uma vez por região.
def _region_kind_issues(region: ObservedRegion) -> list[AuditIssue]:
    """Reporta regiões cuja ``RegionKind`` contradiz uma categoria inequívoca."""
    claim = primary_label_claim(region)
    if claim is None or claim.category is None or claim.region_kind is None:
        return []
    category = normalize_claim_value(claim.category)
    if category not in _INHERENTLY_STUFF_CATEGORIES:
        return []
    if claim.region_kind is RegionKind.STUFF:
        return []
    return [
        AuditIssue(
            AuditSeverity.WARNING,
            "region_kind_inconsistent_with_category",
            f"Region {region.region_id!r} reports category {claim.category!r} with kind "
            f"{claim.region_kind.value!r}; that category is inherently 'stuff'.",
            region_id=region.region_id,
        )
    ]
