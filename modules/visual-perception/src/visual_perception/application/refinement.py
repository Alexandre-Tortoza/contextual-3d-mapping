"""Refinamento seletivo dirigido por suporte estruturado.

Issues: #183 (loop seletivo), #204 (razões explícitas e evidência nova),
#215 (a contradição que voltou a ser sinal).

O refinamento anterior tinha três regras — confiança abaixo de um limiar, área
abaixo de um limiar, e claims contraditórias — e **as três degeneraram** na
configuração real:

| regra                    | medido no run ``20260908T131207Z``                    |
| ------------------------ | ----------------------------------------------------- |
| ``confidence < 0.5``     | 165 de 165 claims valem exatamente 0,90: nunca dispara |
| ``area < 16 px``         | ``min_mask_area`` do perfil real é 500: nunca dispara  |
| ``contradictory_claims`` | 163 de 165 regiões: dispara em quase tudo              |

E, mesmo quando disparava, o loop chamava ``interpret_regions`` com **as mesmas
views, o mesmo prompt e temperatura zero** — pedir de novo esperando outra
resposta.

A #204 troca as três por razões que descrevem um estado de evidência concreto
(:class:`RefinementReason`), e exige que um passe de refinamento veja
**evidência nova**. O escalonamento é explícito na configuração: o primeiro
passe usa ``multimodal_reasoning.region_views``, e o refinamento usa
``refinement.escalation_views``. Quando os dois conjuntos coincidem, não há
evidência nova a oferecer e o loop termina sem gastar chamada nenhuma.

Nada é sobrescrito. Uma reinterpretação vira uma claim ``PRIMARY`` adicional, e
se ela discordar da primeira isso passa a ser uma contradição **real** entre
duas afirmações independentes — exatamente o caso que a política de
exclusividade preservou ao parar de contar alternativas como contradição.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from visual_perception.application.region_semantics import interpret_regions
from visual_perception.application.support import fingerprint_of
from visual_perception.config import MultimodalReasoningConfig, RefinementConfig
from visual_perception.domain.claim_exclusivity import contradicting_claims
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import EvidenceState
from visual_perception.domain.region_reasoning import RegionView
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantic_support import SupportSignalStatus, SupportState
from visual_perception.domain.semantics import ClaimKind, measured_signals
from visual_perception.domain.structural_consistency import StructuralVerdict, region_kind_verdict
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner

__all__ = [
    "RefinementReason",
    "RefinementStep",
    "RefinementTarget",
    "refine_observation",
    "region_refinement_reasons",
    "select_refinement_targets",
]


# Enumera por que uma região merece uma segunda interpretação. Existe para que
# a decisão seja auditável e contável — "esta região foi refinada porque o canal
# independente contradisse a hipótese primária" é uma frase verificável, e
# "porque o score ficou abaixo de 0,7" não era, já que o score é constante.
class RefinementReason(StrEnum):
    """O estado de evidência que justifica reinterpretar uma região."""

    #: A região não recebeu nenhuma hipótese de identidade.
    MISSING_SEMANTICS = "missing_semantics"
    #: Todos os sinais independentes medidos apontam contra a hipótese primária.
    UNSUPPORTED_PRIMARY = "unsupported_primary"
    #: Os sinais independentes não distinguem a primária das concorrentes.
    COMPETING_HYPOTHESES = "competing_hypotheses"
    #: Duas hipóteses afirmadas discordam entre si.
    CONTRADICTORY_CLAIMS = "contradictory_claims"
    #: O conceito declarado é incompatível com a natureza declarada.
    REGION_KIND_CONTRADICTION = "region_kind_contradiction"
    #: A calibração se absteve de pontuar a hipótese primária.
    ABSTAINED_SUPPORT = "abstained_support"
    #: A regra de calibração falhou para esta região.
    FAILED_CALIBRATION = "failed_calibration"
    #: Um slot de evidência configurado não pôde ser produzido.
    EVIDENCE_SLOT_UNAVAILABLE = "evidence_slot_unavailable"
    #: A máscara ocupa pouco do bounding box: a interpretação pode estar
    #: descrevendo o fundo do recorte em vez do sujeito.
    INSUFFICIENT_FOREGROUND = "insufficient_foreground"
    #: A região é pequena. **Nunca** justifica refinamento sozinha: tamanho é
    #: uma condição de risco, não um estado de evidência, e refinar toda região
    #: pequena gastaria a maior parte do orçamento sem nenhuma razão positiva.
    SMALL_REGION = "small_region"


#: Ordem determinística de prioridade quando o orçamento de refinamento é menor
#: que o número de regiões elegíveis. Razões que indicam evidência *ativa* em
#: sentido contrário vêm antes de razões que indicam apenas ausência.
_REASON_PRIORITY: tuple[RefinementReason, ...] = (
    RefinementReason.MISSING_SEMANTICS,
    RefinementReason.CONTRADICTORY_CLAIMS,
    RefinementReason.UNSUPPORTED_PRIMARY,
    RefinementReason.REGION_KIND_CONTRADICTION,
    RefinementReason.COMPETING_HYPOTHESES,
    RefinementReason.INSUFFICIENT_FOREGROUND,
    RefinementReason.FAILED_CALIBRATION,
    RefinementReason.ABSTAINED_SUPPORT,
    RefinementReason.EVIDENCE_SLOT_UNAVAILABLE,
    RefinementReason.SMALL_REGION,
)

#: Razões que, sozinhas, não justificam gastar uma chamada de modelo.
_MODIFIER_REASONS = frozenset({RefinementReason.SMALL_REGION})


# Descreve uma região selecionada para refinamento e por quê. Existe para que o
# histórico registre a razão junto do alvo: um histórico que só listasse ids não
# permitiria distinguir um refinamento bem motivado de um refinamento cego.
@dataclass(frozen=True)
class RefinementTarget:
    """Uma região selecionada para reinterpretação, com as razões que a elegeram."""

    region_id: str
    reasons: tuple[RefinementReason, ...]

    # Ordena os alvos por urgência da razão mais forte, depois por id para
    # manter a seleção determinística sob orçamento.
    @property
    def priority(self) -> tuple[int, str]:
        """A chave de ordenação usada quando o orçamento não cobre todos os alvos."""
        ranks = [_REASON_PRIORITY.index(reason) for reason in self.reasons]
        return (min(ranks) if ranks else len(_REASON_PRIORITY), self.region_id)


# Registro append-only de uma iteração de refinamento. Existe para responder,
# depois do run, exatamente o que a #204 exige: qual região, por qual razão, com
# qual evidência anterior, com qual evidência nova, por qual produtor, sob qual
# fingerprint, e com qual desfecho.
@dataclass(frozen=True)
class RefinementStep:
    """O que uma iteração de refinamento fez, e com base em quê."""

    iteration: int
    targets: tuple[RefinementTarget, ...]
    previous_evidence: tuple[str, ...]
    new_evidence: tuple[str, ...]
    producer: str
    config_fingerprint: str
    failures: tuple[RegionInterpretationFailure, ...] = ()
    refined_region_ids: tuple[str, ...] = ()

    # Expõe os ids alvo, que é o campo mais consultado por quem lê o histórico.
    @property
    def target_region_ids(self) -> tuple[str, ...]:
        """Os ids das regiões que esta iteração tentou reinterpretar."""
        return tuple(target.region_id for target in self.targets)


# Deriva as razões de refinamento de uma única região a partir do estado de
# evidência que os estágios anteriores anexaram. É pura e sem I/O, o que a torna
# testável caso a caso; chamada por select_refinement_targets.
def region_refinement_reasons(
    region: ObservedRegion, config: RefinementConfig
) -> tuple[RefinementReason, ...]:
    """Lista as razões pelas quais esta região precisa de outra interpretação.

    Argumentos:
        region: a região já interpretada, com sinais e suporte anexados.
        config: os limiares que definem foreground insuficiente e região pequena.
    Retorna:
        as razões encontradas, na ordem canônica de prioridade.
    """
    reasons: set[RefinementReason] = set()
    claim = primary_label_claim(region)
    if claim is None:
        reasons.add(RefinementReason.MISSING_SEMANTICS)
    else:
        signals = measured_signals(claim)
        if signals:
            if all(signal.status is SupportSignalStatus.CONTRADICTS for signal in signals):
                reasons.add(RefinementReason.UNSUPPORTED_PRIMARY)
            elif all(
                signal.status is not SupportSignalStatus.SUPPORTS for signal in signals
            ):
                reasons.add(RefinementReason.COMPETING_HYPOTHESES)
        if region_kind_verdict(claim) is StructuralVerdict.CONTRADICTS:
            reasons.add(RefinementReason.REGION_KIND_CONTRADICTION)
        if claim.support is not None:
            if claim.support.state is SupportState.ABSTAINED:
                reasons.add(RefinementReason.ABSTAINED_SUPPORT)
            elif claim.support.state is SupportState.FAILED:
                reasons.add(RefinementReason.FAILED_CALIBRATION)

    if contradicting_claims(region.claims, ClaimKind.LABEL):
        reasons.add(RefinementReason.CONTRADICTORY_CLAIMS)

    for slot in region.evidence:
        if slot.state is EvidenceState.FAILED:
            reasons.add(RefinementReason.EVIDENCE_SLOT_UNAVAILABLE)
        if (
            slot.state is EvidenceState.AVAILABLE
            and slot.is_foreground
            and slot.mask_fill_ratio is not None
            and slot.mask_fill_ratio < config.min_mask_fill_ratio
        ):
            reasons.add(RefinementReason.INSUFFICIENT_FOREGROUND)

    if region.mask.area() < config.small_region_area_px:
        reasons.add(RefinementReason.SMALL_REGION)

    return tuple(reason for reason in _REASON_PRIORITY if reason in reasons)


# Seleciona as regiões que precisam de outro passe, dentro do orçamento e sem
# nunca eleger uma região só por ser pequena. Chamada por refine_observation a
# cada iteração.
def select_refinement_targets(
    observation: VisualObservation, config: RefinementConfig
) -> tuple[RefinementTarget, ...]:
    """Seleciona, de forma determinística, as regiões a reinterpretar.

    Uma região cuja única razão está em :data:`_MODIFIER_REASONS` é descartada:
    tamanho pequeno acompanha uma razão de evidência, e não a substitui.

    Argumentos:
        observation: a observação a inspecionar.
        config: o orçamento e os limiares de seleção.
    Retorna:
        os alvos, ordenados por prioridade de razão e limitados ao orçamento.
    """
    targets = []
    for region in observation.regions:
        reasons = region_refinement_reasons(region, config)
        if not reasons or set(reasons) <= _MODIFIER_REASONS:
            continue
        targets.append(RefinementTarget(region_id=region.region_id, reasons=reasons))
    targets.sort(key=lambda target: target.priority)
    return tuple(targets[: config.max_regions_per_iteration])


# Ponto de entrada do estágio: reinterpreta as regiões que têm razão explícita,
# usando um conjunto de views diferente do que já foi usado. Chamada pelo
# pipeline canônico depois da calibração e antes da reconciliação.
def refine_observation(
    observation: VisualObservation,
    image: ImagePayload,
    views: Mapping[str, tuple[RegionView, ...]],
    reasoner: MultimodalReasoner,
    multimodal_config: MultimodalReasoningConfig,
    refinement_config: RefinementConfig,
) -> tuple[VisualObservation, tuple[RefinementStep, ...]]:
    """Reinterpreta seletivamente as regiões com evidência não resolvida.

    O loop termina por qualquer um de três motivos, todos determinísticos:
    nenhuma região tem razão, o escalonamento não oferece evidência diferente da
    já usada, ou ``max_iterations`` foi atingido.

    Argumentos:
        observation: a observação canônica a refinar.
        image: o payload da imagem completa, que define o frame das views.
        views: as views em pixels por ``region_id``, produzidas uma única vez
            pelo estágio de evidência e reusadas aqui — refinar não recorta a
            imagem de novo, e a geometria continua imutável.
        reasoner: o backend multimodal consultado no passe de refinamento.
        multimodal_config: a configuração usada no passe original; o
            escalonamento deriva dela trocando apenas as views.
        refinement_config: orçamento, limiares e views de escalonamento.
    Retorna:
        a observação refinada e o histórico append-only de cada iteração.
    """
    if not refinement_config.enabled or refinement_config.max_iterations <= 0:
        return observation, ()

    previous_evidence = tuple(multimodal_config.region_views)
    escalated_config = dataclasses.replace(
        multimodal_config, region_views=tuple(refinement_config.escalation_views)
    )
    new_evidence = tuple(escalated_config.region_views)
    if set(new_evidence) <= set(previous_evidence):
        return observation, ()

    current = observation
    history: list[RefinementStep] = []
    used_evidence: set[tuple[str, ...]] = {previous_evidence}

    for iteration in range(refinement_config.max_iterations):
        if new_evidence in used_evidence:
            break
        targets = select_refinement_targets(current, refinement_config)
        if not targets:
            break

        target_ids = {target.region_id for target in targets}
        order = {region.region_id: index for index, region in enumerate(current.regions)}
        selected = tuple(region for region in current.regions if region.region_id in target_ids)
        untouched = tuple(region for region in current.regions if region.region_id not in target_ids)

        refined, failures = interpret_regions(
            selected, image, views, current.scene_context, reasoner, escalated_config
        )
        current = dataclasses.replace(
            current,
            regions=tuple(sorted(untouched + refined, key=lambda region: order[region.region_id])),
        )
        failed_ids = {failure.region_id for failure in failures}
        history.append(
            RefinementStep(
                iteration=iteration,
                targets=targets,
                previous_evidence=previous_evidence,
                new_evidence=new_evidence,
                producer=escalated_config.backend,
                config_fingerprint=fingerprint_of(escalated_config),
                failures=failures,
                refined_region_ids=tuple(
                    sorted(target.region_id for target in targets if target.region_id not in failed_ids)
                ),
            )
        )
        used_evidence.add(new_evidence)

    return current, tuple(history)
