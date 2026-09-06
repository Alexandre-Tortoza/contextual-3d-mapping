"""Conversão da saída canônica do módulo para o contract de predição.

Issues: #198 (contract de predição), #199, #200 (experimentos que o consomem).

A avaliação consome apenas contracts versionados e não conhece
``VisualObservation``. Esta é a fronteira que traduz um para o outro, e ela
vive em ``experiments/`` justamente porque é a única camada autorizada a
depender dos dois lados.

A tradução preserva as distinções que a fase de calibração introduziu: score
bruto e score calibrado viajam em campos separados, e o motivo de uma
abstenção acompanha o valor. Nada é preenchido por conveniência: uma claim
sem score chega à avaliação sem score.
"""

from __future__ import annotations

from visual_perception.domain.region_evidence import EvidenceState
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, SemanticClaim
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.infrastructure.serialization import mask_to_dict
from visual_perception_evaluation.prediction import (
    PredictedRegion,
    PredictedRelation,
    SamplePrediction,
    ScoredValue,
    SupportState,
)

#: Como cada tipo de claim do módulo entra no contract de predição. Os tipos
#: de cena são tratados à parte, porque não pertencem a nenhuma região.
_REGION_CLAIM_FIELDS = {
    ClaimKind.LABEL: "labels",
    ClaimKind.ATTRIBUTE: "attributes",
    ClaimKind.CONDITION: "conditions",
    ClaimKind.MATERIAL: "materials",
    ClaimKind.HAZARD: "hazards",
}


# Converte uma observação canônica na predição versionada de uma amostra.
# Chamada pelos experimentos depois de cada execução do pipeline.
def to_sample_prediction(
    observation: VisualObservation,
    sample_id: str,
    *,
    latency_s: float | None = None,
    peak_vram_bytes: int | None = None,
    model_calls: int | None = None,
) -> SamplePrediction:
    """Traduz uma :class:`VisualObservation` para o contract de predição.

    Argumentos:
        observation: a saída canônica do pipeline.
        sample_id: identidade da amostra no conjunto de referência.
        latency_s: tempo medido da execução, quando disponível.
        peak_vram_bytes: pico de VRAM medido, quando disponível.
        model_calls: quantas chamadas de modelo a execução fez.
    Retorna:
        a :class:`SamplePrediction` correspondente.
    """
    return SamplePrediction(
        sample_id=sample_id,
        width=observation.image_width,
        height=observation.image_height,
        regions=tuple(_region(region) for region in observation.regions),
        relations=tuple(
            PredictedRelation(
                relation_id=relation.relation_id,
                subject_region_id=relation.subject_region_id,
                predicate=relation.predicate,
                object_region_id=relation.object_region_id,
                confidence=None if relation.confidence is None else relation.confidence.value,
            )
            for relation in observation.relations
        ),
        scene_claims=tuple(_scored(claim) for claim in observation.scene_context.claims),
        latency_s=latency_s,
        peak_vram_bytes=peak_vram_bytes,
        model_calls=model_calls,
    )


# Constrói a predição de uma amostra que falhou. Existe para que a falha
# entre na taxa de falha da avaliação em vez de a amostra desaparecer, o que
# inflaria silenciosamente as métricas de qualidade.
def failed_sample_prediction(
    sample_id: str, width: int, height: int, reason: str, *, latency_s: float | None = None
) -> SamplePrediction:
    """Constrói a predição explicitamente falha de uma amostra."""
    return SamplePrediction(
        sample_id=sample_id,
        width=width,
        height=height,
        failed=True,
        failure_reason=reason,
        latency_s=latency_s,
    )


# Converte uma região canônica, distribuindo suas claims pelos campos
# semânticos do contract e registrando quais slots de evidência ficaram
# disponíveis. Helper de to_sample_prediction.
def _region(region: ObservedRegion) -> PredictedRegion:
    """Traduz uma :class:`ObservedRegion` para o contract de predição."""
    grouped: dict[str, list[ScoredValue]] = {field: [] for field in _REGION_CLAIM_FIELDS.values()}
    for claim in region.claims:
        field = _REGION_CLAIM_FIELDS.get(claim.kind)
        if field is not None:
            grouped[field].append(_scored(claim))
    encoded = mask_to_dict(region.mask)
    return PredictedRegion(
        region_id=region.region_id,
        width=int(encoded["width"]),
        height=int(encoded["height"]),
        runs=tuple(int(run) for run in encoded["rle"]),
        labels=tuple(grouped["labels"]),
        attributes=tuple(grouped["attributes"]),
        conditions=tuple(grouped["conditions"]),
        materials=tuple(grouped["materials"]),
        hazards=tuple(grouped["hazards"]),
        evidence_slots=tuple(
            slot.slot.value for slot in region.evidence if slot.state is EvidenceState.AVAILABLE
        ),
    )


# Converte uma claim semântica em valor pontuado, mantendo bruto e calibrado
# separados e preservando o motivo de uma abstenção. Helper compartilhado
# pelas claims de região e de cena.
def _scored(claim: SemanticClaim) -> ScoredValue:
    """Traduz uma :class:`SemanticClaim` para o contract de predição."""
    support = claim.support
    calibrated = None if support is None else support.calibrated_confidence
    return ScoredValue(
        value=claim.value,
        raw_confidence=None if claim.confidence is None else claim.confidence.value,
        calibrated_confidence=None if calibrated is None else calibrated.value,
        support_state=SupportState(support.state.value) if support is not None else SupportState.UNKNOWN,
        reason=None if support is None else support.reason,
        source=(
            claim.confidence.source if claim.confidence is not None else claim.provenance.producer
        ),
        calibration_version=None if support is None else support.calibration_version,
        calibration_artifact=None if support is None else support.calibration_artifact,
        evidence_artifacts=(
            () if support is None else tuple(item.artifact_uri for item in support.evidence)
        ),
    )
