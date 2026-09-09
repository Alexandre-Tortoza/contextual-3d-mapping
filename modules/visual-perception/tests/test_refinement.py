"""Testes do refinamento seletivo dirigido por suporte estruturado (#183, #204)."""

from __future__ import annotations

import dataclasses

import numpy as np

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.application.refinement import (
    RefinementReason,
    refine_observation,
    region_refinement_reasons,
    select_refinement_targets,
)
from visual_perception.application.region_views import build_region_views
from visual_perception.config import MultimodalReasoningConfig, RefinementConfig
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace
from visual_perception.domain.geometry import CoordinateTransform, Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import (
    EvidenceSlot,
    EvidenceState,
    RegionEvidenceSlot,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantic_support import (
    HypothesisSupportSignal,
    SemanticSupport,
    SupportSignalStatus,
    SupportState,
)
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")
_LANGUAGE_SPACE = EmbeddingSpace(
    "clip", "openai/clip-vit-large-patch14", 768, EmbeddingModality.LANGUAGE_ALIGNED
)

#: Configuração usada pelos testes que afirmam a tupla exata de razões. Ela
#: desliga o modificador de tamanho para que o teste meça a razão que está
#: exercitando, e não o fato de a máscara do fixture ser pequena.
_NO_SIZE_MODIFIER = RefinementConfig(small_region_area_px=0)


# Constrói os sinais de alinhamento de uma hipótese com os desfechos pedidos.
# Existe para que cada teste declare exatamente qual estado de evidência está
# exercitando, em vez de depender de um default.
def _signals(value: str, *statuses: SupportSignalStatus) -> tuple[HypothesisSupportSignal, ...]:
    slots = (EvidenceSlot.MASKED_SUBJECT, EvidenceSlot.TIGHT_CROP, EvidenceSlot.CONTEXTUAL_CROP)
    return tuple(
        HypothesisSupportSignal(
            source="clip_alignment",
            hypothesis=value,
            slot=slot,
            status=status,
            score=0.2,
            margin=0.05 if status is SupportSignalStatus.SUPPORTS else -0.05,
            space=_LANGUAGE_SPACE,
        )
        for slot, status in zip(slots, statuses, strict=False)
    )


# Constrói uma claim de identidade com os campos que as razões de refinamento
# inspecionam: score bruto, natureza declarada, sinais e estado de calibração.
def _claim(
    value: str = "floor",
    *,
    confidence: float | None = 0.9,
    region_kind: RegionKind = RegionKind.STUFF,
    signals: tuple[HypothesisSupportSignal, ...] = (),
    support: SemanticSupport | None = None,
    role: HypothesisRole = HypothesisRole.PRIMARY,
) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        None if confidence is None else ConfidenceScore(confidence, source="fake"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=role,
        region_kind=region_kind,
        signals=signals,
        support=support,
    )


# Constrói uma região quadrada com os claims e slots pedidos.
def _region(
    claims: tuple[SemanticClaim, ...] = (),
    *,
    evidence: tuple[RegionEvidenceSlot, ...] = (),
    size: int = 6,
) -> ObservedRegion:
    data = np.zeros((32, 32), dtype=np.bool_)
    data[2 : 2 + size, 2 : 2 + size] = True
    mask = Mask(data, 32, 32)
    return ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p",), claims=claims, evidence=evidence)


# Constrói um slot de foreground com a fração de preenchimento pedida.
def _foreground_slot(mask_fill_ratio: float) -> RegionEvidenceSlot:
    return RegionEvidenceSlot(
        slot=EvidenceSlot.TIGHT_CROP,
        region_id="region-a",
        state=EvidenceState.AVAILABLE,
        crop_box=Mask(np.ones((4, 4), dtype=np.bool_), 4, 4).bounding_box(),
        transform=CoordinateTransform.identity(),
        preprocessing="crop:expansion=0.0",
        artifact_ref="language-region-a",
        space=_LANGUAGE_SPACE,
        mask_fill_ratio=mask_fill_ratio,
    )


# Monta uma observação mínima com uma única região, para exercitar a seleção de
# alvos sem passar pelo pipeline inteiro.
def _observation(region: ObservedRegion) -> VisualObservation:
    return VisualObservation(
        source=image_observation().source,
        image_width=32,
        image_height=32,
        scene_context=SceneContext(claims=()),
        regions=(region,),
        relations=(),
    )


# Uma região cujos sinais independentes todos apontam contra a hipótese
# primária é o caso central da #204: a razão é um estado de evidência concreto,
# e não um número abaixo de um limiar.
def test_a_primary_contradicted_by_every_signal_is_refined() -> None:
    region = _region((_claim(signals=_signals("floor", *([SupportSignalStatus.CONTRADICTS] * 3))),))

    reasons = region_refinement_reasons(region, RefinementConfig())

    assert RefinementReason.UNSUPPORTED_PRIMARY in reasons


# Quando nenhum sinal distingue a primária das concorrentes, a razão é
# ambiguidade, e não falta de suporte: são estados diferentes e o refinamento
# precisa poder contá-los separadamente.
def test_indistinguishable_signals_are_competing_hypotheses() -> None:
    region = _region(
        (_claim(signals=_signals("floor", *([SupportSignalStatus.INDISTINGUISHABLE] * 3))),)
    )

    reasons = region_refinement_reasons(region, RefinementConfig())

    assert RefinementReason.COMPETING_HYPOTHESES in reasons
    assert RefinementReason.UNSUPPORTED_PRIMARY not in reasons


# Um único sinal a favor já basta para a hipótese não ser considerada sem
# suporte: exigir unanimidade transformaria qualquer discordância parcial em
# refinamento, e a medição mostra que discordância parcial é o caso comum.
def test_a_supported_primary_is_not_refined() -> None:
    region = _region(
        (
            _claim(
                signals=_signals(
                    "floor",
                    SupportSignalStatus.SUPPORTS,
                    SupportSignalStatus.CONTRADICTS,
                    SupportSignalStatus.INDISTINGUISHABLE,
                )
            ),
        )
    )

    assert region_refinement_reasons(region, _NO_SIZE_MODIFIER) == ()


# Uma claim não pontuada não decide nada por si: ausência de score não é razão,
# porque não há valor informado para comparar com limiar nenhum.
def test_an_unscored_claim_is_not_a_reason_by_itself() -> None:
    assert region_refinement_reasons(_region((_claim(confidence=None),)), _NO_SIZE_MODIFIER) == ()


# A contradição estrutural entre conceito e natureza é razão própria, e é
# medível: 121 das 165 regiões do run de referência estão nesse estado.
def test_a_region_kind_contradiction_is_a_reason() -> None:
    region = _region((_claim("wall", region_kind=RegionKind.THING),))

    reasons = region_refinement_reasons(region, RefinementConfig())

    assert RefinementReason.REGION_KIND_CONTRADICTION in reasons


# Abstenção da calibração e falha da regra são razões distintas: uma diz que a
# evidência não bastou, a outra que a etapa quebrou.
def test_abstained_and_failed_support_are_distinct_reasons() -> None:
    abstained = _region(
        (_claim(support=SemanticSupport(state=SupportState.ABSTAINED, reason="insufficient support")),)
    )
    failed = _region(
        (_claim(support=SemanticSupport(state=SupportState.FAILED, reason="artifact could not be loaded")),)
    )

    assert RefinementReason.ABSTAINED_SUPPORT in region_refinement_reasons(abstained, RefinementConfig())
    assert RefinementReason.FAILED_CALIBRATION in region_refinement_reasons(failed, RefinementConfig())


# Uma máscara que ocupa pouco do próprio bounding box é razão porque a
# interpretação pode estar descrevendo o fundo do recorte.
def test_insufficient_foreground_is_a_reason() -> None:
    region = _region((_claim(),), evidence=(_foreground_slot(0.05),))

    reasons = region_refinement_reasons(region, RefinementConfig())

    assert RefinementReason.INSUFFICIENT_FOREGROUND in reasons


# Tamanho é um modificador, nunca uma razão: uma região pequena e sem nenhum
# problema de evidência não gasta chamada de modelo.
def test_a_small_region_alone_is_never_a_target() -> None:
    region = _region((_claim(),), size=4)
    config = RefinementConfig(small_region_area_px=10_000)

    assert region_refinement_reasons(region, config) == (RefinementReason.SMALL_REGION,)
    assert select_refinement_targets(_observation(region), config) == ()


# Mas ele acompanha uma razão real, e aparece no registro: o histórico precisa
# mostrar que a região era pequena *além* de ter evidência não resolvida.
def test_a_small_region_with_a_real_reason_is_selected_and_records_both() -> None:
    region = _region(
        (_claim("wall", region_kind=RegionKind.THING),), size=4
    )
    config = RefinementConfig(small_region_area_px=10_000)

    targets = select_refinement_targets(_observation(region), config)

    assert len(targets) == 1
    assert set(targets[0].reasons) == {
        RefinementReason.REGION_KIND_CONTRADICTION,
        RefinementReason.SMALL_REGION,
    }


# Uma região sem nenhuma hipótese é o caso mais simples de razão.
def test_a_region_without_semantics_is_a_target() -> None:
    targets = select_refinement_targets(_observation(_region()), _NO_SIZE_MODIFIER)

    assert [target.region_id for target in targets] == ["region-a"]
    assert targets[0].reasons == (RefinementReason.MISSING_SEMANTICS,)


# O orçamento é respeitado e a seleção é determinística: com mais alvos que
# vagas, quem entra é decidido pela prioridade da razão mais forte.
def test_the_budget_is_respected_and_selection_is_deterministic() -> None:
    observation = run_canonical_pipeline(
        image_observation(), payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),)),
        default_config(), default_ports()
    ).observation
    config = RefinementConfig(max_regions_per_iteration=1, small_region_area_px=10_000)

    first = select_refinement_targets(observation, config)
    second = select_refinement_targets(observation, config)

    assert len(first) <= 1
    assert first == second


# A garantia central da #204: um passe de refinamento precisa ver evidência
# nova. Quando o escalonamento repete as views já usadas, o loop termina sem
# gastar chamada nenhuma — pedir de novo com a mesma entrada e temperatura zero
# é pedir de novo esperando outra resposta.
def test_refinement_without_new_evidence_does_nothing() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())
    reasoning = MultimodalReasoningConfig(region_views=("masked_subject", "tight_crop"))

    refined, history = refine_observation(
        result.observation,
        payload,
        build_region_views(result.observation.regions, payload, default_config()),
        FakeMultimodalReasoner(),
        reasoning,
        RefinementConfig(escalation_views=("tight_crop", "masked_subject")),
    )

    assert history == ()
    assert refined == result.observation


# O loop termina, o histórico registra razão e caminho de evidência, e nenhuma
# claim anterior é perdida: o refinamento acrescenta, nunca sobrescreve.
def test_refinement_appends_and_records_the_evidence_path() -> None:
    config = dataclasses.replace(
        default_config(),
        multi_context=dataclasses.replace(
            default_config().multi_context, contextual_crop_enabled=True
        ),
    )
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, config, default_ports())
    observation = dataclasses.replace(
        result.observation,
        regions=tuple(
            dataclasses.replace(region, claims=()) for region in result.observation.regions
        ),
    )
    reasoning = MultimodalReasoningConfig(region_views=("masked_subject",))

    refined, history = refine_observation(
        observation,
        payload,
        build_region_views(observation.regions, payload, config),
        FakeMultimodalReasoner(),
        reasoning,
        RefinementConfig(
            escalation_views=("masked_subject", "tight_crop", "contextual_crop"),
            max_iterations=2,
            small_region_area_px=0,
        ),
    )

    assert len(history) == 1
    step = history[0]
    assert step.previous_evidence == ("masked_subject",)
    assert step.new_evidence == ("masked_subject", "tight_crop", "contextual_crop")
    assert step.producer == "fake"
    assert step.config_fingerprint
    assert step.targets[0].reasons == (RefinementReason.MISSING_SEMANTICS,)
    assert all(len(region.claims) > 0 for region in refined.regions)
    # A geometria é imutável durante o raciocínio: ids, boxes e ordem intactos.
    assert [region.region_id for region in refined.regions] == [
        region.region_id for region in observation.regions
    ]
    assert [region.box for region in refined.regions] == [
        region.box for region in observation.regions
    ]


# Desligar o estágio é um caminho de custo zero explícito.
def test_a_disabled_refinement_stage_returns_the_observation_untouched() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    refined, history = refine_observation(
        result.observation,
        payload,
        build_region_views(result.observation.regions, payload, default_config()),
        FakeMultimodalReasoner(),
        MultimodalReasoningConfig(),
        RefinementConfig(enabled=False),
    )

    assert history == ()
    assert refined is result.observation
