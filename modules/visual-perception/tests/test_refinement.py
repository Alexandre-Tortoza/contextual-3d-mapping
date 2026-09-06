"""Testes de refinamento seletivo guiado por incerteza (#183)."""

from __future__ import annotations

import numpy as np

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.application.refinement import (
    RefinementConfig,
    refine_observation,
    select_refinement_targets,
)
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext, VisualObservation
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Verifica que, com um threshold de confiança muito permissivo (1.0), toda região da
# observação é selecionada como alvo de refinamento.
def test_refinement_targets_low_confidence_regions() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    targets = select_refinement_targets(result.observation, RefinementConfig(low_confidence_threshold=1.0))
    assert set(targets) == {region.region_id for region in result.observation.regions}


# Confirma duas garantias do loop de refinamento: ele sempre termina (respeitando
# max_iterations) e nunca descarta evidência/claims que já existiam antes de refinar.
def test_refinement_loop_terminates_and_preserves_previous_evidence() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())
    original_claim_count = len(result.observation.regions[0].claims)

    refined, history = refine_observation(
        result.observation,
        payload,
        FakeMultimodalReasoner(),
        MultimodalReasoningConfig(),
        RefinementConfig(low_confidence_threshold=1.0, max_iterations=2),
    )

    assert len(history) <= 2
    assert len(refined.regions[0].claims) >= original_claim_count


# Garante o caso de custo zero: quando nenhuma região atinge o critério de refinamento,
# a observação sai inalterada e nenhuma iteração é registrada no histórico.
def test_no_targets_means_no_refinement_needed() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    refined, history = refine_observation(
        result.observation,
        payload,
        FakeMultimodalReasoner(),
        MultimodalReasoningConfig(),
        RefinementConfig(low_confidence_threshold=0.0, small_region_area_px=0, max_iterations=2),
    )
    assert history == ()
    assert refined == result.observation


# Monta uma observação mínima com uma única região carregando os claims dados;
# usada para exercitar a seleção de alvos sem passar pelo pipeline inteiro.
def _observation_with_claims(claims: tuple[SemanticClaim, ...]) -> VisualObservation:
    data = np.zeros((8, 8), dtype=np.bool_)
    data[0:6, 0:6] = True
    mask = Mask(data, 8, 8)
    region = ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p",), claims=claims)
    observation = run_canonical_pipeline(
        image_observation(width=8, height=8), payload_with_blobs(8, 8), default_config(), default_ports()
    ).observation
    return VisualObservation(
        source=observation.source,
        image_width=8,
        image_height=8,
        scene_context=SceneContext(claims=()),
        regions=(region,),
        relations=(),
    )


# Constrói um claim de label com o score dado, ou sem score quando ``confidence`` é None.
def _label_claim(value: str, confidence: float | None) -> SemanticClaim:
    from visual_perception.domain.semantics import ConfidenceScore

    score = None if confidence is None else ConfidenceScore(confidence, source="fake")
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        score,
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
    )


# Uma claim não pontuada não decide: ela não pode disparar a regra de "confiança
# baixa", porque não há confiança informada para comparar com o limiar.
def test_unscored_claim_does_not_trigger_the_low_confidence_rule() -> None:
    observation = _observation_with_claims((_label_claim("floor", None),))
    targets = select_refinement_targets(
        observation, RefinementConfig(low_confidence_threshold=0.5, small_region_area_px=0)
    )
    assert targets == ()


# O outro lado: uma claim efetivamente pontuada abaixo do limiar continua disparando
# o refinamento, como antes.
def test_scored_claim_below_threshold_still_triggers_refinement() -> None:
    observation = _observation_with_claims((_label_claim("floor", 0.2),))
    targets = select_refinement_targets(
        observation, RefinementConfig(low_confidence_threshold=0.5, small_region_area_px=0)
    )
    assert targets == ("region-a",)
