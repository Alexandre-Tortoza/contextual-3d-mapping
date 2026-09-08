"""Testes do auditor de qualidade da observação visual (#168)."""

from __future__ import annotations

import dataclasses

import numpy as np
from contextual_mapping_contracts import FrameId, ObservationReference, Timestamp

from visual_perception.application.quality_audit import audit_observation
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation


# Helper que monta uma Mask pequena e fixa, reutilizada por todos os testes deste arquivo.
def _mask(width: int = 8, height: int = 8) -> Mask:
    data = np.zeros((height, width), dtype=np.bool_)
    data[0:2, 0:2] = True
    return Mask(data, width, height)


# Helper que monta um SemanticClaim mínimo com o value dado, para testar cenários de
# claims conflitantes/consistentes sem repetir a construção completa.
def _claim(value: str) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.8, source="fake"),
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
        role=HypothesisRole.PRIMARY,
    )


# Helper que monta uma ObservedRegion com os claims dados.
def _region(region_id: str, claims: tuple[SemanticClaim, ...] = ()) -> ObservedRegion:
    mask = _mask()
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",), claims=claims)


# Helper que monta uma ObservationReference mínima válida.
def _source() -> ObservationReference:
    return ObservationReference(
        observation_id="obs-1",
        dataset_id="ds",
        sequence_id="seq",
        sensor_id="cam",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=0, clock_id="rosbag"),
        frame_id=FrameId("cam_optical_frame"),
    )


# Helper que monta uma VisualObservation completa a partir das regiões dadas.
def _observation(regions: tuple[ObservedRegion, ...]) -> VisualObservation:
    return VisualObservation(_source(), 8, 8, SceneContext(), regions, ())


# Verifica o caso feliz: uma observação bem formada passa no audit sem nenhum issue.
def test_valid_observation_passes_without_modification() -> None:
    observation = _observation((_region("a", (_claim("box"),)),))
    result = audit_observation(observation)
    assert result.passed
    assert result.issues == ()


# Confirma o comportamento intencional de "claims, não labels" (ver
# docs/research-traceability.md): claims contraditórios na mesma região são sinalizados
# como warning, mas não fazem a observação inteira falhar no audit.
def test_contradictory_claims_are_flagged_but_observation_still_passes() -> None:
    observation = _observation((_region("a", (_claim("box"), _claim("crate"))),))
    result = audit_observation(observation)
    assert result.passed
    assert any(issue.code == "contradictory_claims" for issue in result.warnings)


# Garante que uma inconsistência estrutural real (a bounding box não bate com a mask) é
# tratada como erro — não como warning — e falha o audit.
def test_box_mask_mismatch_is_an_error() -> None:
    region = _region("a")
    tampered = dataclasses.replace(region, box=type(region.box)(0, 0, 100, 100))
    result = audit_observation(_observation((tampered,)))
    assert not result.passed
    assert any(issue.code == "box_mask_mismatch" for issue in result.errors)


# Confirma que o audit é uma função pura/determinística: rodá-lo duas vezes sobre a
# mesma observação produz exatamente o mesmo resultado.
def test_audit_is_deterministic() -> None:
    observation = _observation((_region("a", (_claim("box"), _claim("crate"))),))
    first = audit_observation(observation)
    second = audit_observation(observation)
    assert first == second


# Constrói um claim descritivo (sem score e sem papel), o formato que
# region_semantics anexa para description, attributes, condition e material.
def _descriptive(kind: ClaimKind, value: str) -> SemanticClaim:
    return SemanticClaim(
        kind,
        value,
        None,
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
    )


# Regressão da #202: até então, dois atributos com texto diferente na mesma
# região viravam um warning ``contradictory_claims``. Medido em
# ``corridor-02-002``, isso marcava as 60 regiões do frame, já que toda região
# recebia ao menos uma ``description`` e um ``attribute``.
def test_multiple_attributes_are_not_reported_as_contradictory() -> None:
    """Atributos coexistentes não produzem warning de contradição."""
    region = _region(
        "region-a",
        (
            _claim("wall"),
            _descriptive(ClaimKind.ATTRIBUTE, "The subject region appears to be a plain wall."),
            _descriptive(ClaimKind.ATTRIBUTE, "smooth"),
            _descriptive(ClaimKind.ATTRIBUTE, "white"),
        ),
    )
    observation = VisualObservation(
        source=_source(),
        image_width=8,
        image_height=8,
        scene_context=SceneContext(),
        regions=(region,),
        relations=(),
    )

    result = audit_observation(observation)

    assert [issue.code for issue in result.issues] == []


# Materiais diferentes descrevem um objeto composto, não uma contradição: uma
# porta pode ser de madeira e de vidro.
def test_multiple_materials_are_not_reported_as_contradictory() -> None:
    """Materiais coexistentes não produzem warning de contradição."""
    region = _region(
        "region-a",
        (
            _claim("door"),
            _descriptive(ClaimKind.MATERIAL, "wood"),
            _descriptive(ClaimKind.MATERIAL, "glass"),
        ),
    )
    observation = VisualObservation(
        source=_source(),
        image_width=8,
        image_height=8,
        scene_context=SceneContext(),
        regions=(region,),
        relations=(),
    )

    assert audit_observation(observation).issues == ()


# A contrapartida: hipóteses de identidade concorrentes continuam sendo
# sinalizadas. Corrigir o falso positivo não pode apagar a contradição real.
def test_competing_identity_hypotheses_are_still_flagged() -> None:
    """Duas hipóteses de identidade distintas continuam gerando warning."""
    region = _region(
        "region-a",
        (
            _claim("wall"),
            SemanticClaim(
                ClaimKind.LABEL,
                "door",
                ConfidenceScore(0.2, source="fake"),
                (Evidence("e"),),
                ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
                role=HypothesisRole.ALTERNATIVE,
            ),
        ),
    )
    observation = VisualObservation(
        source=_source(),
        image_width=8,
        image_height=8,
        scene_context=SceneContext(),
        regions=(region,),
        relations=(),
    )

    result = audit_observation(observation)

    assert [issue.code for issue in result.issues] == ["contradictory_claims"]
    assert result.passed


# Invariantes óbvios de natureza de região, usados só como audit. A lista é
# deliberadamente curta: uma tabela completa de label→kind mascararia o erro do
# reasoner em vez de expô-lo, que é o oposto do que a #202 pede.
def test_a_stuff_category_reported_as_thing_is_flagged() -> None:
    """Uma categoria inerentemente ``stuff`` marcada como ``thing`` vira warning."""
    claim = SemanticClaim(
        ClaimKind.LABEL,
        "plain wall",
        ConfidenceScore(0.9, source="fake"),
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
        role=HypothesisRole.PRIMARY,
        category="wall",
        region_kind=RegionKind.THING,
    )
    observation = VisualObservation(
        source=_source(),
        image_width=8,
        image_height=8,
        scene_context=SceneContext(),
        regions=(_region("region-a", (claim,)),),
        relations=(),
    )

    result = audit_observation(observation)

    assert [issue.code for issue in result.issues] == ["region_kind_inconsistent_with_category"]
    # É sinal, não correção: a observação continua válida e a saída do modelo
    # permanece exatamente como ele a produziu.
    assert result.passed
    assert observation.regions[0].claims[0].region_kind is RegionKind.THING


# A mesma categoria com o kind coerente não gera nada.
def test_a_stuff_category_reported_as_stuff_is_not_flagged() -> None:
    """Categoria e kind coerentes não produzem warning."""
    claim = SemanticClaim(
        ClaimKind.LABEL,
        "carpet",
        None,
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
        role=HypothesisRole.PRIMARY,
        category="flooring",
        region_kind=RegionKind.STUFF,
    )
    observation = VisualObservation(
        source=_source(),
        image_width=8,
        image_height=8,
        scene_context=SceneContext(),
        regions=(_region("region-a", (claim,)),),
        relations=(),
    )

    assert audit_observation(observation).issues == ()


# Uma categoria fora do conjunto de invariantes não é julgada. O módulo não tem
# ontologia de categorias, e inventar uma transformaria toda categoria nova em
# suspeita.
def test_a_category_outside_the_invariants_is_not_judged() -> None:
    """Categoria desconhecida não gera warning, qualquer que seja o kind."""
    claim = SemanticClaim(
        ClaimKind.LABEL,
        "fire extinguisher",
        None,
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
        role=HypothesisRole.PRIMARY,
        category="safety equipment",
        region_kind=RegionKind.STUFF,
    )
    observation = VisualObservation(
        source=_source(),
        image_width=8,
        image_height=8,
        scene_context=SceneContext(),
        regions=(_region("region-a", (claim,)),),
        relations=(),
    )

    assert audit_observation(observation).issues == ()
