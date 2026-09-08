"""Testes do contract ambiental de cena e do isolamento do ego (#202).

O run real de ``corridor-02-002`` produziu, como contexto de cena,
``attributes: ["fisheye lens", "carpeted floor", "suitcase"]`` e a descrição
"there is a suitcase on the ground in the foreground". A "mala" é o próprio
quad que carrega a câmera. Esse texto ia inteiro para o prompt de cada região,
condicionando a interpretação local com um objeto que não existe na cena.

A correção tem duas metades, e as duas são estruturais:

1. o VLM não vê mais o rig ao analisar a cena — a view de cena é recortada
   para a área válida menos a área do ego, sem pintar pixel nenhum;
2. o contract de cena deixa de ter inventário livre de objetos e passa a ser
   ambiental: ``scene_type``, ``environment``, ``layout``, ``lighting``,
   ``visibility``, ``navigability``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from visual_perception.application.scene_context import analyze_scene
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry, ImageAreaMasks
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_reasoning import (
    REGION_SCENE_CLAIM_KINDS,
    SceneContextMode,
    select_region_scene_claims,
)
from visual_perception.domain.semantics import ClaimKind

_WIDTH = 64
_HEIGHT = 64


# Resposta ambiental completa e válida, com overrides pontuais.
def _response(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "scene_type": "corridor",
        "environment": "indoor",
        "layout": "a narrow corridor running away from the camera",
        "lighting": "dim artificial light",
        "visibility": "clear",
        "navigability": "open path ahead",
        "confidence": 0.85,
    }
    base.update(overrides)
    return base


# Reasoner de teste que devolve a resposta pedida e registra a imagem recebida,
# para que os testes possam afirmar o que o VLM efetivamente viu.
class _RecordingReasoner:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.seen: ImagePayload | None = None

    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> Any:
        self.seen = image
        return self.response

    def analyze_region(self, request: Any, config: MultimodalReasoningConfig) -> Any:
        raise AssertionError("analyze_region must not be called by the scene stage.")


def _image() -> ImagePayload:
    return ImagePayload(np.full((_HEIGHT, _WIDTH, 3), 200, dtype=np.uint8), width=_WIDTH, height=_HEIGHT)


# Cada campo ambiental vira um claim tipado próprio, em vez de texto livre
# dentro de uma descrição.
def test_environmental_fields_become_typed_claims() -> None:
    """Os seis campos ambientais viram claims de kinds distintos."""
    context = analyze_scene(_image(), _RecordingReasoner(_response()), MultimodalReasoningConfig())

    by_kind = {claim.kind: claim.value for claim in context.claims}
    assert by_kind[ClaimKind.SCENE_TYPE] == "corridor"
    assert by_kind[ClaimKind.ENVIRONMENT] == "indoor"
    assert by_kind[ClaimKind.LAYOUT] == "a narrow corridor running away from the camera"
    assert by_kind[ClaimKind.LIGHTING] == "dim artificial light"
    assert by_kind[ClaimKind.VISIBILITY] == "clear"
    assert by_kind[ClaimKind.NAVIGABILITY] == "open path ahead"


# Só ``scene_type`` carrega o score do produtor; os campos ambientais são
# descritivos e não podem carregar confiança bruta (#195).
def test_only_the_scene_type_carries_a_reported_score() -> None:
    """Os campos ambientais ficam sem score bruto."""
    context = analyze_scene(_image(), _RecordingReasoner(_response()), MultimodalReasoningConfig())

    for claim in context.claims:
        if claim.kind is ClaimKind.SCENE_TYPE:
            assert claim.confidence is not None
            assert claim.confidence.value == pytest.approx(0.85)
        else:
            assert claim.confidence is None


# Um campo ambiental ausente é ausência: nenhum claim é emitido, e nada é
# preenchido com string vazia nem com um valor default.
def test_a_missing_environmental_field_produces_no_claim() -> None:
    """Campo ambiental ausente não vira claim vazio."""
    context = analyze_scene(
        _image(), _RecordingReasoner(_response(lighting=None)), MultimodalReasoningConfig()
    )

    assert ClaimKind.LIGHTING not in {claim.kind for claim in context.claims}
    assert all(claim.value for claim in context.claims)


# O inventário livre de objetos sai do contract: uma resposta que ainda traga
# ``attributes`` é malformada, e não silenciosamente aceita. Era por ali que a
# "suitcase" entrava.
def test_a_free_object_inventory_is_rejected() -> None:
    """``attributes`` não faz mais parte do contract de cena."""
    with pytest.raises(ValueError, match="attributes"):
        analyze_scene(
            _image(),
            _RecordingReasoner(_response(attributes=["fisheye lens", "suitcase"])),
            MultimodalReasoningConfig(),
        )


# O que chega ao raciocínio de região é exatamente o conjunto ambiental.
# Nenhum objeto inferido globalmente pode induzir a identidade de uma região.
def test_only_environmental_claims_reach_region_reasoning() -> None:
    """As claims propagadas à região são todas ambientais."""
    context = analyze_scene(_image(), _RecordingReasoner(_response()), MultimodalReasoningConfig())

    selected = select_region_scene_claims(context, mode=SceneContextMode.CONTEXT_ASSISTED)

    assert selected
    assert {claim.kind for claim in selected} <= REGION_SCENE_CLAIM_KINDS
    assert ClaimKind.SCENE_DESCRIPTION not in REGION_SCENE_CLAIM_KINDS
    assert ClaimKind.ATTRIBUTE not in REGION_SCENE_CLAIM_KINDS
    assert ClaimKind.HAZARD not in REGION_SCENE_CLAIM_KINDS


# A view de cena exclui a área do ego-veículo. É a metade estrutural da
# correção: o modelo não pode inventariar um objeto que não recebeu.
def test_the_scene_view_excludes_the_ego_vehicle_area() -> None:
    """A cena é analisada sobre a área válida menos a área do rig."""
    ego = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    ego[48:, :] = True
    masks = ImageAreaMasks(ego_vehicle=Mask(ego, _WIDTH, _HEIGHT))
    reasoner = _RecordingReasoner(_response())

    analyze_scene(_image(), reasoner, MultimodalReasoningConfig(), area_masks=masks)

    assert reasoner.seen is not None
    assert reasoner.seen.height == 48
    assert reasoner.seen.width == _WIDTH


# Excluir o rig é um recorte, nunca uma pintura: os pixels de origem
# permanecem byte a byte iguais, que é a regra estabelecida pela #212.
def test_excluding_the_ego_area_never_alters_the_source_pixels() -> None:
    """A exclusão do rig na cena é recorte, não pintura."""
    ego = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    ego[48:, :] = True
    image = _image()
    before = image.pixels.copy()

    analyze_scene(
        image,
        _RecordingReasoner(_response()),
        MultimodalReasoningConfig(),
        area_masks=ImageAreaMasks(ego_vehicle=Mask(ego, _WIDTH, _HEIGHT)),
    )

    assert np.array_equal(image.pixels, before)


# Sem geometria declarada, a cena é analisada sobre a imagem inteira: nada é
# recortado por suposição.
def test_without_declared_areas_the_whole_image_is_analysed() -> None:
    """Sem áreas declaradas, a cena recebe o frame inteiro."""
    reasoner = _RecordingReasoner(_response())

    analyze_scene(_image(), reasoner, MultimodalReasoningConfig(), area_masks=ImageAreaMasks())

    assert reasoner.seen is not None
    assert (reasoner.seen.width, reasoner.seen.height) == (_WIDTH, _HEIGHT)


# A área válida do sensor também limita a cena: o corpo preto da lente não é
# cena, e incluí-lo faz o modelo descrever a vinheta.
def test_the_scene_view_is_limited_to_the_valid_sensor_area() -> None:
    """A cena é recortada à área útil do sensor."""
    valid = ImageAreaGeometry(circle=CircleArea(32.0, 32.0, 20.0)).rasterize(_WIDTH, _HEIGHT)
    reasoner = _RecordingReasoner(_response())

    analyze_scene(
        _image(), reasoner, MultimodalReasoningConfig(), area_masks=ImageAreaMasks(valid_area=valid)
    )

    assert reasoner.seen is not None
    assert reasoner.seen.width < _WIDTH
    assert reasoner.seen.height < _HEIGHT
