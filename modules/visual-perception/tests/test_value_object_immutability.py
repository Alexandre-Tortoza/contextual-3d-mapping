"""Testes de imutabilidade, igualdade e hashabilidade dos value objects de domínio (#244)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.application.region_views import crop_payload
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.grounding import (
    GroundingPrediction,
    GroundingStatus,
    SemanticGrounding,
    SpatialRegionFootprint,
)
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import SubjectEmphasis
from visual_perception.domain.regions import ObservedRegion


# Constrói uma máscara 8x8 com um quadrado ocupado.
def _mask() -> Mask:
    data = np.zeros((8, 8), dtype=np.bool_)
    data[2:6, 2:6] = True
    return Mask(data, 8, 8)


# Constrói um mapa de features pequeno, com suporte válido declarado.
def _feature_map() -> FeatureMap:
    return FeatureMap(
        data=np.ones((2, 2, 3), dtype=np.float32),
        stride_x=4.0,
        stride_y=4.0,
        dimension=3,
        model_id="fake",
        valid_support=np.ones((2, 2), dtype=np.bool_),
    )


# Regressão da #244: ``frozen`` só impedia reatribuir o campo, e escrever no
# array através do objeto funcionava — uma máscara "congelada" mudava.
@pytest.mark.parametrize(
    ("build", "attribute"),
    [
        (_mask, "data"),
        (lambda: ImagePayload(np.zeros((4, 4, 3), dtype=np.uint8), 4, 4), "pixels"),
        (_feature_map, "data"),
        (_feature_map, "valid_support"),
    ],
)
def test_arrays_held_by_domain_value_objects_refuse_writes(build, attribute: str) -> None:
    """Escrever em qualquer array guardado por um value object levanta erro."""
    value_object = build()
    with pytest.raises(ValueError, match="read-only"):
        getattr(value_object, attribute)[0, 0] = 0


# A proteção é uma visão sobre os dados: ela não pode desligar a escrita do
# array de quem construiu o objeto.
def test_freezing_does_not_change_the_callers_array() -> None:
    """O array entregue na construção continua gravável para quem o criou."""
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    ImagePayload(pixels, 4, 4)
    pixels[0, 0, 0] = 1
    assert pixels.flags.writeable


# Todo ramo de ``crop_payload`` precisa entregar a mesma semântica de posse:
# antes, o ramo sem ênfase devolvia uma visão gravável da imagem de origem e os
# demais devolviam cópias.
@pytest.mark.parametrize("emphasis", list(SubjectEmphasis))
def test_every_crop_emphasis_returns_read_only_pixels(emphasis: SubjectEmphasis) -> None:
    """Nenhum ramo de recorte entrega pixels graváveis."""
    image = ImagePayload(np.full((8, 8, 3), 200, dtype=np.uint8), 8, 8)
    region = ObservedRegion("region-a", _mask(), BoundingBox(2, 2, 6, 6), 0.9, ("p-1",))

    cropped = crop_payload(image, region, BoundingBox(1, 1, 7, 7), emphasis=emphasis)

    assert not cropped.pixels.flags.writeable


# A hashabilidade é declarada, não acidental: regiões não entram em set nem em
# chave de dict, e o erro nomeia o tipo usado em vez de ``Mask``.
def test_regions_are_explicitly_unhashable() -> None:
    """``hash`` de uma região levanta ``TypeError`` nomeando ``ObservedRegion``."""
    region = ObservedRegion("region-a", _mask(), BoundingBox(2, 2, 6, 6), 0.9, ("p-1",))
    with pytest.raises(TypeError, match="ObservedRegion"):
        hash(region)


# Payload e mapa de features se apresentam como value objects; a igualdade
# passou a ser por conteúdo, e não por identidade.
def test_payloads_and_feature_maps_compare_by_content() -> None:
    """Dois objetos com os mesmos dados são iguais; dados diferentes, não."""
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert ImagePayload(pixels, 4, 4) == ImagePayload(pixels.copy(), 4, 4)
    assert ImagePayload(pixels, 4, 4) != ImagePayload(pixels[::-1].copy(), 4, 4)
    assert _feature_map() == _feature_map()
    other = _feature_map()
    assert other != FeatureMap(
        data=np.zeros((2, 2, 3), dtype=np.float32), stride_x=4.0, stride_y=4.0, dimension=3, model_id="fake"
    )


# O diagnóstico publicado junto do grounding era um dict mutável num tipo
# congelado: qualquer consumidor podia reescrevê-lo.
def test_grounding_diagnostics_are_read_only() -> None:
    """Alterar o diagnóstico de grounding ou de footprint levanta erro."""
    prediction = GroundingPrediction(region_id="region-a", concept="chair", reason="no detection")
    grounding = SemanticGrounding(prediction, GroundingStatus.FAILED, diagnostics={"reason": "no detection"})
    footprint = SpatialRegionFootprint("region-a", "chair", None, False, {"reason": "no detection"})

    for diagnostics in (grounding.diagnostics, footprint.diagnostics):
        with pytest.raises(TypeError):
            diagnostics["reason"] = "rewritten"  # type: ignore[index]
    assert dict(grounding.diagnostics) == {"reason": "no detection"}
