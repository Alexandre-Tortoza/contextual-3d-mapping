"""Testes das views que o reasoner multimodal recebe por região (#203/#202).

O run real de ``corridor-02-002`` mostrou que só o embedding denso era
mask-aware: ``tight_crop`` e ``contextual_crop`` chegavam ao VLM com
``mask_ref = null``, isto é, sem nenhuma indicação de quais pixels do recorte
são o sujeito. Numa região cuja máscara ocupa parte pequena do bounding box, o
que o modelo interpretava era a caixa, não a região.

Estes testes fixam as duas evidências que fecham essa lacuna: um recorte com o
sujeito isolado sobre fundo neutro, e um recorte de contexto com o sujeito
demarcado — sem destruir o contexto que serve para desambiguar.
"""

from __future__ import annotations

import numpy as np

from visual_perception.application.region_views import build_region_views, crop_payload
from visual_perception.config import ModuleConfig, MultiContextConfig
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import FOREGROUND_SLOTS, EvidenceSlot, SubjectEmphasis
from visual_perception.domain.regions import ObservedRegion

_WIDTH = 40
_HEIGHT = 40


# Imagem sintética com um gradiente por linha, para que qualquer pixel alterado
# pelo preprocessamento seja detectável por comparação exata.
def _image() -> ImagePayload:
    pixels = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    for row in range(_HEIGHT):
        pixels[row, :, :] = (row * 6) % 256
    return ImagePayload(pixels, width=_WIDTH, height=_HEIGHT)


# Região quadrada e pequena no meio do frame, com máscara menor que sua caixa
# quando ``diagonal`` é pedido — o caso em que o bounding box domina.
def _region(*, diagonal: bool = False) -> ObservedRegion:
    data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    if diagonal:
        for offset in range(12):
            data[14 + offset, 14 + offset] = True
    else:
        data[14:26, 14:26] = True
    mask = Mask(data, _WIDTH, _HEIGHT)
    return ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("region-a-p",))


# Configuração com todos os slots ligados, para exercitar as views juntas.
def _config() -> ModuleConfig:
    return ModuleConfig(
        multi_context=MultiContextConfig(
            tight_crop_enabled=True,
            contextual_crop_enabled=True,
            masked_subject_enabled=True,
        )
    )


# Localiza uma view pelo slot; devolve ``None`` quando ela não foi construída.
def _view(views: dict[str, tuple], slot: EvidenceSlot):  # type: ignore[type-arg]
    for view in views["region-a"]:
        if view.slot is slot:
            return view
    return None


# O slot ``masked_subject`` existe e é evidência de foreground: ele mostra o
# sujeito e nada mais.
def test_masked_subject_is_a_foreground_slot() -> None:
    assert EvidenceSlot.MASKED_SUBJECT in FOREGROUND_SLOTS


# O sujeito é isolado sobre um fundo neutro — e o fundo **não** é preto. Preto é
# exatamente a cor da vinheta do fisheye nestes frames: usá-lo tornaria "fora da
# máscara" indistinguível de "fora da lente" para o modelo.
def test_masked_subject_neutralizes_the_background_without_using_black() -> None:
    """O fundo do recorte isolado é neutro e distinguível da vinheta."""
    views = build_region_views((_region(),), _image(), _config())
    view = _view(views, EvidenceSlot.MASKED_SUBJECT)

    assert view is not None
    assert view.masked
    # O recorte é exatamente a caixa da máscara quadrada, então não há fundo a
    # neutralizar: um recorte diagonal expõe o fundo.
    diagonal_views = build_region_views((_region(diagonal=True),), _image(), _config())
    diagonal = _view(diagonal_views, EvidenceSlot.MASKED_SUBJECT)
    assert diagonal is not None
    background = diagonal.payload.pixels[0, 1]
    assert not np.array_equal(background, np.zeros(3, dtype=np.uint8))


# Os pixels do sujeito continuam sendo os pixels originais: a neutralização
# afeta só o fundo, ou a evidência deixaria de descrever a região.
def test_masked_subject_preserves_the_subject_pixels() -> None:
    """Dentro da máscara, os pixels do recorte isolado são os originais."""
    image = _image()
    views = build_region_views((_region(diagonal=True),), image, _config())
    view = _view(views, EvidenceSlot.MASKED_SUBJECT)

    assert view is not None
    x_min, y_min = int(view.crop_box.x_min), int(view.crop_box.y_min)
    assert np.array_equal(view.payload.pixels[0, 0], image.pixels[y_min, x_min])


# O contexto continua íntegro: o crop contextual não pode ser mascarado, ou
# perderia justamente o entorno que serve para desambiguar.
def test_contextual_crop_keeps_its_surroundings() -> None:
    """O crop contextual preserva os pixels do entorno."""
    image = _image()
    views = build_region_views((_region(),), image, _config())
    view = _view(views, EvidenceSlot.CONTEXTUAL_CROP)

    assert view is not None
    assert view.crop_box.width > _view(views, EvidenceSlot.TIGHT_CROP).crop_box.width
    # O canto do recorte contextual fica fora da máscara e longe do contorno,
    # de modo que ele tem de ser exatamente o pixel de origem.
    x_min, y_min = int(view.crop_box.x_min), int(view.crop_box.y_min)
    assert np.array_equal(view.payload.pixels[0, 0], image.pixels[y_min, x_min])


# Mas o sujeito precisa ser localizável dentro do contexto: o contorno da
# máscara é desenhado sobre o recorte, para que o modelo saiba de qual região
# está falando sem que o fundo seja destruído.
def test_contextual_crop_marks_the_subject_boundary() -> None:
    """O contorno do sujeito é desenhado sobre o crop contextual."""
    image = _image()
    views = build_region_views((_region(),), image, _config())
    view = _view(views, EvidenceSlot.CONTEXTUAL_CROP)

    assert view is not None
    x_min, y_min = int(view.crop_box.x_min), int(view.crop_box.y_min)
    original = image.crop(
        x_min, y_min, int(view.crop_box.x_max), int(view.crop_box.y_max)
    ).pixels
    assert not np.array_equal(view.payload.pixels, original)
    # A diferença fica confinada à borda da máscara: o interior do sujeito e o
    # entorno distante permanecem intactos.
    changed = np.any(view.payload.pixels != original, axis=2)
    assert changed.sum() > 0
    assert not bool(changed[6, 6])


# A imagem de origem nunca é alterada por nenhuma view. É a mesma invariante
# que a #212 estabeleceu para o SAM, aplicada ao caminho de evidência.
def test_building_views_never_mutates_the_source_image() -> None:
    """Construir as views não altera nenhum pixel da imagem original."""
    image = _image()
    before = image.pixels.copy()

    build_region_views((_region(), _region(diagonal=True)), image, _config())

    assert np.array_equal(image.pixels, before)


# Desenha o contorno de uma máscara 6x8 pela API pública e devolve onde os
# pixels mudaram: fundo e sujeito são da mesma cor, então todo pixel alterado é
# contorno.
def _contour(mask_data: np.ndarray) -> np.ndarray:
    """Retorna a máscara dos pixels pintados como contorno em ``crop_payload``."""
    image = ImagePayload(np.full((6, 8, 3), 90, dtype=np.uint8), 8, 6)
    mask = Mask(mask_data, 8, 6)
    region = ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p-1",))
    view = crop_payload(image, region, BoundingBox(0, 0, 8, 6), emphasis=SubjectEmphasis.CONTOUR)
    changed: np.ndarray = (view.pixels != image.pixels).any(axis=2)
    return changed


# Regressão da #246: com np.roll circular, uma máscara que tocava duas bordas
# opostas do recorte tinha a primeira linha tratada como vizinha da última, e
# o contorno sumia justamente em paredes, colunas e corrimãos.
def test_contour_is_drawn_on_crop_edges_for_a_mask_spanning_the_crop() -> None:
    """Uma máscara que atravessa o recorte tem contorno nas duas bordas tocadas."""
    vertical = np.zeros((6, 8), dtype=np.bool_)
    vertical[:, 2:6] = True
    contour = _contour(vertical)
    assert contour[0, 2:6].all() and contour[-1, 2:6].all()
    assert not contour[1:-1, 3:5].any()

    horizontal = np.zeros((6, 8), dtype=np.bool_)
    horizontal[2:5, :] = True
    contour = _contour(horizontal)
    assert contour[2:5, 0].all() and contour[2:5, -1].all()


# O caso que já funcionava continua igual: sem tocar as bordas, contorno
# fechado de um pixel; tocando uma borda só, contorno também nela.
def test_contour_is_unchanged_away_from_edges_and_closed_on_a_single_edge() -> None:
    """Máscara interna produz o anel de sempre; máscara numa borda fecha o contorno ali."""
    inner = np.zeros((6, 8), dtype=np.bool_)
    inner[1:5, 2:6] = True
    expected = inner.copy()
    expected[2:4, 3:5] = False
    assert np.array_equal(_contour(inner), expected)

    top = np.zeros((6, 8), dtype=np.bool_)
    top[0:3, 2:6] = True
    contour = _contour(top)
    assert contour[0, 2:6].all()
    assert not contour[1, 3:5].any()
