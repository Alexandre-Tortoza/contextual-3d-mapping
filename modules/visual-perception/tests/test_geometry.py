"""Testes de invariantes de região, mask e coordenadas de imagem (#155)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.domain.geometry import BoundingBox, CoordinateTransform, Mask


# Constrói uma Mask a partir de uma matriz booleana literal, inferindo width/height do
# próprio array; helper usado por todos os testes de geometria para evitar repetição.
def _mask(data: list[list[bool]]) -> Mask:
    array = np.array(data, dtype=np.bool_)
    return Mask(array, image_width=array.shape[1], image_height=array.shape[0])


# Protege o invariante de que uma BoundingBox precisa ter área positiva.
def test_bounding_box_rejects_non_positive_area() -> None:
    with pytest.raises(ValueError):
        BoundingBox(0, 0, 0, 5)


# Confirma que clipped_to recorta a box corretamente para dentro dos limites da imagem,
# mesmo quando a box original ultrapassa esses limites.
def test_bounding_box_clips_to_image_bounds() -> None:
    box = BoundingBox(-5, -5, 15, 15).clipped_to(width=10, height=10)
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0, 0, 10, 10)


# Protege o invariante de que uma Mask deve ser booleana — outros dtypes são rejeitados.
def test_mask_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError):
        Mask(np.zeros((2, 2), dtype=np.int32), image_width=2, image_height=2)


# Protege o invariante de que o shape do array deve bater com image_width/image_height
# declarados.
def test_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        Mask(np.zeros((2, 2), dtype=np.bool_), image_width=3, image_height=2)


# Garante que pedir a bounding box de uma mask vazia falha explicitamente, em vez de
# retornar uma box degenerada sem sentido.
def test_empty_mask_bounding_box_raises() -> None:
    mask = _mask([[False, False], [False, False]])
    assert mask.is_empty
    with pytest.raises(ValueError):
        mask.bounding_box()


# Verifica o caso de borda de um único pixel ativo no canto — confere que a bounding
# box respeita a convenção half-open (max exclusive) mesmo no limite da imagem.
def test_mask_bounding_box_for_edge_pixel() -> None:
    mask = _mask([[True, False], [False, False]])
    box = mask.bounding_box()
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0.0, 0.0, 1.0, 1.0)


# Verifica os cálculos de IoU e containment_ratio contra um caso conhecido à mão,
# incluindo a assimetria esperada de containment_ratio (a->b difere de b->a).
def test_mask_iou_and_containment() -> None:
    a = _mask([[True, True], [False, False]])
    b = _mask([[True, False], [False, False]])
    assert a.iou(b) == pytest.approx(0.5)
    assert a.containment_ratio(b) == pytest.approx(1.0)
    assert b.containment_ratio(a) == pytest.approx(0.5)


# Confirma que a transform identidade não altera a geometria — caso base antes de
# testar transforms com escala/offset reais.
def test_identity_transform_is_a_no_op() -> None:
    transform = CoordinateTransform.identity()
    box = BoundingBox(1, 2, 3, 4)
    assert transform.box_to_global(box) == box


# Garante que box_to_global seguido de box_to_local recupera a box original — protege
# o invariante de que CoordinateTransform é invertível em ambas as direções.
def test_transform_box_round_trip() -> None:
    transform = CoordinateTransform(scale_x=2.0, scale_y=2.0, offset_x=10.0, offset_y=5.0)
    local_box = BoundingBox(1, 1, 4, 4)
    global_box = transform.box_to_global(local_box)
    recovered = transform.box_to_local(global_box)
    assert recovered.x_min == pytest.approx(local_box.x_min)
    assert recovered.y_min == pytest.approx(local_box.y_min)
    assert recovered.x_max == pytest.approx(local_box.x_max)
    assert recovered.y_max == pytest.approx(local_box.y_max)


# Verifica que remapear uma mask de tile para coordenadas globais, com um offset
# inteiro, posiciona os pixels ativos exatamente onde esperado (sem erro de arredondamento).
def test_tile_mask_remaps_exactly_for_integer_offset() -> None:
    local_mask = _mask([[True, False], [False, True]])
    transform = CoordinateTransform(scale_x=1.0, scale_y=1.0, offset_x=2.0, offset_y=3.0)
    global_mask = transform.mask_to_global(local_mask, global_width=6, global_height=6)
    expected = np.zeros((6, 6), dtype=np.bool_)
    expected[3, 2] = True
    expected[4, 3] = True
    assert np.array_equal(global_mask.data, expected)
