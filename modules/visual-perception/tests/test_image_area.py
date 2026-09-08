"""Testes da geometria de área de imagem: lente válida e ego-veículo (#202).

O módulo precisa saber que parte do frame é evidência utilizável. Duas áreas o
limitam em ``corridor-02``: o círculo útil da lente fisheye (fora dele os
pixels são o corpo preto da objetiva) e a silhueta do próprio rig, que aparece
fixa na parte de baixo de todo frame.

A geometria é declarada, versionada e rasterizada de forma determinística. O
módulo nunca a infere por limiar de cor em runtime, e nunca altera os pixels de
origem para aplicá-la.
"""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry


# Um círculo declarado rasteriza como disco fechado: o pixel exatamente sobre o
# raio pertence à área, e o de fora não.
def test_circle_rasterizes_as_a_closed_disc() -> None:
    geometry = ImageAreaGeometry(circle=CircleArea(center_x=5.0, center_y=5.0, radius=3.0))

    mask = geometry.rasterize(11, 11)

    assert bool(mask.data[5, 5])
    assert bool(mask.data[5, 8])
    assert not bool(mask.data[5, 9])


# A rasterização é determinística: a mesma geometria produz a mesma máscara.
def test_rasterization_is_deterministic() -> None:
    geometry = ImageAreaGeometry(circle=CircleArea(4.0, 4.0, 2.5))

    assert np.array_equal(geometry.rasterize(9, 9).data, geometry.rasterize(9, 9).data)


# Um polígono rasteriza pela regra de cruzamento: pontos internos entram,
# externos ficam de fora. É o formato usado para a silhueta do rig.
def test_polygon_rasterizes_its_interior() -> None:
    geometry = ImageAreaGeometry(polygons=(((1.0, 1.0), (7.0, 1.0), (7.0, 5.0), (1.0, 5.0)),))

    mask = geometry.rasterize(10, 10)

    assert bool(mask.data[3, 4])
    assert not bool(mask.data[7, 4])
    assert not bool(mask.data[3, 8])


# Círculo e polígonos compõem por união: a área é tudo que qualquer uma das
# formas declara.
def test_circle_and_polygons_compose_as_a_union() -> None:
    geometry = ImageAreaGeometry(
        circle=CircleArea(2.0, 2.0, 1.5),
        polygons=(((6.0, 6.0), (9.0, 6.0), (9.0, 9.0), (6.0, 9.0)),),
    )

    mask = geometry.rasterize(10, 10)

    assert bool(mask.data[2, 2])
    assert bool(mask.data[7, 7])
    assert not bool(mask.data[2, 7])


# Uma geometria sem forma nenhuma não descreve área alguma. Rasterizá-la
# devolveria uma máscara vazia que, usada como área válida, rejeitaria todo o
# frame em silêncio.
def test_geometry_without_any_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ImageAreaGeometry()


# Raio não positivo não é um círculo.
def test_non_positive_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        CircleArea(1.0, 1.0, 0.0)


# Um polígono precisa de ao menos três vértices para ter interior.
def test_polygon_with_fewer_than_three_vertices_is_rejected() -> None:
    with pytest.raises(ValueError, match="three"):
        ImageAreaGeometry(polygons=(((0.0, 0.0), (1.0, 1.0)),))


# A resolução pedida precisa ser positiva; caso contrário não há máscara a
# construir e o erro deve aparecer aqui, não como shape inválido lá na frente.
@pytest.mark.parametrize(("width", "height"), [(0, 4), (4, 0), (-1, 4)])
def test_non_positive_resolution_is_rejected(width: int, height: int) -> None:
    geometry = ImageAreaGeometry(circle=CircleArea(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        geometry.rasterize(width, height)


# A geometria medida em corridor-02: o círculo útil cobre a maior parte do
# frame 640x480 e deixa de fora os quatro cantos, que é exatamente onde a
# vinheta da lente aparece.
def test_the_measured_corridor_02_circle_excludes_the_four_corners() -> None:
    """O círculo medido para corridor-02 exclui os cantos e mantém o centro."""
    geometry = ImageAreaGeometry(circle=CircleArea(326.0, 245.0, 324.0))

    mask = geometry.rasterize(640, 480)

    assert bool(mask.data[240, 320])
    for row, column in ((0, 0), (0, 639), (479, 0), (479, 639)):
        assert not bool(mask.data[row, column])
