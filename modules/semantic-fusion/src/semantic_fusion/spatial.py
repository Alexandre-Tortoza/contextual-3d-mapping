"""Medição de suporte espacial de um label contra a vizinhança geométrica."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .models import SpatialNeighbourhood

_NEIGHBOUR_OFFSETS = tuple(
    (x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
)


# Representa a entrada mínima da medição espacial: onde o ponto está e o que
# ele afirma ser. Existe para que a medição não dependa do formato do artifact
# nem de como o label foi escolhido.
@dataclass(frozen=True)
class LabelledPoint:
    """Ponto do mapa com o label primário que ele carrega.

    Argumentos:
        geometry_id: identidade estável do ponto no mapa.
        coordinates_m: coordenadas no frame do mapa, em metros.
        label: label primário atribuído ao ponto.
    """

    geometry_id: str
    coordinates_m: tuple[float, float, float]
    label: str


# Mede quanto a vizinhança geométrica concorda com o label de cada ponto.
#
# Existe porque um label pode estar geometricamente deslocado sem que nada na
# imagem o denuncie: um ponto atrás de uma parede que escapa do teste de
# oclusão recebe o label da superfície da frente e fica cercado de pontos que
# afirmam outra coisa. A concordância entre keyframes não pega esse caso quando
# todos os keyframes cometem o mesmo erro de projeção; a vizinhança 3D, sim.
#
# É chamada pelo mapping-runtime depois da fusão multi-view, e o resultado é
# publicado ao lado do claim — a medição não reescreve nenhum label.
def measure_spatial_support(
    points: Iterable[LabelledPoint],
    neighbourhood: SpatialNeighbourhood | None = None,
) -> dict[str, float | None]:
    """Mede a fração de vizinhos rotulados que concorda com cada ponto.

    Argumentos:
        points: pontos do mapa que carregam um label primário.
        neighbourhood: parâmetros da grade; usa os defaults quando ausente.
    Retorna:
        suporte por ``geometry_id``, ou ``None`` quando a vizinhança tem menos
        vizinhos rotulados que o mínimo configurado. ``None`` significa
        indefinido, e não ausência de suporte: um ponto isolado não foi
        contradito por ninguém e não pode ser tratado como se tivesse sido.
    Levanta:
        ValueError: se dois pontos declararem a mesma identidade.
    """
    resolved = neighbourhood or SpatialNeighbourhood()
    edge = float(resolved.voxel_edge_m)
    material = list(points)
    cells: dict[str, tuple[int, int, int]] = {}
    labels_by_cell: defaultdict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    for point in material:
        if point.geometry_id in cells:
            raise ValueError("points must have unique geometry ids.")
        cell = tuple(int(value // edge) for value in point.coordinates_m)
        cells[point.geometry_id] = cell  # type: ignore[assignment]
        labels_by_cell[cell][point.label] += 1  # type: ignore[index]
    support: dict[str, float | None] = {}
    for point in material:
        cell_x, cell_y, cell_z = cells[point.geometry_id]
        neighbours: Counter[str] = Counter()
        for offset_x, offset_y, offset_z in _NEIGHBOUR_OFFSETS:
            found = labels_by_cell.get((cell_x + offset_x, cell_y + offset_y, cell_z + offset_z))
            if found is not None:
                neighbours.update(found)
        # O próprio ponto está contado na sua célula e não é evidência sobre si.
        neighbours[point.label] -= 1
        total = sum(neighbours.values())
        if total < resolved.minimum_neighbours:
            support[point.geometry_id] = None
            continue
        support[point.geometry_id] = neighbours[point.label] / total
    return support
