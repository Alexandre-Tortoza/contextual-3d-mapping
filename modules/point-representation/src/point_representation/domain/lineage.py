"""Rastreamento de proveniência de índice através do pré-processamento.

Issue: #8.

Cada transform (crop #5, voxel downsampling #6) já devolve, junto com a
nuvem transformada, os índices de origem de cada ponto de saída —
um-para-um no crop, um-para-muitos no downsampling. ``PointLineage`` é o
contract comum que representa esse mapeamento e permite compô-lo através de
uma sequência de transforms, para que o ponto final de um pipeline de
pré-processamento sempre resolva de volta ao(s) ponto(s) da amostra
original, não importa quantos estágios o produziram.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


# Representa, para cada ponto de saída de um estágio, os índices (no espaço
# de entrada daquele estágio) que contribuíram para ele. Um grupo com mais de
# um índice representa proveniência muitos-para-um (ex: voxel downsampling);
# um grupo de um único índice representa um-para-um (ex: crop).
@dataclass(frozen=True)
class PointLineage:
    """Mapeamento de cada ponto de saída para os índices de entrada que o originaram.

    Argumentos:
        groups: para cada ponto de saída, na ordem, a tupla ordenada e sem
            duplicatas dos índices de entrada que contribuíram para ele.
    """

    groups: tuple[tuple[int, ...], ...]

    # Rejeita um grupo vazio, com índice negativo, ou com duplicata — nenhum
    # desses descreve uma proveniência válida.
    def __post_init__(self) -> None:
        """Valida que todo grupo é não-vazio, não-negativo e sem duplicatas."""
        for position, group in enumerate(self.groups):
            if not group:
                raise ValueError(f"PointLineage.groups[{position}] must not be empty.")
            if any(index < 0 for index in group):
                raise ValueError(f"PointLineage.groups[{position}] must not contain negative indices.")
            if len(set(group)) != len(group):
                raise ValueError(f"PointLineage.groups[{position}] must not contain duplicate indices.")

    # Quantos pontos de saída este mapeamento descreve, usado para validar
    # composição e batching sem que o chamador precise chamar ``len(groups)``.
    @property
    def point_count(self) -> int:
        """Retorna quantos pontos de saída este mapeamento cobre."""
        return len(self.groups)

    # Resolve um único ponto de saída para os índices de entrada que o
    # originaram — a utilidade pública que a #8 exige para consumidores que
    # só precisam da proveniência de um ponto específico.
    def resolve(self, output_index: int) -> tuple[int, ...]:
        """Retorna os índices de entrada que originaram ``output_index``."""
        return self.groups[output_index]

    # Constrói um lineage um-para-um a partir de um array de índices, como o
    # devolvido por ``crop_points`` (#5).
    @staticmethod
    def from_indices(indices: np.ndarray) -> PointLineage:
        """Constrói um ``PointLineage`` um-para-um a partir de ``indices``."""
        return PointLineage(tuple((int(index),) for index in indices))

    # Constrói um lineage possivelmente muitos-para-um a partir de grupos de
    # índices, como o devolvido por ``voxel_downsample`` (#6).
    @staticmethod
    def from_groups(groups: Sequence[np.ndarray]) -> PointLineage:
        """Constrói um ``PointLineage`` a partir de grupos de índices de entrada."""
        return PointLineage(tuple(tuple(sorted(int(index) for index in group)) for group in groups))


# Compõe dois lineages consecutivos em um único mapeamento da entrada do
# primeiro estágio para a saída do segundo. Existe para que uma sequência de
# transforms (crop seguido de downsampling, por exemplo) nunca precise que o
# chamador reimplemente a junção dos grupos manualmente.
def compose_lineage(earlier: PointLineage, later: PointLineage) -> PointLineage:
    """Compõe ``earlier`` (entrada -> intermediário) com ``later`` (intermediário -> saída).

    Argumentos:
        earlier: lineage do primeiro estágio aplicado.
        later: lineage do estágio seguinte, cujos índices de entrada devem
            referenciar pontos de saída de ``earlier``.
    Retorna:
        o lineage composto, da entrada original de ``earlier`` até a saída
        final de ``later``.
    Levanta:
        ValueError: se algum índice de ``later`` estiver fora do intervalo
            de pontos de saída de ``earlier``.
    """
    composed: list[tuple[int, ...]] = []
    for position, group in enumerate(later.groups):
        if any(index >= earlier.point_count for index in group):
            raise ValueError(
                f"compose_lineage: later.groups[{position}] references an index outside "
                f"earlier's point_count={earlier.point_count}."
            )
        merged = sorted({source for index in group for source in earlier.groups[index]})
        composed.append(tuple(merged))
    return PointLineage(tuple(composed))
