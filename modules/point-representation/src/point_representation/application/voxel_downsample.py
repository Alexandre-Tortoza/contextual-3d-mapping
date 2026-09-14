"""Downsampling por voxel com proveniência de índice de origem.

Issue: #6.

Política documentada: cada voxel ocupado vira um ponto de saída no centróide
(média) das coordenadas — e, igualmente, na média de cada canal auxiliar —
dos pontos que caíram nele. A saída é ordenada pela chave de voxel (não pela
ordem de chegada dos pontos de entrada), para que o resultado seja
determinístico independente de como a nuvem de entrada estava ordenada, sem
depender de nenhum dataset específico.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from point_representation.config import VoxelDownsampleConfig
from point_representation.domain.point_cloud import PointCloud


# Reduz a densidade de ``cloud`` agrupando pontos por célula de voxel e
# publicando um representante por célula ocupada. É o único ponto que decide
# a política de agregação (centróide) e a proveniência por grupo, para que
# nenhum chamador precise reimplementar o agrupamento.
def voxel_downsample(cloud: PointCloud, config: VoxelDownsampleConfig) -> tuple[PointCloud, tuple[np.ndarray, ...]]:
    """Reduz ``cloud`` para um representante por voxel ocupado.

    Argumentos:
        cloud: a nuvem de pontos de origem.
        config: aresta da célula de voxel usada para agrupar pontos.
    Retorna:
        a nuvem reduzida (um ponto por voxel ocupado, ordenada pela chave de
        voxel) e, para cada ponto de saída, o array de índices em ``cloud``
        dos pontos que contribuíram para aquele voxel.
    """
    voxel_size = config.voxel_size_m
    voxel_keys = np.floor(cloud.coordinates / voxel_size).astype(np.int64)

    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for point_index, key in enumerate(map(tuple, voxel_keys)):
        groups[key].append(point_index)

    ordered_keys = sorted(groups)
    source_index_groups = tuple(np.asarray(groups[key], dtype=np.int64) for key in ordered_keys)

    coordinates = np.stack(
        [cloud.coordinates[indices].mean(axis=0) for indices in source_index_groups]
    ) if source_index_groups else np.zeros((0, 3), dtype=np.float64)
    channels = {
        name: np.stack([values[indices].mean(axis=0) for indices in source_index_groups])
        if source_index_groups
        else np.zeros((0, values.shape[1]), dtype=np.float64)
        for name, values in cloud.channels.items()
    }

    downsampled = PointCloud(coordinates, cloud.frame_id, channels=channels)
    return downsampled, source_index_groups
