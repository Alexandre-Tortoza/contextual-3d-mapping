"""Recorte espacial determinístico de uma nuvem de pontos.

Issue: #5.

O recorte nunca reordena os pontos retidos (preserva a ordem de origem) e
sempre devolve os índices de origem junto com a nuvem recortada, para que um
consumidor (#8) componha essa proveniência com transforms subsequentes sem
precisar recalcular quais pontos sobreviveram.
"""

from __future__ import annotations

import numpy as np

from point_representation.config import CropConfig
from point_representation.domain.point_cloud import PointCloud


# Recorta ``cloud`` para dentro dos limites configurados e, opcionalmente,
# subamostra deterministicamente o resultado. É o único ponto que decide
# quais pontos sobrevivem a um recorte, para que a política (limites,
# subsample determinístico) nunca fique duplicada por um chamador.
def crop_points(cloud: PointCloud, config: CropConfig) -> tuple[PointCloud, np.ndarray]:
    """Recorta ``cloud`` segundo ``config`` e retorna a nuvem recortada e seus índices de origem.

    Argumentos:
        cloud: a nuvem de pontos de origem.
        config: limites do recorte e subsample determinístico opcional.
    Retorna:
        a nuvem recortada (coordenadas e canais alinhados) e um array
        ``(M,)`` com o índice, em ``cloud``, de cada ponto retido, na mesma
        ordem em que aparece na nuvem recortada.
    """
    lo = np.asarray(config.bounds.min_m, dtype=np.float64)
    hi = np.asarray(config.bounds.max_m, dtype=np.float64)
    inside = np.all((cloud.coordinates >= lo) & (cloud.coordinates <= hi), axis=1)
    indices = np.flatnonzero(inside)

    if config.max_points is not None and len(indices) > config.max_points:
        rng = np.random.default_rng(config.seed)
        chosen = rng.choice(indices, size=config.max_points, replace=False)
        indices = np.sort(chosen)

    cropped = PointCloud(
        cloud.coordinates[indices],
        cloud.frame_id,
        channels={name: values[indices] for name, values in cloud.channels.items()},
    )
    return cropped, indices
