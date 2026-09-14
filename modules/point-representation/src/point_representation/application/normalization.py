"""Normalização reversível de coordenadas.

Issue: #7.

A escala usa a maior magnitude absoluta das coordenadas já centralizadas
(norma Chebyshev), não o desvio padrão: o objetivo é garantir que toda
coordenada normalizada caiba em ``[-1, 1]`` por eixo, o que um backbone de
grade/voxel (#20) precisa para discretização estável, mais do que uma
distribuição unitária no sentido estatístico.
"""

from __future__ import annotations

import numpy as np

from point_representation.config import NormalizationConfig
from point_representation.domain.normalization import NormalizationTransform
from point_representation.domain.point_cloud import PointCloud

#: Abaixo desta magnitude a nuvem é tratada como sem extensão (um único ponto
#: ou pontos duplicados) — escalar por um valor tão pequeno explodiria a
#: normalização em vez de deixá-la degenerar de forma segura para a
#: identidade.
_DEGENERATE_SCALE_EPSILON = 1e-12


# Centraliza e/ou escala as coordenadas de ``cloud`` segundo ``config``,
# preservando os canais auxiliares intactos. É o único ponto que decide a
# métrica de escala e o fallback degenerado, para que a inversão
# (:func:`denormalize_coordinates`) sempre tenha os parâmetros exatos usados
# aqui.
def normalize_coordinates(cloud: PointCloud, config: NormalizationConfig) -> tuple[PointCloud, NormalizationTransform]:
    """Normaliza as coordenadas de ``cloud`` segundo ``config``.

    Argumentos:
        cloud: a nuvem de pontos de origem.
        config: quais componentes da normalização (centralização, escala)
            aplicar.
    Retorna:
        a nuvem com coordenadas normalizadas (canais inalterados) e a
        transformação necessária para reverter a normalização.
    """
    if config.center and len(cloud) > 0:
        centroid = cloud.coordinates.mean(axis=0)
    else:
        centroid = np.zeros(3, dtype=np.float64)
    centered = cloud.coordinates - centroid

    if config.scale and len(cloud) > 0:
        magnitude = float(np.abs(centered).max())
        scale = magnitude if magnitude > _DEGENERATE_SCALE_EPSILON else 1.0
    else:
        scale = 1.0

    normalized_coordinates = centered / scale
    transform = NormalizationTransform(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        scale=scale,
        centered=config.center,
        scaled=config.scale,
    )
    normalized_cloud = PointCloud(normalized_coordinates, cloud.frame_id, channels=cloud.channels)
    return normalized_cloud, transform


# Reverte uma normalização a partir do ``transform`` que a produziu. Existe
# separado de ``normalize_coordinates`` porque a reversão é aplicada tipicamente
# a uma saída do encoder (posições preditas), não a uma ``PointCloud`` inteira.
def denormalize_coordinates(coordinates: np.ndarray, transform: NormalizationTransform) -> np.ndarray:
    """Reverte a normalização descrita por ``transform`` sobre ``coordinates``.

    Argumentos:
        coordinates: coordenadas normalizadas, forma ``(N, 3)``.
        transform: os parâmetros produzidos por :func:`normalize_coordinates`.
    Retorna:
        as coordenadas no frame original, dentro da tolerância numérica.
    """
    restored = np.asarray(coordinates, dtype=np.float64) * transform.scale
    return restored + np.asarray(transform.centroid, dtype=np.float64)
