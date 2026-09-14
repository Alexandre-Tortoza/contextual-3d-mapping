"""Montagem da entrada do encoder a partir dos canais configurados.

Issue: #4.

A entrada do encoder separa coordenadas (sempre geometria pura, nunca
aprendida) das features selecionadas (canais auxiliares concatenados na
ordem configurada). Essa separação existe porque um backbone concreto (#20)
usa coordenadas para discretização espacial e features para o canal
aprendido — confundir os dois quebraria qualquer backbone que dependa de
coordenadas cruas para indexação espacial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from point_representation.config import FeatureSelectionConfig
from point_representation.domain.errors import FeatureSelectionError
from point_representation.domain.point_cloud import PointCloud


# Representa a entrada canônica de um encoder concreto: coordenadas
# preservadas separadamente e as features aprendidas já concatenadas na
# ordem configurada. Existe para que um adapter de backbone (#20) nunca
# precise conhecer ``PointCloud.channels`` diretamente.
@dataclass(frozen=True)
class EncoderInput:
    """Entrada montada para um encoder, com coordenadas e features separadas.

    Argumentos:
        coordinates: array ``(N, 3)`` de coordenadas, nunca parte do canal
            aprendido.
        features: array ``(N, feature_dimension)`` das features selecionadas,
            concatenadas na ordem de ``selected_channels``; ``(N, 0)`` quando
            nenhum canal foi selecionado.
        feature_dimension: largura total de ``features``.
        selected_channels: os canais que compõem ``features``, na ordem de
            concatenação.
    """

    coordinates: np.ndarray
    features: np.ndarray
    feature_dimension: int
    selected_channels: tuple[str, ...]


# Monta a ``EncoderInput`` de uma nuvem segundo a configuração de seleção de
# canais. É o único ponto que resolve nome de canal -> posição na
# concatenação, para que essa decisão nunca seja duplicada por um adapter.
def select_features(cloud: PointCloud, config: FeatureSelectionConfig) -> EncoderInput:
    """Seleciona e concatena os canais configurados de ``cloud`` em uma ``EncoderInput``.

    Argumentos:
        cloud: a nuvem de pontos de origem.
        config: quais canais incluir, e em que ordem.
    Retorna:
        a entrada montada do encoder.
    Levanta:
        FeatureSelectionError: se um canal configurado não existir em ``cloud``.
    """
    point_count = len(cloud)
    blocks: list[np.ndarray] = []
    for name in config.channels:
        if name not in cloud.channels:
            raise FeatureSelectionError(
                f"Requested feature channel {name!r} is not available in this PointCloud "
                f"(available: {sorted(cloud.channels)})."
            )
        blocks.append(cloud.channels[name])

    features = np.concatenate(blocks, axis=1) if blocks else np.zeros((point_count, 0), dtype=np.float64)
    features.setflags(write=False)
    return EncoderInput(
        coordinates=cloud.coordinates,
        features=features,
        feature_dimension=features.shape[1],
        selected_channels=config.channels,
    )
