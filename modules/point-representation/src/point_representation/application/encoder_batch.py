"""Colagem de entradas de encoder por amostra em um `EncoderInputBatch`.

Issue: #19.

Compõe :func:`~point_representation.application.feature_selection.select_features`
(#4) com o batching de tamanho variável (#9): cada amostra é selecionada
independentemente e depois concatenada, para que um `PointEncoder` receba um
único lote em vez de uma amostra por chamada.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from point_representation.application.feature_selection import EncoderInput
from point_representation.domain.encoder_io import EncoderInputBatch
from point_representation.domain.errors import PointCloudValidationError


# Concatena ``inputs`` (já selecionados por amostra) em um `EncoderInputBatch`.
# É o único ponto que decide como entradas de encoder por amostra viram um
# lote, para que nenhum chamador precise reimplementar a concatenação ou a
# checagem de canais/dimensão compatíveis.
def collate_encoder_inputs(inputs: Sequence[EncoderInput]) -> EncoderInputBatch:
    """Colige ``inputs`` (uma por amostra) em um `EncoderInputBatch`.

    Argumentos:
        inputs: as entradas de encoder por amostra, todas com os mesmos
            ``selected_channels``/``feature_dimension``.
    Retorna:
        o lote combinado.
    Levanta:
        PointCloudValidationError: se ``inputs`` estiver vazio ou as amostras
            declararem canais selecionados diferentes entre si.
    """
    if not inputs:
        raise PointCloudValidationError("collate_encoder_inputs requires at least one EncoderInput.")

    selected_channels = inputs[0].selected_channels
    for position, sample in enumerate(inputs[1:], start=1):
        if sample.selected_channels != selected_channels:
            raise PointCloudValidationError(
                f"collate_encoder_inputs: sample {position} selected channels "
                f"{sample.selected_channels}, expected {selected_channels} (from sample 0)."
            )

    coordinates = np.concatenate([sample.coordinates for sample in inputs], axis=0)
    features = np.concatenate([sample.features for sample in inputs], axis=0)
    point_counts = [len(sample.coordinates) for sample in inputs]
    sample_offsets = tuple(int(value) for value in np.concatenate([[0], np.cumsum(point_counts)]))

    return EncoderInputBatch(
        coordinates=coordinates,
        features=features,
        feature_dimension=inputs[0].feature_dimension,
        selected_channels=selected_channels,
        sample_offsets=sample_offsets,
    )
