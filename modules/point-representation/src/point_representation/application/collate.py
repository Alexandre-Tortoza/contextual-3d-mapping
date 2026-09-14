"""Colagem (collate) de nuvens de pontos de tamanho variável em um lote.

Issue: #9.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from point_representation.domain.batch import PointBatch
from point_representation.domain.errors import PointCloudValidationError
from point_representation.domain.lineage import PointLineage
from point_representation.domain.point_cloud import PointCloud


# Concatena ``clouds`` em um único ``PointBatch``, preservando fronteiras de
# amostra e lineage opcional. É o único ponto que decide como amostras
# heterogêneas viram um lote, para que nenhum chamador precise reimplementar
# a concatenação ou a checagem de canais compatíveis.
def collate_point_clouds(
    clouds: Sequence[PointCloud], lineages: Sequence[PointLineage | None] | None = None
) -> PointBatch:
    """Colige ``clouds`` (e, opcionalmente, seus lineages) em um ``PointBatch``.

    Argumentos:
        clouds: as nuvens a combinar, em ordem; podem ter contagens de ponto
            diferentes, mas devem declarar os mesmos canais.
        lineages: lineage de pré-processamento por amostra, na mesma ordem
            de ``clouds``, ou ``None`` quando nenhuma é rastreada.
    Retorna:
        o lote combinado.
    Levanta:
        PointCloudValidationError: se ``clouds`` estiver vazio ou as amostras
            declararem conjuntos de canais diferentes entre si.
    """
    if not clouds:
        raise PointCloudValidationError("collate_point_clouds requires at least one PointCloud.")

    channel_names = set(clouds[0].channels)
    for position, cloud in enumerate(clouds[1:], start=1):
        if set(cloud.channels) != channel_names:
            raise PointCloudValidationError(
                f"collate_point_clouds: sample {position} declares channels "
                f"{sorted(cloud.channels)}, expected {sorted(channel_names)} (from sample 0)."
            )

    coordinates = np.concatenate([cloud.coordinates for cloud in clouds], axis=0)
    channels = {
        name: np.concatenate([cloud.channels[name] for cloud in clouds], axis=0) for name in channel_names
    }
    point_counts = [len(cloud) for cloud in clouds]
    sample_offsets = tuple(int(value) for value in np.concatenate([[0], np.cumsum(point_counts)]))

    return PointBatch(
        coordinates=coordinates,
        channels=channels,
        sample_offsets=sample_offsets,
        lineages=tuple(lineages) if lineages is not None else (),
    )
