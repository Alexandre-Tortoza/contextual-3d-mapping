"""Contract de entrada canônico de nuvem de pontos.

Issue: #1 (contracts públicos), #3 (invariantes estruturais validados aqui).

``PointCloud`` é a representação de entrada compartilhada por treino e
inferência: coordenadas obrigatórias mais canais auxiliares por ponto
opcionais (cor, intensidade, normais, ou qualquer outro canal que um
produtor decida registrar pelo nome). Ela nunca aponta para um tensor ou
framework específico — apenas ``numpy``, para que backbones concretos fiquem
atrás de adapters (ver ``ports``, issue #19).

Uma nuvem com zero pontos é estruturalmente válida aqui (formas e canais
ainda fazem sentido para N=0); a decisão de aceitar ou rejeitar uma nuvem
vazia é política de fronteira, não invariante estrutural, e vive em
``domain/validation.py`` (#3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from contextual_mapping_contracts import FrameId


# Copia e valida um array de canal por ponto, garantindo que ele acompanha a
# contagem de pontos declarada e não contém valores não-finitos. Compartilhado
# entre coordenadas e canais auxiliares para que as duas checagens nunca
# divirjam.
def _per_point_array(value: np.ndarray, *, field_name: str, point_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != point_count:
        raise ValueError(
            f"{field_name} must have shape (point_count, channel_width) with "
            f"point_count={point_count}, got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    array = array.copy()
    array.setflags(write=False)
    return array


# Representa a nuvem de pontos de entrada canônica do módulo, com canais
# auxiliares opcionais registrados por nome. Existe como o único contract de
# entrada que treino, pré-processamento e inferência compartilham.
@dataclass(frozen=True)
class PointCloud:
    """Nuvem de pontos canônica: coordenadas obrigatórias mais canais opcionais por nome.

    Argumentos:
        coordinates: array ``(N, 3)`` de coordenadas XYZ finitas.
        frame_id: frame espacial em que ``coordinates`` está expresso.
        channels: canais auxiliares por ponto, cada um ``(N, largura)``,
            indexados pelo nome do canal (ex: ``"color"``, ``"intensity"``,
            ``"normals"``). Vazio quando a nuvem só tem geometria.
    """

    coordinates: np.ndarray
    frame_id: FrameId
    channels: Mapping[str, np.ndarray] = field(default_factory=dict)

    # Valida a forma e a finitude das coordenadas e de todo canal declarado,
    # e congela os arrays para que a nuvem nunca seja mutada depois de criada.
    def __post_init__(self) -> None:
        """Valida forma, finitude e alinhamento de coordenadas e canais."""
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                f"PointCloud.coordinates must have shape (N, 3), got {coordinates.shape}."
            )
        if not np.isfinite(coordinates).all():
            raise ValueError("PointCloud.coordinates must contain only finite values.")
        coordinates = coordinates.copy()
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)

        point_count = len(coordinates)
        validated_channels = {
            name: _per_point_array(channel, field_name=f"PointCloud.channels[{name!r}]", point_count=point_count)
            for name, channel in self.channels.items()
        }
        object.__setattr__(self, "channels", validated_channels)

    # Responde quantos pontos a nuvem carrega, usado por validação e batching
    # sem que o chamador precise conhecer o layout interno dos arrays.
    def __len__(self) -> int:
        """Retorna a contagem de pontos da nuvem."""
        return len(self.coordinates)
