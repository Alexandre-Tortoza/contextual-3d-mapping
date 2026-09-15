"""Contract da menor amostra de cena consumida pelo treino de representação.

Issue: #2.

``PointTrainingSample`` envolve uma :class:`PointCloud` com o que só faz
sentido durante treino: features de professor alinhadas a ponto (produzidas
por outro módulo, ex: visual-perception via distilação 2D->3D, #38) e índices
de correspondência que rastreiam cada ponto de volta à amostra de origem
(usados por crop/downsampling, #5-#8). Nenhum dos dois participa da
inferência pública (#48); por isso vivem aqui, e não em ``PointCloud``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from point_representation.domain.point_cloud import PointCloud


# Copia e valida um array por ponto opcional, alinhado à contagem de pontos
# da nuvem. Compartilhado entre teacher_features e correspondence_indices
# para que as duas checagens de alinhamento nunca divirjam.
def _aligned_array(value: np.ndarray, *, field_name: str, point_count: int, dtype: type) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim not in (1, 2) or array.shape[0] != point_count:
        raise ValueError(
            f"{field_name} must be per-point (leading dimension {point_count}), got shape {array.shape}."
        )
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    array = array.copy()
    array.setflags(write=False)
    return array


# Representa a menor amostra de cena que o treino de representação consome:
# uma nuvem de pontos mais a evidência de professor e a proveniência de
# correspondência que só existem durante treino.
@dataclass(frozen=True)
class PointTrainingSample:
    """Uma nuvem de pontos de treino, com professor e correspondência opcionais.

    Argumentos:
        point_cloud: a nuvem de entrada (geometria mais canais auxiliares).
        teacher_features: features alinhadas a ponto de um professor externo
            (ex: dense visual features projetadas, #38), forma
            ``(N, feature_dim)``, ou ``None`` quando não disponíveis.
        correspondence_indices: índice de cada ponto na amostra de origem
            (antes de qualquer crop/downsampling), forma ``(N,)``, ou ``None``
            quando a amostra não deriva de outra.
    """

    point_cloud: PointCloud
    teacher_features: np.ndarray | None = None
    correspondence_indices: np.ndarray | None = None

    # Valida que os dois campos opcionais, quando presentes, acompanham a
    # contagem de pontos da nuvem — a #2 exige que uma amostra nunca carregue
    # evidência de professor ou correspondência desalinhada em silêncio.
    def __post_init__(self) -> None:
        """Valida o alinhamento de ``teacher_features``/``correspondence_indices`` com a nuvem."""
        point_count = len(self.point_cloud)
        if self.teacher_features is not None:
            object.__setattr__(
                self,
                "teacher_features",
                _aligned_array(
                    self.teacher_features,
                    field_name="PointTrainingSample.teacher_features",
                    point_count=point_count,
                    dtype=np.float64,
                ),
            )
        if self.correspondence_indices is not None:
            indices = _aligned_array(
                self.correspondence_indices,
                field_name="PointTrainingSample.correspondence_indices",
                point_count=point_count,
                dtype=np.int64,
            )
            if indices.ndim != 1:
                raise ValueError("PointTrainingSample.correspondence_indices must be one-dimensional.")
            if (indices < 0).any():
                raise ValueError("PointTrainingSample.correspondence_indices must be non-negative.")
            object.__setattr__(self, "correspondence_indices", indices)
