"""Contract de lote de nuvens de pontos de tamanho variável.

Issue: #9.

``PointBatch`` concatena várias :class:`~point_representation.domain.point_cloud.PointCloud`
de contagens de ponto diferentes em uma única representação, preservando a
fronteira de cada amostra via offsets estilo CSR — o mesmo esquema que
``BatchedPointEmbeddings`` já usa para a saída pública (#1), para que entrada
e saída de um encoder concordem sobre como um lote é particionado.

``frame_id`` não é preservado aqui: um lote combina amostras para um
forward de encoder, que opera em coordenadas já normalizadas por amostra
(#7); qual frame espacial cada amostra tinha antes do batching é proveniência
de treino, não parte deste contract de entrada do encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from point_representation.domain.csr import validate_csr_offsets
from point_representation.domain.lineage import PointLineage


# Representa um lote de pontos de amostras variáveis, concatenados com as
# fronteiras de amostra preservadas. Existe para que um encoder processe um
# lote inteiro em uma chamada sem perder de que amostra (e de que ponto
# dentro dela) cada linha do lote veio.
@dataclass(frozen=True)
class PointBatch:
    """Um lote de pontos concatenados, com fronteiras de amostra e lineage opcional.

    Argumentos:
        coordinates: array ``(total_points, 3)`` de todas as amostras
            concatenadas, na ordem de ``sample_offsets``.
        channels: canais auxiliares concatenados da mesma forma, alinhados a
            ``coordinates``.
        sample_offsets: offsets estilo CSR; a amostra ``i`` ocupa
            ``coordinates[sample_offsets[i]:sample_offsets[i+1]]``.
        lineages: lineage de pré-processamento por amostra (#8), na mesma
            ordem de ``sample_offsets``; tupla vazia quando nenhuma amostra
            carrega lineage.
    """

    coordinates: np.ndarray
    channels: Mapping[str, np.ndarray]
    sample_offsets: tuple[int, ...]
    lineages: tuple[PointLineage | None, ...] = ()

    # Valida que os offsets particionam ``coordinates``/``channels``
    # corretamente e que todo lineage presente cobre exatamente os pontos da
    # sua amostra — um lineage com contagem errada corromperia
    # silenciosamente a resolução de proveniência (#8).
    def __post_init__(self) -> None:
        """Valida offsets, alinhamento de canais e consistência de lineage."""
        total_points = len(self.coordinates)
        validate_csr_offsets(self.sample_offsets, total_points, field="PointBatch.sample_offsets")
        for name, values in self.channels.items():
            if len(values) != total_points:
                raise ValueError(
                    f"PointBatch.channels[{name!r}] must have {total_points} rows, got {len(values)}."
                )
        if self.lineages:
            sample_count = len(self.sample_offsets) - 1
            if len(self.lineages) != sample_count:
                raise ValueError(
                    f"PointBatch.lineages must have one entry per sample ({sample_count}), "
                    f"got {len(self.lineages)}."
                )
            for index, lineage in enumerate(self.lineages):
                if lineage is None:
                    continue
                expected = self.sample_offsets[index + 1] - self.sample_offsets[index]
                if lineage.point_count != expected:
                    raise ValueError(
                        f"PointBatch.lineages[{index}] covers {lineage.point_count} points, "
                        f"but sample {index} has {expected} points."
                    )

    # Quantas amostras o lote contém, derivado dos offsets para que nunca
    # divirja de ``sample_offsets``.
    @property
    def sample_count(self) -> int:
        """Retorna quantas amostras compõem o lote."""
        return len(self.sample_offsets) - 1

    # Resolve um índice global do lote para a amostra e o índice local de
    # onde ele veio — a utilidade que a #9 exige para que um consumidor
    # nunca precise varrer ``sample_offsets`` manualmente.
    def resolve_source(self, batch_index: int) -> tuple[int, int]:
        """Retorna ``(sample_index, local_index)`` de origem para ``batch_index``.

        Argumentos:
            batch_index: posição do ponto na concatenação do lote.
        Retorna:
            o índice da amostra de origem e o índice do ponto dentro dela.
        """
        if not 0 <= batch_index < self.sample_offsets[-1]:
            raise IndexError(f"batch_index {batch_index} is out of range for this PointBatch.")
        sample_index = int(np.searchsorted(self.sample_offsets, batch_index, side="right") - 1)
        local_index = batch_index - self.sample_offsets[sample_index]
        return sample_index, local_index
