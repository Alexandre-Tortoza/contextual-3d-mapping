"""Contracts de entrada e saída do port `PointEncoder`.

Issue: #19.

``EncoderInputBatch`` é a forma batched de :class:`~point_representation.application.feature_selection.EncoderInput`
(coordenadas e features já selecionadas, #4) somada às fronteiras de amostra
de um :class:`~point_representation.domain.batch.PointBatch` (#9). ``EncoderOutput``
é deliberadamente mais fraco que "uma feature por ponto de entrada": um
backbone concreto (#20) pode discretizar internamente e mesclar pontos, então
a saída carrega seu próprio ``point_lineage`` (#8) de volta para os índices de
entrada, em vez de assumir uma correspondência 1:1 que só um adapter real
pode garantir (#21).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from point_representation.domain.csr import validate_csr_offsets
from point_representation.domain.lineage import PointLineage


# Entrada batched de um encoder concreto: coordenadas e features já
# selecionadas e concatenadas através de múltiplas amostras, com as
# fronteiras de amostra preservadas via offsets CSR.
@dataclass(frozen=True)
class EncoderInputBatch:
    """Entrada de um `PointEncoder`, com coordenadas e features batched.

    Argumentos:
        coordinates: array ``(N, 3)`` de todas as amostras concatenadas.
        features: array ``(N, feature_dimension)`` das features selecionadas
            (#4); ``(N, 0)`` para entrada somente-coordenadas.
        feature_dimension: largura de ``features``.
        selected_channels: canais que compõem ``features``, na ordem de
            concatenação.
        sample_offsets: offsets estilo CSR sobre os ``N`` pontos de entrada.
    """

    coordinates: np.ndarray
    features: np.ndarray
    feature_dimension: int
    selected_channels: tuple[str, ...]
    sample_offsets: tuple[int, ...]

    # Valida forma e alinhamento entre coordenadas, features e offsets — um
    # encoder concreto não deveria ter que redescobrir essas invariantes.
    def __post_init__(self) -> None:
        """Valida forma de ``coordinates``/``features`` e os offsets de amostra."""
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError(f"EncoderInputBatch.coordinates must have shape (N, 3), got {self.coordinates.shape}.")
        if self.features.shape[0] != len(self.coordinates):
            raise ValueError(
                f"EncoderInputBatch.features must align with coordinates "
                f"({len(self.coordinates)} rows), got {self.features.shape[0]}."
            )
        if self.features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"EncoderInputBatch.feature_dimension={self.feature_dimension} does not match "
                f"features.shape[1]={self.features.shape[1]}."
            )
        validate_csr_offsets(self.sample_offsets, len(self.coordinates), field="EncoderInputBatch.sample_offsets")

    @property
    def sample_count(self) -> int:
        """Retorna quantas amostras compõem o lote de entrada."""
        return len(self.sample_offsets) - 1


# Saída de um encoder concreto: uma feature por linha de saída, com o
# lineage que resolve cada linha de volta aos pontos de entrada que a
# originaram e os offsets de amostra sobre o espaço de saída.
@dataclass(frozen=True)
class EncoderOutput:
    """Saída de um `PointEncoder`.

    Argumentos:
        features: array ``(M, embedding_dimension)`` de features por linha
            de saída; ``M`` pode ser menor que o número de pontos de entrada
            quando o backbone mescla pontos internamente (ver
            ``point_lineage``).
        point_lineage: para cada linha de ``features``, os índices (no
            espaço de entrada, ``EncoderInputBatch.coordinates``) que a
            originaram. Um-para-um quando o backbone não mescla pontos.
        sample_offsets: offsets estilo CSR sobre as ``M`` linhas de saída,
            preservando a fronteira de amostra de ``EncoderInputBatch``.
    """

    features: np.ndarray
    point_lineage: PointLineage
    sample_offsets: tuple[int, ...]

    # Valida que features, lineage e offsets concordam sobre quantas linhas
    # de saída existem, e que os valores são finitos — uma saída inconsistente
    # corromperia silenciosamente qualquer consumidor que confie em uma das
    # três sem checar as outras.
    def __post_init__(self) -> None:
        """Valida consistência entre ``features``, ``point_lineage`` e ``sample_offsets``."""
        if self.features.ndim != 2:
            raise ValueError(f"EncoderOutput.features must be 2D, got shape {self.features.shape}.")
        if not np.isfinite(self.features).all():
            raise ValueError("EncoderOutput.features must contain only finite values.")
        if self.features.shape[0] != self.point_lineage.point_count:
            raise ValueError(
                f"EncoderOutput.features has {self.features.shape[0]} rows, but point_lineage "
                f"covers {self.point_lineage.point_count} output points."
            )
        validate_csr_offsets(self.sample_offsets, self.features.shape[0], field="EncoderOutput.sample_offsets")

    @property
    def embedding_dimension(self) -> int:
        """Retorna a dimensão de embedding observada em ``features``."""
        return int(self.features.shape[1])
