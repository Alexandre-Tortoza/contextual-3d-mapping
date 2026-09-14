"""Fronteira substituível para obter features 3D aprendidas por ponto.

Issue: #19.

Treino e inferência dependem só deste Protocol, nunca de um backbone
concreto (#20) ou de um framework de tensor específico — isso é o que
permite trocar o backbone (ou testar com um fake determinístico, sem GPU)
sem tocar em nenhum código de composição.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from point_representation.domain.encoder_io import EncoderInputBatch, EncoderOutput


# Contract framework-agnóstico de um encoder 3D. Uma implementação declara
# sua dimensão de embedding e quais canais exige, e ``encode`` converte um
# lote de entrada em features por ponto (possivelmente mescladas
# internamente, ver ``EncoderOutput.point_lineage``).
@runtime_checkable
class PointEncoder(Protocol):
    """Encoder que produz features 3D aprendidas a partir de um `EncoderInputBatch`."""

    @property
    def embedding_dimension(self) -> int:
        """A dimensão do vetor de saída por ponto que este encoder produz."""
        ...

    @property
    def required_channels(self) -> frozenset[str]:
        """Os canais que ``batch.selected_channels`` deve conter para ``encode`` funcionar.

        Vazio declara suporte a entrada somente-coordenadas.
        """
        ...

    def encode(self, batch: EncoderInputBatch) -> EncoderOutput:
        """Codifica ``batch`` em features por ponto.

        Argumentos:
            batch: entrada batched, com ``batch.selected_channels`` incluindo
                ao menos ``required_channels``.
        Retorna:
            a saída do encoder, com ``embedding_dimension`` colunas por linha
            e proveniência de volta aos pontos de ``batch`` via
            ``EncoderOutput.point_lineage``.
        """
        ...
