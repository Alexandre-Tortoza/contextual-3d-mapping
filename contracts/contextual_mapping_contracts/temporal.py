"""Valores temporais compartilhados.

Timestamps usam nanossegundos inteiros em um clock nomeado. O armazenamento
como inteiro evita arredondamento de ponto flutuante dependente de
plataforma durante sincronização e replay.
"""

from __future__ import annotations

from dataclasses import dataclass


# Um instante não-negativo em um clock nomeado. Existe para que timestamps
# de sensores/fontes diferentes só sejam comparados quando estiverem no
# mesmo clock_id, evitando comparação implícita entre clocks distintos.
@dataclass(frozen=True, order=True)
class Timestamp:
    """Um instante não-negativo expresso em nanossegundos em ``clock_id``."""

    nanoseconds: int
    clock_id: str

    def __post_init__(self) -> None:
        if isinstance(self.nanoseconds, bool) or not isinstance(self.nanoseconds, int):
            raise TypeError("nanoseconds must be an integer.")
        if self.nanoseconds < 0:
            raise ValueError("nanoseconds must be non-negative.")
        if not self.clock_id.strip():
            raise ValueError("clock_id must not be empty.")
