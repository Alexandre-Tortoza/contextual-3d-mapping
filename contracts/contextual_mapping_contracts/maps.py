"""Identidades compartilhadas de mapas persistidos."""

from __future__ import annotations

from dataclasses import dataclass


# Identifica um mapa persistido sem acoplar consumidores a um backend de
# armazenamento. É usado por geometric-map, persistência e map-explorer.
@dataclass(frozen=True, order=True)
class MapId:
    """Identidade estável de um mapa persistido.

    Argumentos:
        value: identificador não vazio do mapa.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("map id must not be empty.")

    def __str__(self) -> str:
        """Retorna a representação textual estável do mapa."""
        return self.value
