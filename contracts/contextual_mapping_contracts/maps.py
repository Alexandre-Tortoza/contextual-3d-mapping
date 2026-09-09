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

    # Impede identidades vazias de alcançarem persistência ou composição.
    def __post_init__(self) -> None:
        """Valida que a identidade possui conteúdo não vazio."""
        if not self.value.strip():
            raise ValueError("map id must not be empty.")

    # Fornece a forma textual usada em paths, logs e artifacts públicos.
    def __str__(self) -> str:
        """Retorna a representação textual estável do mapa."""
        return self.value
