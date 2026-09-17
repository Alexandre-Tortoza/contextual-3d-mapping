"""Contracts serializáveis de request e resultado de consulta de mapa."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MapQueryRequest:
    """Pedido textual de busca em um mapa aberto."""

    map_id: str
    text: str
    limit: int = 10

    def __post_init__(self) -> None:
        """Valida identidade, texto e limite da consulta."""
        if not self.map_id.strip() or not self.text.strip() or self.limit < 1:
            raise ValueError("MapQueryRequest requires map_id, text and positive limit.")


@dataclass(frozen=True)
class MapQueryResult:
    """Match semântico compacto ligado a uma geometria persistente."""

    semantic_id: str
    geometry_id: str
    coordinates_m: tuple[float, float, float]
    score: float
    embedding_space: str
    evidence_references: tuple[str, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        """Valida resultado transportável ao consumidor de mapa."""
        if not self.semantic_id.strip() or not self.geometry_id.strip() or not self.embedding_space.strip():
            raise ValueError("MapQueryResult requires identities and embedding space.")
        if len(self.coordinates_m) != 3 or not isfinite(self.score):
            raise ValueError("MapQueryResult requires finite score and 3D coordinates.")


@dataclass(frozen=True)
class QueryOutcome:
    """Resultado ou indisponibilidade explícita de uma capability."""

    status: str
    results: tuple[MapQueryResult, ...] = ()
    reason: str | None = None
