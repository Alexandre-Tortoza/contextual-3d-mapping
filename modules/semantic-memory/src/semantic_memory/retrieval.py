"""Índice vetorial exato e determinístico para embeddings semânticos."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


# Representa uma entrada mínima de recuperação sem duplicar a geometria dona.
@dataclass(frozen=True)
class SemanticRetrievalEntry:
    """Entrada vetorial ligada a uma geometria persistente."""

    semantic_id: str
    geometry_id: str
    coordinates_m: tuple[float, float, float]
    values: tuple[float, ...]
    embedding_space: str
    producer: str
    evidence_references: tuple[str, ...] = ()
    label: str | None = None

    # Valida a fronteira antes de indexar um vetor sem sentido ou sem identidade.
    def __post_init__(self) -> None:
        """Valida identidade, vetor normalizado e provenance da entrada."""
        if not self.semantic_id.strip() or not self.geometry_id.strip() or not self.embedding_space.strip() or not self.producer.strip():
            raise ValueError("SemanticRetrievalEntry requires identity, space and producer.")
        if len(self.coordinates_m) != 3 or not all(isfinite(float(value)) for value in self.coordinates_m):
            raise ValueError("SemanticRetrievalEntry coordinates must be finite 3D values.")
        if not self.values or not all(isfinite(float(value)) for value in self.values):
            raise ValueError("SemanticRetrievalEntry values must be finite and non-empty.")


@dataclass(frozen=True)
class VectorSearchResult:
    """Resultado compacto de uma busca vetorial."""

    entry: SemanticRetrievalEntry
    score: float


# Busca por cosseno normalizado e desempata por identidade estável. O índice
# exato é intencional nesta primeira versão: reproduzível e suficiente ao mapa local.
def search_entries(
    entries: tuple[SemanticRetrievalEntry, ...], query: tuple[float, ...], *, limit: int,
    embedding_space: str, producer: str,
) -> tuple[VectorSearchResult, ...]:
    """Retorna candidatos compatíveis ordenados por similaridade cosseno."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    if not entries:
        return ()
    compatible = tuple(entry for entry in entries if entry.embedding_space == embedding_space and entry.producer == producer)
    if not compatible:
        raise ValueError("No indexed semantic embeddings are compatible with the query encoder.")
    dimensions = {len(entry.values) for entry in compatible}
    if dimensions != {len(query)}:
        raise ValueError("Query dimension is incompatible with indexed semantic embeddings.")
    query_values = np.asarray(query, dtype=np.float64)
    norm = float(np.linalg.norm(query_values))
    if norm == 0.0 or not np.isfinite(query_values).all():
        raise ValueError("Query embedding must be finite and non-zero.")
    query_values /= norm
    ranked = []
    for entry in compatible:
        vector = np.asarray(entry.values, dtype=np.float64)
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0.0:
            raise ValueError("Indexed semantic embedding must be non-zero.")
        ranked.append(VectorSearchResult(entry, float(query_values @ (vector / vector_norm))))
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.entry.semantic_id))[:limit])
