"""Persistência compacta de embeddings semânticos ligados à geometria."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from semantic_fusion import FusedLanguageEmbedding


# Escreve vetores fora do JSON do mapa para preservar a fronteira de payloads
# grandes. É usada pela composição que publica um mapa semântico consolidado.
def write_semantic_embedding_archive(
    path: Path, embeddings: dict[str, FusedLanguageEmbedding],
) -> dict[str, dict]:
    """Persiste embeddings por geometry_id e retorna metadata serializável.

    Argumentos:
        path: archive NPZ de destino do mapa semântico.
        embeddings: embeddings fundidos indexados pela geometria estável.
    Retorna:
        metadata compacta por geometria, sem vetores.
    """
    if not embeddings:
        return {}
    np.savez_compressed(path, **{
        geometry_id: np.asarray(embedding.values, dtype=np.float32)
        for geometry_id, embedding in embeddings.items()
    })
    return {
        geometry_id: {
            "archive_uri": path.resolve().as_uri(),
            "embedding_space": embedding.embedding_space,
            "dimension": embedding.dimension,
            "producer": embedding.producer,
            "normalized": embedding.normalized,
            "contributors": [asdict(reference) for reference in embedding.contributor_references],
            "effective_weights": list(embedding.effective_weights),
        }
        for geometry_id, embedding in embeddings.items()
    }


# Reabre um vetor apenas quando o consumidor precisa indexar ou comparar. O
# mapa continua transportando metadata e URI, nunca uma cópia em JSON.
def read_semantic_embedding(path: Path, geometry_id: str, *, dimension: int) -> tuple[float, ...]:
    """Resolve um embedding persistido e valida sua dimensão declarada."""
    if not path.is_file():
        raise ValueError(f"Semantic embedding archive not found: {path}.")
    with np.load(path) as archive:
        if geometry_id not in archive.files:
            raise ValueError(f"Semantic embedding for geometry_id {geometry_id!r} is missing.")
        values = tuple(float(value) for value in archive[geometry_id])
    if len(values) != dimension or not np.isfinite(values).all():
        raise ValueError("Semantic embedding archive contains invalid vector metadata or values.")
    return values
