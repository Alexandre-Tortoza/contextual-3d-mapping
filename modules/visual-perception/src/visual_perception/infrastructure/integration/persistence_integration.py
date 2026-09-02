"""Integrate visual evidence artifacts with repository persistence.

Issue: #180.

``persistence`` (#112) is not implemented yet. This module defines the small
port visual-perception needs from it — an opaque key/value evidence store —
and functions that round-trip a canonical observation and its embeddings
through that port. No storage-specific client or path appears in the public
module contracts: everything crosses this boundary as plain JSON-able dicts.
"""

from __future__ import annotations

from typing import Any, Protocol

from visual_perception.domain.embeddings import LanguageEmbedding, VisualEmbedding
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.infrastructure.serialization import deserialize_observation, serialize_observation


class EvidencePersistencePort(Protocol):
    """The minimal capability visual-perception needs from repository persistence."""

    def put(self, artifact_id: str, payload: dict[str, Any]) -> str:
        """Persist ``payload`` and return a stable, resolvable reference (a URI)."""
        ...

    def get(self, reference: str) -> dict[str, Any]:
        """Resolve a reference previously returned by :meth:`put`."""
        ...


def persist_observation(observation: VisualObservation, store: EvidencePersistencePort) -> str:
    return store.put(observation.observation_id, serialize_observation(observation))


def reload_observation(reference: str, store: EvidencePersistencePort) -> VisualObservation:
    return deserialize_observation(store.get(reference))


def persist_visual_embeddings(
    embeddings: tuple[VisualEmbedding, ...], store: EvidencePersistencePort
) -> dict[str, str]:
    """Persist each embedding and return {embedding_id: reference}."""
    return {
        embedding.embedding_id: store.put(
            embedding.embedding_id,
            {
                "region_id": embedding.region_id,
                "vector": list(embedding.vector),
                "dimension": embedding.dimension,
                "pooling_method": embedding.pooling_method,
                "feature_resolution": embedding.feature_resolution,
                "model_id": embedding.model_id,
                "normalized": embedding.normalized,
            },
        )
        for embedding in embeddings
    }


def persist_language_embeddings(
    embeddings: tuple[LanguageEmbedding, ...], store: EvidencePersistencePort
) -> dict[str, str]:
    """Persist each embedding and return {embedding_id: reference}."""
    return {
        embedding.embedding_id: store.put(
            embedding.embedding_id,
            {
                "region_id": embedding.region_id,
                "vector": list(embedding.vector),
                "dimension": embedding.dimension,
                "model_id": embedding.model_id,
                "checkpoint": embedding.checkpoint,
                "normalized": embedding.normalized,
                "dtype": embedding.dtype,
            },
        )
        for embedding in embeddings
    }
