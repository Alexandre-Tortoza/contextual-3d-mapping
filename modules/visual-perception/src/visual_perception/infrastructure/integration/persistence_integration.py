"""Integra artifacts de evidência visual com a persistência do repositório.

Issue: #180.

``persistence`` (#112) ainda não está implementado. Este módulo define o
pequeno port que visual-perception precisa dele — um evidence store opaco de
chave/valor — e funções que fazem o round-trip de uma observação canônica e
seus embeddings através desse port. Nenhum client ou path específico de
armazenamento aparece nos contracts públicos do módulo: tudo cruza essa
fronteira como dicts simples, prontos para JSON.
"""

from __future__ import annotations

from typing import Any, Protocol

from visual_perception.domain.embeddings import LanguageEmbedding, VisualEmbedding
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.infrastructure.serialization import deserialize_observation, serialize_observation


# Define a capacidade mínima que visual-perception exige da persistência do
# repositório. Existe para que este módulo nunca dependa de um client de
# storage concreto — qualquer implementação de persistência que satisfaça
# este Protocol serve.
class EvidencePersistencePort(Protocol):
    """A capacidade mínima que visual-perception exige da persistência do repositório."""

    # Persiste o payload e devolve uma referência estável e resolvível
    # (uma URI), usada depois por reload_observation/get para reconstruir o
    # objeto original.
    def put(self, artifact_id: str, payload: dict[str, Any]) -> str:
        """Persiste ``payload`` e devolve uma referência estável e resolvível (uma URI)."""
        ...

    # Resolve uma referência previamente devolvida por put, completando o
    # round-trip de persistência.
    def get(self, reference: str) -> dict[str, Any]:
        """Resolve uma referência previamente devolvida por :meth:`put`."""
        ...


# Serializa a observação canônica (via serialization.py) e a envia ao evidence
# store através do port, devolvendo a referência que permite recarregá-la
# depois com reload_observation.
def persist_observation(observation: VisualObservation, store: EvidencePersistencePort) -> str:
    return store.put(observation.observation_id, serialize_observation(observation))


# Resolve a referência no evidence store e reconstrói a VisualObservation
# original a partir do dict serializado — o lado inverso de
# persist_observation.
def reload_observation(reference: str, store: EvidencePersistencePort) -> VisualObservation:
    return deserialize_observation(store.get(reference))


# Persiste cada embedding visual individualmente no evidence store, para que
# embeddings (potencialmente grandes) sejam referenciados por id em vez de
# embutidos na observação serializada.
def persist_visual_embeddings(
    embeddings: tuple[VisualEmbedding, ...], store: EvidencePersistencePort
) -> dict[str, str]:
    """Persiste cada embedding e devolve {embedding_id: reference}."""
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


# Persiste cada embedding de linguagem individualmente no evidence store,
# mesmo raciocínio de persist_visual_embeddings mas para o embedding
# textual/linguístico de cada região.
def persist_language_embeddings(
    embeddings: tuple[LanguageEmbedding, ...], store: EvidencePersistencePort
) -> dict[str, str]:
    """Persiste cada embedding e devolve {embedding_id: reference}."""
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
