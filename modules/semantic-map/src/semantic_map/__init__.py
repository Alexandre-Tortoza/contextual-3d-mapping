"""Fronteira pública da consolidação persistente de mapas semânticos."""

from .consolidation import (
    CONSOLIDATED_ARTIFACT_TYPE,
    ConsolidatedContextMap,
    PublishedContextRun,
    consolidate_context_runs,
    geometry_fingerprint,
)
from .embeddings import read_semantic_embedding, write_semantic_embedding_archive

__all__ = [
    "CONSOLIDATED_ARTIFACT_TYPE",
    "ConsolidatedContextMap",
    "PublishedContextRun",
    "consolidate_context_runs",
    "geometry_fingerprint",
    "read_semantic_embedding",
    "write_semantic_embedding_archive",
]
