"""Fronteira pública da consolidação persistente de mapas semânticos."""

from .consolidation import (
    CONSOLIDATED_ARTIFACT_TYPE,
    ConsolidatedContextMap,
    PublishedContextRun,
    consolidate_context_runs,
    geometry_fingerprint,
)

__all__ = [
    "CONSOLIDATED_ARTIFACT_TYPE",
    "ConsolidatedContextMap",
    "PublishedContextRun",
    "consolidate_context_runs",
    "geometry_fingerprint",
]
