"""Fronteira pública de consultas unificadas ao mapa."""

from .contracts import MapQueryRequest, MapQueryResult, QueryOutcome
from .routing import query_semantic_map

__all__ = ["MapQueryRequest", "MapQueryResult", "QueryOutcome", "query_semantic_map"]
