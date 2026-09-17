"""Roteamento de consulta textual para a capability semântica disponível."""

from collections.abc import Callable

from semantic_memory import SemanticRetrievalEntry, search_entries

from .contracts import MapQueryRequest, MapQueryResult, QueryOutcome


# Executa somente a rota semântica disponível e declara ausência sem fabricar
# resultados. A aplicação fornece o encoder para não acoplar CLIP ao módulo.
def query_semantic_map(
    request: MapQueryRequest, entries: tuple[SemanticRetrievalEntry, ...],
    encode_text: Callable[[str], tuple[float, ...]], *, embedding_space: str, producer: str,
) -> QueryOutcome:
    """Consulta um índice semântico por texto e normaliza o resultado público."""
    if not entries:
        return QueryOutcome("capability_unavailable", reason="semantic_embedding_capability_unavailable")
    matches = search_entries(entries, encode_text(request.text), limit=request.limit,
                            embedding_space=embedding_space, producer=producer)
    return QueryOutcome("ok", tuple(MapQueryResult(
        semantic_id=item.entry.semantic_id, geometry_id=item.entry.geometry_id,
        coordinates_m=item.entry.coordinates_m, score=item.score,
        embedding_space=item.entry.embedding_space,
        evidence_references=item.entry.evidence_references, label=item.entry.label,
    ) for item in matches))
