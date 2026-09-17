"""Testes do roteamento de consulta textual para a capability semântica (#117)."""

from __future__ import annotations

from query_engine import MapQueryRequest, query_semantic_map
from semantic_memory import SemanticRetrievalEntry


def _entry(geometry_id: str, values: tuple[float, ...]) -> SemanticRetrievalEntry:
    """Cria uma entrada de teste no espaço fixo usado por estes testes."""
    return SemanticRetrievalEntry(
        semantic_id=f"semantic:{geometry_id}", geometry_id=geometry_id, coordinates_m=(0.0, 0.0, 0.0),
        values=values, embedding_space="clip:vit-b-32:language_aligned",
        producer="language_embedding:clip:vit-b-32",
    )


# O caminho comum: com entradas e um encoder de texto, a rota devolve um
# QueryOutcome de sucesso normalizado a partir dos matches do índice.
def test_query_semantic_map_devolve_outcome_ok() -> None:
    """Confere status ok e resultado normalizado quando a busca encontra matches."""
    request = MapQueryRequest(map_id="corridor-02", text="porta", limit=5)
    outcome = query_semantic_map(
        request, (_entry("geom-1", (1.0, 0.0)),), lambda _text: (1.0, 0.0),
        embedding_space="clip:vit-b-32:language_aligned", producer="language_embedding:clip:vit-b-32",
    )
    assert outcome.status == "ok"
    assert outcome.results[0].geometry_id == "geom-1"


# Sem nenhuma entrada indexada, a rota declara a ausência da capability em
# vez de chamar o encoder de texto sobre um índice vazio.
def test_query_semantic_map_sem_entradas_declara_capability_indisponivel() -> None:
    """Confere ``capability_unavailable`` quando não há entradas indexadas."""
    request = MapQueryRequest(map_id="corridor-02", text="porta")
    outcome = query_semantic_map(
        request, (), lambda _text: (1.0, 0.0),
        embedding_space="clip:vit-b-32:language_aligned", producer="language_embedding:clip:vit-b-32",
    )
    assert outcome.status == "capability_unavailable"
    assert outcome.reason == "semantic_embedding_capability_unavailable"
    assert outcome.results == ()
