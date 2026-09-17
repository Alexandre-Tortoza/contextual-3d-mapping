"""Testes da busca vetorial exata e determinística (#116)."""

from __future__ import annotations

import pytest
from semantic_memory import SemanticRetrievalEntry, search_entries


# Constrói uma entrada compatível variando só o que cada teste precisa.
def _entry(geometry_id: str, values: tuple[float, ...], *, embedding_space: str = "clip:vit-b-32:language_aligned",
           producer: str = "language_embedding:clip:vit-b-32", label: str | None = None) -> SemanticRetrievalEntry:
    """Cria uma entrada de teste com coordenadas fixas e identidade compatível."""
    return SemanticRetrievalEntry(
        semantic_id=f"semantic:{geometry_id}", geometry_id=geometry_id, coordinates_m=(0.0, 0.0, 0.0),
        values=values, embedding_space=embedding_space, producer=producer, label=label,
    )


# O caminho comum: a entrada mais alinhada com a query vem primeiro.
def test_ordena_por_similaridade_de_cosseno_decrescente() -> None:
    """Confere ordenação por cosseno com a entrada mais próxima primeiro."""
    entries = (
        _entry("geom-1", (1.0, 0.0), label="porta"),
        _entry("geom-2", (0.0, 1.0), label="janela"),
    )
    results = search_entries(entries, (0.9, 0.1), limit=10, embedding_space="clip:vit-b-32:language_aligned",
                              producer="language_embedding:clip:vit-b-32")
    assert [result.entry.geometry_id for result in results] == ["geom-1", "geom-2"]
    assert results[0].score > results[1].score


# O limite precisa cortar a lista, não só truncar a exibição em outro lugar.
def test_respeita_o_limite_de_resultados() -> None:
    """Confere que só ``limit`` entradas são devolvidas."""
    entries = tuple(_entry(f"geom-{i}", (1.0, float(i))) for i in range(5))
    results = search_entries(entries, (1.0, 0.0), limit=2, embedding_space="clip:vit-b-32:language_aligned",
                              producer="language_embedding:clip:vit-b-32")
    assert len(results) == 2


# Entradas de outro espaço/produtor não podem ser comparadas por cosseno como
# se fossem compatíveis: filtrar é a única forma segura de lidar com isso.
def test_filtra_entradas_de_espaco_ou_produtor_incompativel() -> None:
    """Ignora entradas fora do espaço/produtor pedido pela query."""
    entries = (
        _entry("geom-1", (1.0, 0.0)),
        _entry("geom-2", (1.0, 0.0), embedding_space="siglip:base:language_aligned",
               producer="language_embedding:siglip:base"),
    )
    results = search_entries(entries, (1.0, 0.0), limit=10, embedding_space="clip:vit-b-32:language_aligned",
                              producer="language_embedding:clip:vit-b-32")
    assert [result.entry.geometry_id for result in results] == ["geom-1"]


# Nenhuma entrada compatível é diferente de nenhuma entrada: o índice tem
# conteúdo, só não serve a esta query, e isso precisa ser um erro nomeado.
def test_recusa_quando_nenhuma_entrada_e_compativel() -> None:
    """Recusa a busca quando nada no índice combina com espaço/produtor."""
    entries = (_entry("geom-1", (1.0, 0.0), embedding_space="siglip:base:language_aligned",
                       producer="language_embedding:siglip:base"),)
    with pytest.raises(ValueError, match="No indexed"):
        search_entries(entries, (1.0, 0.0), limit=10, embedding_space="clip:vit-b-32:language_aligned",
                        producer="language_embedding:clip:vit-b-32")


# Uma query de dimensão diferente da indexada não pode ser comparada por
# produto escalar sem erro numérico silencioso.
def test_recusa_query_com_dimensao_incompativel() -> None:
    """Recusa quando a dimensão da query diverge das entradas indexadas."""
    entries = (_entry("geom-1", (1.0, 0.0, 0.0)),)
    with pytest.raises(ValueError, match="dimension"):
        search_entries(entries, (1.0, 0.0), limit=10, embedding_space="clip:vit-b-32:language_aligned",
                        producer="language_embedding:clip:vit-b-32")


# Uma query nula não tem direção, e "similaridade" com um vetor sem norma
# não tem significado geométrico.
def test_recusa_query_nula() -> None:
    """Recusa uma query com norma zero."""
    entries = (_entry("geom-1", (1.0, 0.0)),)
    with pytest.raises(ValueError, match="finite and non-zero"):
        search_entries(entries, (0.0, 0.0), limit=10, embedding_space="clip:vit-b-32:language_aligned",
                        producer="language_embedding:clip:vit-b-32")


# Sem entradas, a busca não é um erro: é o índice vazio, o estado comum antes
# de qualquer mapa publicar embeddings.
def test_indice_vazio_devolve_tupla_vazia() -> None:
    """Devolve tupla vazia para um índice sem nenhuma entrada."""
    results = search_entries((), (1.0, 0.0), limit=10, embedding_space="clip:vit-b-32:language_aligned",
                              producer="language_embedding:clip:vit-b-32")
    assert results == ()
