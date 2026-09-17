"""Testes da montagem do índice de busca a partir de um artifact de mapa."""

from __future__ import annotations

from pathlib import Path

import pytest
from map_explorer_api.map_index import load_semantic_index
from semantic_fusion import FusedLanguageEmbedding, LanguageEmbeddingReference
from semantic_map import write_semantic_embedding_archive


# Publica um archive de embeddings real e devolve a metadata compacta que um
# artifact de mapa publicaria, para exercitar o índice sobre um archive
# resolvível de verdade em vez de um caminho fabricado.
def _published_map(tmp_path: Path) -> dict:
    """Constrói um payload de mapa mínimo com um embedding CLIP publicado."""
    archive = tmp_path / "semantic-embeddings.npz"
    reference = LanguageEmbeddingReference(
        embedding_id="frame-a:region-1", archive_uri="file:///run/frame-a/embeddings.npz",
        embedding_space="clip:vit-b-32:language_aligned", dimension=2,
        producer="language_embedding:clip:vit-b-32",
    )
    fused = FusedLanguageEmbedding(
        values=(0.6, 0.8), embedding_space=reference.embedding_space, dimension=2,
        producer=reference.producer, normalized=True,
        contributor_references=(reference,), effective_weights=(1.0,),
    )
    semantic_embeddings = write_semantic_embedding_archive(archive, {"geom-1": fused})
    return {
        "points": [
            {"geometry_id": "geom-1", "coordinates_m": [1.0, 2.0, 3.0], "context": {"label": "porta"}},
            {"geometry_id": "geom-2", "coordinates_m": [4.0, 5.0, 6.0]},
        ],
        "semantic_embeddings": semantic_embeddings,
    }


# O caminho comum: uma geometria com embedding publicado vira uma entrada de
# busca com coordenadas, label e vetor resolvido do archive real.
def test_carrega_entradas_com_embedding_publicado(tmp_path: Path) -> None:
    """Confere entrada completa para a geometria com embedding CLIP."""
    entries = load_semantic_index(_published_map(tmp_path))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.geometry_id == "geom-1"
    assert entry.coordinates_m == (1.0, 2.0, 3.0)
    assert entry.values == pytest.approx((0.6, 0.8))
    assert entry.embedding_space == "clip:vit-b-32:language_aligned"
    assert entry.producer == "language_embedding:clip:vit-b-32"
    assert entry.label == "porta"
    assert entry.evidence_references == ("file:///run/frame-a/embeddings.npz#frame-a:region-1",)


# Um mapa sem nenhum embedding CLIP publicado é o estado comum enquanto a
# fusão de linguagem for opt-in: o índice fica vazio, e não é um erro.
def test_mapa_sem_semantic_embeddings_devolve_indice_vazio() -> None:
    """Devolve tupla vazia quando o mapa não publicou embeddings CLIP."""
    payload = {"points": [{"geometry_id": "geom-1", "coordinates_m": [0.0, 0.0, 0.0]}]}
    assert load_semantic_index(payload) == ()


# Uma geometria publicada em semantic_embeddings sem o ponto correspondente
# indicaria um mapa inconsistente; o índice ignora em vez de inventar
# coordenadas, mas não deve propagar um vetor sem geometria dona.
def test_ignora_geometria_publicada_sem_ponto_correspondente(tmp_path: Path) -> None:
    """Ignora entradas de ``semantic_embeddings`` sem ponto correspondente."""
    payload = _published_map(tmp_path)
    payload["points"] = [point for point in payload["points"] if point["geometry_id"] != "geom-1"]
    assert load_semantic_index(payload) == ()
