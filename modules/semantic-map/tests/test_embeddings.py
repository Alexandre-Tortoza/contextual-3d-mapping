"""Testes de round-trip do archive de embeddings semânticos (#116, #117)."""

from __future__ import annotations

from pathlib import Path

import pytest
from semantic_fusion import FusedLanguageEmbedding, LanguageEmbeddingReference
from semantic_map import read_semantic_embedding, write_semantic_embedding_archive


# Constrói um embedding fundido válido para uma única geometria, variando só
# o que cada teste precisa exercitar.
def _fused_embedding(values: tuple[float, ...] = (0.6, 0.8)) -> FusedLanguageEmbedding:
    """Cria um ``FusedLanguageEmbedding`` de teste com uma única referência."""
    reference = LanguageEmbeddingReference(
        embedding_id="region-1", archive_uri="file:///run/frame-000/embeddings.npz",
        embedding_space="clip:vit-b-32:language_aligned", dimension=len(values),
        producer="language_embedding:clip:vit-b-32",
    )
    return FusedLanguageEmbedding(
        values=values, embedding_space=reference.embedding_space, dimension=reference.dimension,
        producer=reference.producer, normalized=True,
        contributor_references=(reference,), effective_weights=(1.0,),
    )


# O caminho comum: o mapa persiste o vetor fora do JSON e devolve metadata
# suficiente para reabri-lo depois sem carregar o vetor em memória a mais.
def test_round_trip_persiste_e_reabre_o_vetor(tmp_path: Path) -> None:
    """Escreve e relê o embedding de uma geometria por ``geometry_id``."""
    archive = tmp_path / "semantic-embeddings.npz"
    metadata = write_semantic_embedding_archive(archive, {"geom-1": _fused_embedding()})

    assert archive.is_file()
    assert metadata["geom-1"]["archive_uri"] == archive.resolve().as_uri()
    assert metadata["geom-1"]["embedding_space"] == "clip:vit-b-32:language_aligned"
    assert metadata["geom-1"]["dimension"] == 2
    assert metadata["geom-1"]["producer"] == "language_embedding:clip:vit-b-32"
    assert metadata["geom-1"]["normalized"] is True
    assert metadata["geom-1"]["effective_weights"] == [1.0]
    assert metadata["geom-1"]["contributors"][0]["embedding_id"] == "region-1"
    assert "values" not in metadata["geom-1"]

    values = read_semantic_embedding(archive, "geom-1", dimension=2)
    assert values == pytest.approx((0.6, 0.8))


# Múltiplas geometrias precisam persistir e reabrir de forma independente,
# sem que uma vazie ou sobrescreva a outra no mesmo archive.
def test_round_trip_com_multiplas_geometrias(tmp_path: Path) -> None:
    """Persiste e relê duas geometrias distintas do mesmo archive."""
    archive = tmp_path / "semantic-embeddings.npz"
    write_semantic_embedding_archive(archive, {
        "geom-1": _fused_embedding((0.6, 0.8)),
        "geom-2": _fused_embedding((1.0, 0.0)),
    })
    assert read_semantic_embedding(archive, "geom-1", dimension=2) == pytest.approx((0.6, 0.8))
    assert read_semantic_embedding(archive, "geom-2", dimension=2) == pytest.approx((1.0, 0.0))


# Um mapa sem nenhuma geometria com embedding CLIP não é um erro: é o caso
# comum enquanto a fusão de linguagem for opt-in. Nenhum arquivo deve sobrar.
def test_mapa_sem_embeddings_nao_escreve_archive(tmp_path: Path) -> None:
    """Não cria o archive e devolve metadata vazia quando não há embeddings."""
    archive = tmp_path / "semantic-embeddings.npz"
    metadata = write_semantic_embedding_archive(archive, {})
    assert metadata == {}
    assert not archive.exists()


# Reabrir de um archive ausente é um erro de produção da run, não um mapa
# vazio: o metadata publicado prometeu uma geometria que o archive não tem.
def test_reabertura_recusa_archive_ausente(tmp_path: Path) -> None:
    """Recusa ler um archive que nunca foi escrito."""
    with pytest.raises(ValueError, match="not found"):
        read_semantic_embedding(tmp_path / "missing.npz", "geom-1", dimension=2)


# O metadata pode declarar uma geometria que o archive concreto não contém,
# por exemplo após uma reescrita parcial: a fronteira precisa recusar, não
# devolver um vetor de outra geometria por acidente.
def test_reabertura_recusa_geometry_id_ausente_no_archive(tmp_path: Path) -> None:
    """Recusa ler uma geometria que não está no archive."""
    archive = tmp_path / "semantic-embeddings.npz"
    write_semantic_embedding_archive(archive, {"geom-1": _fused_embedding()})
    with pytest.raises(ValueError, match="missing"):
        read_semantic_embedding(archive, "geom-2", dimension=2)


# Dimensão divergente entre o metadata e o vetor persistido é o sintoma de
# um metadata desatualizado ou de um archive de outra run. A leitura precisa
# recusar em vez de devolver um vetor que o resolver de fusão calaria.
def test_reabertura_recusa_dimensao_declarada_incompativel(tmp_path: Path) -> None:
    """Recusa ler quando a dimensão declarada diverge do vetor persistido."""
    archive = tmp_path / "semantic-embeddings.npz"
    write_semantic_embedding_archive(archive, {"geom-1": _fused_embedding((0.6, 0.8))})
    with pytest.raises(ValueError, match="invalid vector"):
        read_semantic_embedding(archive, "geom-1", dimension=3)
