"""Testes da fronteira pública de persistência/resolução de embeddings (#217)."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_perception.domain.embeddings import LanguageEmbedding, VisualEmbedding
from visual_perception.infrastructure.embedding_archive import (
    UnresolvableEmbeddingRefError,
    resolve_embedding_vector,
    write_embedding_archive,
)


# Constrói um VisualEmbedding mínimo válido, reutilizado pelos testes abaixo
# para focar no comportamento do archive em vez da construção do embedding.
def _visual_embedding(embedding_id: str, region_id: str, vector: tuple[float, ...]) -> VisualEmbedding:
    return VisualEmbedding(
        embedding_id=embedding_id,
        region_id=region_id,
        vector=vector,
        dimension=len(vector),
        pooling_method="mean",
        feature_resolution="native",
        model_id="fake-visual-encoder",
        # Os vetores destes testes são arbitrários: declará-los normalizados
        # seria falso, e o contract recusa essa declaração (#243).
        normalized=False,
    )


# Constrói um LanguageEmbedding mínimo válido, análogo ao helper acima.
def _language_embedding(embedding_id: str, region_id: str, vector: tuple[float, ...]) -> LanguageEmbedding:
    return LanguageEmbedding(
        embedding_id=embedding_id,
        region_id=region_id,
        vector=vector,
        dimension=len(vector),
        model_id="fake-language-encoder",
        checkpoint="fake-checkpoint",
        normalized=False,
    )


# Garante o contract central da #217: um consumidor fora do módulo que só
# conhece embedding_id consegue recuperar exatamente o vetor que foi escrito,
# sem depender de nenhum detalhe do backend de persistência.
def test_resolve_embedding_vector_round_trips_written_vectors(tmp_path: Path) -> None:
    archive_path = tmp_path / "embeddings.npz"
    visual = _visual_embedding("emb-visual-1", "region-a", (1.0, 2.0, 3.0))
    language = _language_embedding("emb-language-1", "region-a", (0.5, -0.5))

    write_embedding_archive(archive_path, (visual, language))

    assert resolve_embedding_vector(archive_path, "emb-visual-1") == (1.0, 2.0, 3.0)
    assert resolve_embedding_vector(archive_path, "emb-language-1") == (0.5, -0.5)


# Um embedding_ref que nenhum produtor jamais escreveu precisa falhar de
# forma explícita e nomeada, não silenciosamente ou com um KeyError genérico
# do numpy.
def test_resolve_embedding_vector_rejects_unknown_ref(tmp_path: Path) -> None:
    archive_path = tmp_path / "embeddings.npz"
    write_embedding_archive(archive_path, (_visual_embedding("emb-visual-1", "region-a", (1.0,)),))

    with pytest.raises(UnresolvableEmbeddingRefError):
        resolve_embedding_vector(archive_path, "does-not-exist")


# Um archive que nunca foi escrito (nenhum embedding produzido para o frame)
# também deve falhar de forma nomeada, em vez de um FileNotFoundError cru.
def test_resolve_embedding_vector_rejects_missing_archive(tmp_path: Path) -> None:
    with pytest.raises(UnresolvableEmbeddingRefError):
        resolve_embedding_vector(tmp_path / "embeddings.npz", "emb-visual-1")


# write_embedding_archive não cria arquivo nenhum quando não há embeddings a
# persistir, espelhando o guard condicional já usado por quem grava o
# artifact do frame (frame_artifacts.py).
def test_write_embedding_archive_skips_file_when_no_embeddings(tmp_path: Path) -> None:
    archive_path = tmp_path / "embeddings.npz"

    write_embedding_archive(archive_path, ())

    assert not archive_path.exists()
