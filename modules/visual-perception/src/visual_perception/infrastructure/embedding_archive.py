"""Fronteira de persistência e resolução dos vetores de embedding por referência.

Issue: #217.

``VisualEmbedding``/``LanguageEmbedding`` viajam pelo pipeline por referência
(``embedding_id``/``artifact_ref``, ver ``domain/embeddings.py`` e
``docs/artifacts.md``): a observação canônica nunca embute o vetor. Até a
#217, o único código que escrevia e lia o artifact onde esses vetores
realmente vivem estava dentro de ``benchmarks/frame_artifacts.py``, que não é
empacotado (``pyproject.toml`` só empacota ``src/visual_perception``) e
portanto não é importável por nenhum outro módulo. Uma referência que nenhum
consumidor consegue resolver não é diferente, na prática, de um vetor
descartado. Este módulo é a fronteira pública que fecha esse gap: quem
produz o artifact (``frame_artifacts.py``) e quem eventualmente for
consumi-lo fora do módulo (sensor-association, semantic-fusion) devem
depender só destas duas funções.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from visual_perception.domain.embeddings import LanguageEmbedding, VisualEmbedding


# Sinaliza que um embedding_ref anunciado por uma observação (visual_embedding_ref,
# language_embedding_ref ou RegionEvidenceSlot.artifact_ref) não resolve para
# nenhum vetor no artifact indicado. Existe como erro nomeado, e não um KeyError
# genérico do numpy, para que a fronteira falhe com uma mensagem acionável em
# vez de um traceback interno de biblioteca.
class UnresolvableEmbeddingRefError(ValueError):
    """Um ``embedding_ref`` não resolve para nenhum vetor no archive indicado."""


# Persiste os vetores de um conjunto de embeddings, indexados por embedding_id.
# Existe para que a escrita do artifact tenha um único dono público, em vez de
# ficar duplicada entre o harness de benchmark e qualquer outro escritor futuro
# (mapping-runtime, por exemplo, ao compor um run fora do benchmark).
def write_embedding_archive(
    path: Path, embeddings: Iterable[VisualEmbedding | LanguageEmbedding]
) -> None:
    """Escreve os vetores de ``embeddings`` em ``path``, indexados por ``embedding_id``.

    Argumentos:
        path: caminho do artifact ``.npz`` a criar (diretório pai deve existir).
        embeddings: os embeddings cujos vetores serão persistidos.
    """
    vectors = {
        embedding.embedding_id: np.asarray(embedding.vector, dtype=np.float32)
        for embedding in embeddings
    }
    if not vectors:
        return
    # mypy trata **vectors como podendo preencher o allow_pickle: bool nomeado
    # de savez_compressed, já que o stub não usa TypedDict; é um falso positivo
    # do stub, não um erro real de chamada.
    np.savez_compressed(path, **vectors)  # type: ignore[arg-type]


# Resolve um embedding_ref para o vetor que ele referencia. Existe como o
# único caminho público de leitura do artifact, para que um consumidor fora
# do módulo nunca precise saber que o backend de persistência é um .npz.
def resolve_embedding_vector(path: Path, embedding_ref: str) -> tuple[float, ...]:
    """Resolve ``embedding_ref`` para o vetor persistido em ``path``.

    Argumentos:
        path: caminho do artifact ``.npz`` escrito por :func:`write_embedding_archive`.
        embedding_ref: o ``embedding_id`` a resolver (o mesmo valor usado em
            ``ObservedRegion.visual_embedding_ref``/``language_embedding_ref``
            ou ``RegionEvidenceSlot.artifact_ref``).
    Retorna:
        o vetor como tupla de floats finitos.
    Levanta:
        UnresolvableEmbeddingRefError: se ``path`` não existir ou não contiver
            ``embedding_ref``.
    """
    if not path.is_file():
        raise UnresolvableEmbeddingRefError(
            f"Embedding archive not found: {path} (looking for ref {embedding_ref!r})."
        )
    with np.load(path) as archive:
        if embedding_ref not in archive.files:
            raise UnresolvableEmbeddingRefError(
                f"embedding_ref {embedding_ref!r} not found in archive {path}."
            )
        vector = tuple(float(component) for component in archive[embedding_ref])
    if any(math.isnan(component) or math.isinf(component) for component in vector):
        raise UnresolvableEmbeddingRefError(
            f"embedding_ref {embedding_ref!r} in archive {path} resolved to a non-finite vector."
        )
    return vector
