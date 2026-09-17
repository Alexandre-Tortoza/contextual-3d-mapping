"""Monta o índice de busca vetorial a partir de um artifact de mapa publicado.

Existe porque nem ``semantic-map`` (que só sabe persistir e reabrir vetores)
nem ``semantic-memory`` (que só sabe indexar entradas já construídas) devem
saber como um artifact de mapa nomeia pontos e metadata de embedding — essa
adaptação de payload pertence à aplicação, não a nenhum dos dois módulos.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse

from semantic_map import read_semantic_embedding
from semantic_memory import SemanticRetrievalEntry


# Converte o ``archive_uri`` publicado (``file://...``) de volta em um Path
# local. A fronteira de fusão só trafega URIs para não amarrar o artifact a
# um sistema de arquivos específico; aqui é onde a aplicação os resolve.
def _archive_path(archive_uri: str) -> Path:
    """Resolve um ``archive_uri`` de arquivo local para o ``Path`` correspondente."""
    return Path(unquote(urlparse(archive_uri).path))


# Constrói o índice de busca de um mapa a partir do payload publicado. A
# ausência de ``semantic_embeddings`` não é um erro: é o estado comum
# enquanto a fusão de linguagem for opt-in, e o chamador trata isso como
# capability indisponível, não como falha.
def load_semantic_index(map_payload: Mapping[str, object]) -> tuple[SemanticRetrievalEntry, ...]:
    """Carrega as entradas de busca semântica de um artifact de mapa.

    Argumentos:
        map_payload: documento JSON já desserializado do mapa (run contextual
            ou mapa consolidado) com ``points`` e, opcionalmente,
            ``semantic_embeddings``.
    Retorna:
        entradas prontas para ``semantic_memory.search_entries``; vazia se o
        mapa não publicou nenhum embedding CLIP fundido.
    """
    semantic_embeddings = map_payload.get("semantic_embeddings")
    if not isinstance(semantic_embeddings, Mapping) or not semantic_embeddings:
        return ()
    points = map_payload.get("points")
    if not isinstance(points, list):
        raise ValueError("map payload with semantic_embeddings must declare points.")
    points_by_geometry_id = {
        point["geometry_id"]: point
        for point in points
        if isinstance(point, Mapping) and isinstance(point.get("geometry_id"), str)
    }
    entries = []
    for geometry_id, metadata in semantic_embeddings.items():
        point = points_by_geometry_id.get(geometry_id)
        if point is None:
            continue
        coordinates = tuple(float(value) for value in point["coordinates_m"])
        values = read_semantic_embedding(
            _archive_path(metadata["archive_uri"]), geometry_id, dimension=int(metadata["dimension"])
        )
        label = (point.get("context") or {}).get("label")
        contributors = metadata.get("contributors") or ()
        entries.append(
            SemanticRetrievalEntry(
                semantic_id=f"semantic:{geometry_id}",
                geometry_id=geometry_id,
                coordinates_m=coordinates,
                values=values,
                embedding_space=str(metadata["embedding_space"]),
                producer=str(metadata["producer"]),
                evidence_references=tuple(
                    f"{contributor['archive_uri']}#{contributor['embedding_id']}"
                    for contributor in contributors
                ),
                label=label if isinstance(label, str) else None,
            )
        )
    return tuple(entries)
