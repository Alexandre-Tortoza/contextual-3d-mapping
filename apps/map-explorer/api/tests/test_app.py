"""Testes da composição executável: carregar um mapa do disco e servi-lo."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

from map_explorer_api.app import build_single_map_service
from map_explorer_api.service import serve
from semantic_fusion import FusedLanguageEmbedding, LanguageEmbeddingReference
from semantic_map import write_semantic_embedding_archive


# Publica um mapa mínimo em disco com o backend fake, para exercitar a
# composição completa sem depender de um checkpoint CLIP real.
def _write_map(tmp_path: Path) -> Path:
    """Escreve um artifact de mapa mínimo com um embedding fake publicado."""
    embeddings_archive = tmp_path / "semantic-embeddings.npz"
    reference = LanguageEmbeddingReference(
        embedding_id="frame-a:region-1", archive_uri="file:///run/frame-a/embeddings.npz",
        embedding_space="fake:none:language_aligned", dimension=8, producer="language_embedding:fake:none",
    )
    fused = FusedLanguageEmbedding(
        values=tuple([1.0] + [0.0] * 7), embedding_space=reference.embedding_space, dimension=8,
        producer=reference.producer, normalized=True,
        contributor_references=(reference,), effective_weights=(1.0,),
    )
    semantic_embeddings = write_semantic_embedding_archive(embeddings_archive, {"geom-1": fused})
    map_path = tmp_path / "context.json"
    map_path.write_text(json.dumps({
        "points": [{"geometry_id": "geom-1", "coordinates_m": [1.0, 2.0, 3.0], "context": {"label": "porta"}}],
        "semantic_embeddings": semantic_embeddings,
    }), encoding="utf-8")
    return map_path


# De ponta a ponta: um mapa publicado em disco deve responder a uma consulta
# textual real via HTTP, usando o encoder fake resolvido automaticamente.
def test_build_single_map_service_responde_a_consulta_real(tmp_path: Path) -> None:
    """Sobe o serviço a partir de um mapa em disco e consulta via HTTP."""
    handler = build_single_map_service("corridor-02", _write_map(tmp_path))
    server = serve(handler, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address, timeout=5)
        body = json.dumps({"text": "porta", "limit": 5}).encode("utf-8")
        connection.request("POST", "/v1/maps/corridor-02/query", body=body)
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        assert payload["status"] == "ok"
        assert payload["results"][0]["geometry_id"] == "geom-1"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# Um mapa sem embeddings publicados ainda precisa subir o serviço: a
# ausência da capability é um resultado válido, não uma falha de composição.
def test_build_single_map_service_sem_embeddings(tmp_path: Path) -> None:
    """Compõe o serviço normalmente quando o mapa não publicou embeddings."""
    map_path = tmp_path / "context.json"
    map_path.write_text(json.dumps({"points": []}), encoding="utf-8")
    handler = build_single_map_service("corridor-02", map_path)
    server = serve(handler, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address, timeout=5)
        body = json.dumps({"text": "porta"}).encode("utf-8")
        connection.request("POST", "/v1/maps/corridor-02/query", body=body)
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["status"] == "capability_unavailable"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
