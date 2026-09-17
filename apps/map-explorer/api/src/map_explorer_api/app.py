"""Composição executável do serviço HTTP para um mapa publicado (#224).

Ponto de entrada de aplicação: escolhe a leitura concreta do artifact em
disco e o backend CLIP real, e conecta essas escolhas ao serviço genérico de
``service.py``. Um deployment com múltiplos mapas substituiria só esta
composição, sem tocar em ``service.py``.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from map_explorer_api.clip_encoder import (
    IncompatibleEmbeddingSpaceError,
    MapEmbeddingSpace,
    build_text_encoder,
)
from map_explorer_api.map_index import load_semantic_index
from map_explorer_api.service import create_app, serve


# Lê o espaço de embedding declarado por um mapa a partir da primeira entrada
# de ``semantic_embeddings``: a fusão de linguagem exige espaço/produtor
# idênticos entre geometrias do mesmo mapa, então qualquer entrada é
# representativa da identidade publicada.
def _map_embedding_space(payload: dict) -> MapEmbeddingSpace | None:
    """Extrai a identidade de espaço CLIP publicada por um mapa, se houver."""
    semantic_embeddings = payload.get("semantic_embeddings") or {}
    metadata = next(iter(semantic_embeddings.values()), None)
    if metadata is None:
        return None
    return MapEmbeddingSpace(
        embedding_space=str(metadata["embedding_space"]), dimension=int(metadata["dimension"]),
        producer=str(metadata["producer"]), normalized=bool(metadata["normalized"]),
    )


# Compõe o handler HTTP para um único mapa carregado do disco. É a
# composição padrão do serviço local: um map-explorer aberto inspeciona uma
# run ou mapa consolidado por vez.
def build_single_map_service(map_id: str, map_path: Path) -> type[BaseHTTPRequestHandler]:
    """Carrega ``map_path`` e compõe o handler HTTP para consultá-lo como ``map_id``.

    Argumentos:
        map_id: identidade pela qual o mapa é endereçado em
            ``POST /v1/maps/{map_id}/query``.
        map_path: artifact JSON do mapa (run contextual ou mapa consolidado).
    Retorna:
        classe de handler pronta para ``map_explorer_api.service.serve``.
    """
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    entries = load_semantic_index(payload)
    space = _map_embedding_space(payload)
    try:
        encode_text = build_text_encoder(space) if space is not None else None
    except IncompatibleEmbeddingSpaceError:
        encode_text = None

    return create_app(
        load_entries=lambda queried: entries if queried == map_id else None,
        load_encoder=lambda queried: encode_text if queried == map_id else None,
        embedding_space=lambda _queried: space.embedding_space if space is not None else "",
        producer=lambda _queried: space.producer if space is not None else "",
    )


# CLI mínima para subir o serviço local a partir de um artifact já publicado.
def main() -> None:
    """Sobe o serviço HTTP local de consulta textual para um único mapa."""
    parser = argparse.ArgumentParser(
        description="Serviço HTTP local de consulta textual sobre um mapa contextual publicado."
    )
    parser.add_argument("map_id", help="identidade do mapa nas rotas HTTP.")
    parser.add_argument("map_path", type=Path, help="artifact JSON do mapa a servir.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = build_single_map_service(args.map_id, args.map_path)
    server = serve(handler, host=args.host, port=args.port)
    print(f"map-explorer-api listening on http://{args.host}:{args.port} for map {args.map_id!r}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
