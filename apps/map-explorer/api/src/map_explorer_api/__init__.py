"""Fronteira pública do serviço HTTP local de consulta textual sobre mapas."""

from .app import build_single_map_service
from .clip_encoder import IncompatibleEmbeddingSpaceError, MapEmbeddingSpace, build_text_encoder
from .map_index import load_semantic_index
from .service import create_app, serve

__all__ = [
    "IncompatibleEmbeddingSpaceError",
    "MapEmbeddingSpace",
    "build_single_map_service",
    "build_text_encoder",
    "create_app",
    "load_semantic_index",
    "serve",
]
