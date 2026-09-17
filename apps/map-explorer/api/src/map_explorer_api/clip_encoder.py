"""Adapter que projeta texto no mesmo espaço CLIP publicado por um mapa (#224).

Existe porque ``query_engine`` não pode importar ``visual_perception``
(fronteira entre capacidades de busca e de percepção visual): alguém precisa
verificar, do lado da aplicação, que o backend de texto realmente reproduz o
espaço, checkpoint, dimensão e normalização que o mapa publicou antes de
qualquer vetor ser comparado por cosseno.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from visual_perception import LanguageEmbeddingConfig, ModuleConfig, create_perception_ports


# Sinaliza que o encoder de texto disponível não pode reproduzir o espaço
# declarado pelo mapa. Um erro nomeado evita que uma comparação de cosseno
# entre espaços incompatíveis pareça um resultado de busca válido.
class IncompatibleEmbeddingSpaceError(ValueError):
    """O encoder de texto disponível não corresponde ao espaço do mapa."""


# Identidade mínima e serializável do espaço CLIP que um mapa publicou,
# extraída da metadata de ``semantic_embeddings`` sem carregar nenhum vetor.
@dataclass(frozen=True)
class MapEmbeddingSpace:
    """Espaço de embedding alinhado à linguagem declarado por um mapa."""

    embedding_space: str
    dimension: int
    producer: str
    normalized: bool


# Extrai backend e checkpoint do identificador de produtor publicado pela
# fusão (``language_embedding:{backend}:{checkpoint}``, ver
# ``mapping_runtime.corridor02_context._language_embedding_reference``).
def _parse_producer(producer: str) -> tuple[str, str]:
    """Recupera backend e checkpoint de um identificador de produtor."""
    parts = producer.split(":")
    if len(parts) != 3 or parts[0] != "language_embedding":
        raise IncompatibleEmbeddingSpaceError(f"Unrecognised language embedding producer: {producer!r}.")
    return parts[1], parts[2]


# Cria o ``encode_text`` que a aplicação injeta em ``query_engine``. É a
# única fronteira que instancia um backend CLIP real a partir da identidade
# publicada pelo mapa, e falha cedo quando essa identidade não é reproduzível.
def build_text_encoder(space: MapEmbeddingSpace) -> Callable[[str], tuple[float, ...]]:
    """Cria um ``encode_text`` compatível com o espaço declarado por um mapa.

    Argumentos:
        space: identidade do espaço CLIP publicado pelo mapa consultado.
    Retorna:
        função que projeta uma query de texto no mesmo espaço.
    Levanta:
        IncompatibleEmbeddingSpaceError: se o produtor for irreconhecível, se
            o ``embedding_space`` declarado não corresponder ao backend e
            checkpoint do produtor, ou se o encoder devolver uma dimensão
            diferente da declarada pelo mapa.
    """
    backend, checkpoint = _parse_producer(space.producer)
    expected_space = f"{backend}:{checkpoint}:language_aligned"
    if space.embedding_space != expected_space:
        raise IncompatibleEmbeddingSpaceError(
            f"Map declares embedding_space {space.embedding_space!r}, "
            f"expected {expected_space!r} for producer {space.producer!r}."
        )
    config = LanguageEmbeddingConfig(
        backend=backend, checkpoint=checkpoint, dimension=space.dimension, normalize=space.normalized,
    )
    encoder = create_perception_ports(ModuleConfig(language_embedding=config)).language_encoder

    def encode_text(text: str) -> tuple[float, ...]:
        """Projeta ``text`` no espaço alinhado à linguagem publicado pelo mapa."""
        vector = encoder.encode_text(text, config)
        if len(vector) != space.dimension:
            raise IncompatibleEmbeddingSpaceError(
                f"Text encoder returned dimension {len(vector)}, expected {space.dimension}."
            )
        return vector

    return encode_text
