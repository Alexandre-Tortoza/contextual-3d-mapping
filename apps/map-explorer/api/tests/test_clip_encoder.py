"""Testes do adapter que projeta texto no espaço CLIP publicado por um mapa."""

from __future__ import annotations

import pytest
from map_explorer_api.clip_encoder import (
    IncompatibleEmbeddingSpaceError,
    MapEmbeddingSpace,
    build_text_encoder,
)


# O backend "fake" é determinístico e livre de GPU, exatamente o que este
# teste precisa para validar a fronteira sem carregar um modelo CLIP real.
def _fake_space(dimension: int = 8) -> MapEmbeddingSpace:
    """Cria a identidade de um mapa publicado com o backend fake."""
    return MapEmbeddingSpace(
        embedding_space="fake:none:language_aligned", dimension=dimension,
        producer="language_embedding:fake:none", normalized=True,
    )


# O caminho comum: o produtor publicado é reconhecido e o encoder devolve um
# vetor determinístico e finito na dimensão declarada pelo mapa.
def test_build_text_encoder_projeta_no_espaco_declarado() -> None:
    """Confere que o encoder resultante devolve vetores compatíveis com o mapa."""
    encode_text = build_text_encoder(_fake_space())
    vector = encode_text("porta de emergência")
    assert len(vector) == 8
    assert all(value == value for value in vector)
    # Determinismo: a mesma query produz sempre o mesmo vetor.
    assert encode_text("porta de emergência") == vector


# Um produtor fora do formato ``language_embedding:{backend}:{checkpoint}``
# não tem como ser reconstruído, e não deve silenciosamente virar um encoder
# incorreto.
def test_recusa_produtor_com_formato_desconhecido() -> None:
    """Recusa construir o encoder quando o produtor é irreconhecível."""
    space = MapEmbeddingSpace(
        embedding_space="clip:vit-b-32:language_aligned", dimension=4,
        producer="clip:vit-b-32", normalized=True,
    )
    with pytest.raises(IncompatibleEmbeddingSpaceError, match="producer"):
        build_text_encoder(space)


# O espaço declarado precisa corresponder exatamente ao backend/checkpoint do
# produtor: um mapa corrompido ou publicado por outra versão da fusão não
# pode comparar vetores de espaços diferentes como se fossem compatíveis.
def test_recusa_espaco_declarado_incompativel_com_o_produtor() -> None:
    """Recusa quando ``embedding_space`` diverge do backend/checkpoint do produtor."""
    space = MapEmbeddingSpace(
        embedding_space="clip:vit-b-32:language_aligned", dimension=8,
        producer="language_embedding:fake:none", normalized=True,
    )
    with pytest.raises(IncompatibleEmbeddingSpaceError, match="expected"):
        build_text_encoder(space)
