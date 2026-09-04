"""Testes de fronteira de embedding de região alinhado à linguagem (#163)."""

from __future__ import annotations

import numpy as np

from fixtures import payload_with_blobs
from visual_perception.application.language_embedding import encode_regions
from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder


# Helper que monta uma ObservedRegion mínima com uma mask quadrada, reutilizado pelos
# testes abaixo para focar no comportamento de encode_regions em vez da construção de fixture.
def _region(region_id: str, width: int, height: int) -> ObservedRegion:
    data = np.zeros((height, width), dtype=np.bool_)
    data[4:10, 4:10] = True
    mask = Mask(data, width, height)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",))


# Garante que cada embedding produzido carrega o region_id correto, tem a dimensão
# configurada, e não contém valores não-finitos (NaN/inf) que corromperiam consumidores
# downstream (ex: busca por similaridade).
def test_encode_regions_associates_embeddings_with_region_ids() -> None:
    payload = payload_with_blobs(width=32, height=32)
    region = _region("region-a", 32, 32)
    config = LanguageEmbeddingConfig(dimension=16)

    embeddings = encode_regions((region,), payload, FakeLanguageAlignedEncoder(), config)

    assert len(embeddings) == 1
    assert embeddings[0].region_id == "region-a"
    assert embeddings[0].dimension == 16
    assert np.isfinite(embeddings[0].vector).all()


# Confirma que texto e imagem são codificados no mesmo espaço vetorial (mesma dimensão)
# — invariante necessário para que embeddings de texto e de região sejam comparáveis.
def test_text_and_image_embeddings_share_dimension() -> None:
    config = LanguageEmbeddingConfig(dimension=16)
    encoder = FakeLanguageAlignedEncoder()
    text_vector = encoder.encode_text("a red box", config)
    assert len(text_vector) == config.dimension
