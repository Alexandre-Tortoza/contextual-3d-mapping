"""Testes do contract de normalização dos embeddings de região (#243)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.embeddings import LanguageEmbedding, VisualEmbedding
from visual_perception.infrastructure.adapters.language_embedding_backend import _to_vector
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder


# Constrói um embedding visual com o vetor e a declaração pedidos.
def _visual(vector: tuple[float, ...], *, normalized: bool) -> VisualEmbedding:
    return VisualEmbedding("visual-a", "region-a", vector, len(vector), "mean", "2x2", "model", normalized)


# Constrói um embedding alinhado a linguagem com o vetor e a declaração pedidos.
def _language(vector: tuple[float, ...], *, normalized: bool) -> LanguageEmbedding:
    return LanguageEmbedding("language-a", "region-a", vector, len(vector), "model", "ckpt", normalized)


# Regressão da #243: um vetor de norma 5 era aceito como normalizado, e todo
# consumidor que confia na declaração calculava "cossenos" que não eram.
@pytest.mark.parametrize("build", [_visual, _language])
def test_a_vector_declared_normalized_must_have_unit_norm(build) -> None:
    """Norma diferente de 1 com ``normalized=True`` falha; sem a declaração, passa."""
    with pytest.raises(ValueError, match="declared normalized"):
        build((3.0, 4.0), normalized=True)
    assert build((3.0, 4.0), normalized=False).vector == (3.0, 4.0)
    assert build((0.6, 0.8), normalized=True).normalized


# O arredondamento de um vetor float32 normalizado não pode ser confundido com
# um vetor que não foi normalizado.
def test_float32_rounding_of_a_unit_vector_is_accepted() -> None:
    """Um vetor unitário calculado em float32 continua aceito como normalizado."""
    vector = np.random.default_rng(7).normal(size=768).astype(np.float32)
    vector /= np.linalg.norm(vector)
    assert _language(tuple(float(value) for value in vector), normalized=True).normalized


# Fake e adapter real precisam cumprir a mesma regra do port: normalizar só
# quando a config pede. Antes o fake normalizava sempre e escondia o caminho
# ``normalize=False`` dos testes.
@pytest.mark.parametrize("normalize", [True, False])
def test_fake_and_real_encoders_follow_the_same_normalization_rule(normalize: bool) -> None:
    """Para a mesma config, fake e adapter real devolvem vetores com a mesma regra de norma."""
    config = LanguageEmbeddingConfig(backend="clip", checkpoint="weights", dimension=3, normalize=normalize)

    class TensorLike:
        """Simula a cadeia mínima de conversão de tensor usada pelo adapter."""

        def __getitem__(self, _: int) -> TensorLike:
            """Mantém o próprio objeto ao selecionar o primeiro batch."""
            return self

        def detach(self) -> TensorLike:
            """Simula a desconexão do grafo de autograd."""
            return self

        def float(self) -> TensorLike:
            """Simula a conversão para float."""
            return self

        def cpu(self) -> TensorLike:
            """Simula a transferência para CPU."""
            return self

        def numpy(self) -> np.ndarray:
            """Devolve um vetor finito de norma 5."""
            return np.asarray([3.0, 4.0, 0.0])

    real_norm = float(np.linalg.norm(_to_vector(TensorLike(), config, "clip")))
    fake_norm = float(np.linalg.norm(FakeLanguageAlignedEncoder().encode_text("wall", config)))

    assert (real_norm == pytest.approx(1.0)) is normalize
    assert (fake_norm == pytest.approx(1.0)) is normalize
