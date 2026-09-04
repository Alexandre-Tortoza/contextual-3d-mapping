"""Fake determinístico e livre de GPU para a fronteira de embedding alinhado a linguagem (#163)."""

from __future__ import annotations

import hashlib

import numpy as np

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.image_payload import ImagePayload


# Implementação fake do port LanguageAlignedEncoder. Existe para permitir
# testar e desenvolver o pipeline sem GPU nem modelo real, substituindo o
# adapter real (#188) até que ele esteja pronto.
class FakeLanguageAlignedEncoder:
    """Mapeia conteúdo de imagem/texto em um espaço de dimensão fixa via hash com seed.

    Sem significado semântico real, mas determinístico, finito e com a
    dimensão correta, que é tudo que o contract de fronteira (#163) exige
    de um fake.
    """

    # Gera o embedding fake de uma imagem a partir da média de cor dos
    # pixels, usada como seed determinística.
    def encode_image(self, image: ImagePayload, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        mean_rgb = image.pixels.astype(np.float64).mean(axis=(0, 1)) / 255.0
        seed = int(hashlib.sha256(mean_rgb.tobytes()).hexdigest(), 16) % (2**32)
        return self._vector(seed, config.dimension)

    # Gera o embedding fake de um texto a partir do hash do próprio texto,
    # usado como seed determinística.
    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        return self._vector(seed, config.dimension)

    # Helper compartilhado por encode_image/encode_text: converte uma seed
    # inteira em um vetor normalizado da dimensão pedida.
    @staticmethod
    def _vector(seed: int, dimension: int) -> tuple[float, ...]:
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        return tuple(vector.tolist())
