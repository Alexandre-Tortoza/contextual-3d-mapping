"""Fake determinístico e livre de GPU para a fronteira de extração de features densas (#161)."""

from __future__ import annotations

import numpy as np

from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload

_DIMENSION = 4  # média de R, média de G, média de B, termo de bias constante.


# Implementação fake do port DenseFeatureExtractor. Existe para permitir
# testar e desenvolver o pipeline sem GPU nem modelo real: produz sinal
# real (não aleatório) a partir dos pixels, mas sem qualquer dependência
# de ML, servindo de substituto até o adapter real (#187) estar pronto.
class FakeDenseFeatureExtractor:
    """Faz average-pooling dos pixels brutos em um grid pequeno: sinal real, zero dependências de ML."""

    # Extrai o FeatureMap fake a partir da imagem, com resolução de grid
    # limitada pelas dimensões da imagem e por config.feature_resolution.
    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        grid_h = max(1, min(config.feature_resolution, image.height))
        grid_w = max(1, min(config.feature_resolution, image.width))
        stride_y = image.height / grid_h
        stride_x = image.width / grid_w

        data = np.empty((grid_h, grid_w, _DIMENSION), dtype=np.float64)
        for gy in range(grid_h):
            y0, y1 = int(gy * stride_y), max(int(gy * stride_y) + 1, int((gy + 1) * stride_y))
            for gx in range(grid_w):
                x0, x1 = int(gx * stride_x), max(int(gx * stride_x) + 1, int((gx + 1) * stride_x))
                cell = image.pixels[y0:y1, x0:x1].astype(np.float64) / 255.0
                mean_rgb = cell.reshape(-1, 3).mean(axis=0)
                data[gy, gx] = (*mean_rgb, 1.0)

        return FeatureMap(
            data=data, stride_x=stride_x, stride_y=stride_y, dimension=_DIMENSION, model_id="fake-features"
        )
