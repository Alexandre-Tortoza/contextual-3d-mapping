"""Amostragem de evidência visual densa em pixels já associados."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


# Representa o artifact denso no limite de sensor-association. Existe para que
# a associação leia features sem importar tipos privados de visual-perception.
@dataclass(frozen=True)
class DenseFeatureMap:
    """Mapa denso de features e sua proveniência verificável.

    Argumentos:
        values: array ``(altura, largura, dimensão)`` na grade declarada pelo transform.
        embedding_space: identidade estável do espaço de embedding.
        producer: estágio e modelo que produziram o artifact.
        artifact_reference: referência compacta para reabrir o artifact original.
        valid_support: máscara opcional que distingue cobertura ausente de vetor nulo.
    """

    values: np.ndarray
    embedding_space: str
    producer: str
    artifact_reference: str
    valid_support: np.ndarray | None = None
    stride_x: float = 1.0
    stride_y: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    # Rejeita artifacts ambíguos antes de qualquer ponto receber uma feature.
    def __post_init__(self) -> None:
        """Valida shape, valores, proveniência e suporte do mapa denso."""
        if self.values.ndim != 3 or self.values.shape[2] < 1:
            raise ValueError("DenseFeatureMap.values must have shape (height, width, dimension).")
        if not np.isfinite(self.values).all():
            raise ValueError("DenseFeatureMap.values must be finite.")
        if not all(isfinite(value) and value > 0.0 for value in (self.stride_x, self.stride_y)):
            raise ValueError("DenseFeatureMap strides must be finite and positive.")
        if not all(isfinite(value) for value in (self.origin_x, self.origin_y)):
            raise ValueError("DenseFeatureMap origins must be finite.")
        for name in ("embedding_space", "producer", "artifact_reference"):
            if not getattr(self, name).strip():
                raise ValueError(f"DenseFeatureMap.{name} must not be empty.")
        if self.valid_support is not None and (
            self.valid_support.shape != self.values.shape[:2] or self.valid_support.dtype != np.bool_
        ):
            raise ValueError("DenseFeatureMap.valid_support must match (height, width) with bool dtype.")

    @property
    def dimension(self) -> int:
        """Retorna a dimensão declarada por cada vetor do artifact."""
        return int(self.values.shape[2])


# Carrega a feature de um pixel aceito pela projeção. A ausência continua um
# resultado explícito, pois vetor zero seria confundido com evidência real.
def sample_dense_feature(
    feature_map: DenseFeatureMap, pixel: tuple[int, int], *, image_size: tuple[int, int],
) -> tuple[float, ...] | None:
    """Amostra um vetor denso no pixel RGB associado.

    Argumentos:
        feature_map: artifact denso alinhado ao frame RGB.
        pixel: coordenada inteira ``(x, y)`` validada pela projeção.
        image_size: largura e altura do frame ao qual o artifact pertence.
    Retorna:
        vetor imutável, ou ``None`` se o pixel não tiver cobertura densa.
    Levanta:
        ValueError: se o pixel estiver fora do frame RGB.
    """
    width, height = image_size
    x, y = pixel
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("pixel must stay inside the RGB frame.")
    column = round((x + 0.5 - feature_map.origin_x) / feature_map.stride_x - 0.5)
    row = round((y + 0.5 - feature_map.origin_y) / feature_map.stride_y - 0.5)
    if not (0 <= column < feature_map.values.shape[1] and 0 <= row < feature_map.values.shape[0]):
        return None
    if feature_map.valid_support is not None and not bool(feature_map.valid_support[row, column]):
        return None
    vector = tuple(float(value) for value in feature_map.values[row, column])
    if not all(isfinite(value) for value in vector):
        raise ValueError("DenseFeatureMap sample must be finite.")
    return vector
