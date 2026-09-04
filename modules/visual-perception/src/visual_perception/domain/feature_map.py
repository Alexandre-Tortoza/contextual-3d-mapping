"""Contract do mapa de features visuais denso.

Issue: #161.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Representa uma grade espacial de features visuais densas alinhada a uma
# imagem. Existe para dar ao pooling mask-aware (#162) a metadata de stride
# necessária para alinhar a máscara de uma região a esta grade.
@dataclass(frozen=True, eq=False)
class FeatureMap:
    """Uma grade espacial de features visuais densas alinhada a uma imagem.

    ``stride_x``/``stride_y`` descrevem quantos pixels da imagem original
    uma célula da grade de features cobre, que é exatamente a metadata de
    que o pooling mask-aware (#162) precisa para alinhar a máscara de uma
    região a esta grade.
    """

    data: np.ndarray  # shape (grid_height, grid_width, dimension)
    stride_x: float
    stride_y: float
    dimension: int
    model_id: str

    # Valida shape, consistência de dimensão, strides positivos e ausência
    # de NaN/Inf nos dados, para que consumidores downstream possam confiar
    # cegamente no FeatureMap.
    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"FeatureMap.data must have shape (H, W, D), got {self.data.shape}.")
        if self.data.shape[2] != self.dimension:
            raise ValueError(
                f"FeatureMap dimension mismatch: declared {self.dimension}, "
                f"got last axis {self.data.shape[2]}."
            )
        if self.stride_x <= 0 or self.stride_y <= 0:
            raise ValueError("FeatureMap strides must be positive.")
        if not np.isfinite(self.data).all():
            raise ValueError("FeatureMap.data must be finite (no NaN/Inf).")

    # Expõe a altura da grade de features sem expor o array numpy bruto
    # diretamente.
    @property
    def grid_height(self) -> int:
        return int(self.data.shape[0])

    # Expõe a largura da grade de features sem expor o array numpy bruto
    # diretamente.
    @property
    def grid_width(self) -> int:
        return int(self.data.shape[1])
