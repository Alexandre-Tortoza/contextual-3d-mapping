"""Contract dos parâmetros de normalização de coordenadas.

Issue: #7.

``NormalizationTransform`` é o que torna a normalização reversível: sem
preservar ``centroid``/``scale``, um consumidor não teria como devolver um
embedding ou uma predição ao frame de coordenadas original da nuvem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Preserva os parâmetros necessários para reverter uma normalização de
# coordenadas. ``scale`` é sempre o divisor efetivamente aplicado — mesmo
# quando a normalização de escala está desligada ou caiu no caso degenerado
# (nuvem sem extensão), caso em que ele vale 1.0 e a operação é a identidade.
@dataclass(frozen=True)
class NormalizationTransform:
    """Parâmetros para reverter uma normalização de coordenadas.

    Argumentos:
        centroid: centro subtraído das coordenadas antes de escalar, ou
            ``(0, 0, 0)`` quando a centralização estava desligada.
        scale: divisor aplicado às coordenadas centralizadas; ``1.0`` quando
            a normalização de escala estava desligada ou degenerada.
        centered: se a centralização estava habilitada.
        scaled: se a normalização de escala estava habilitada.
    """

    centroid: tuple[float, float, float]
    scale: float
    centered: bool
    scaled: bool

    # Valida que o centróide é finito e que a escala é positiva e finita —
    # uma escala zero ou negativa tornaria a inversão indefinida ou instável.
    def __post_init__(self) -> None:
        """Valida finitude do centróide e positividade da escala."""
        if len(self.centroid) != 3 or not all(math.isfinite(value) for value in self.centroid):
            raise ValueError(f"NormalizationTransform.centroid must be 3 finite values, got {self.centroid!r}.")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"NormalizationTransform.scale must be a positive finite value, got {self.scale}.")
