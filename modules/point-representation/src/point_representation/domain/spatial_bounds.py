"""Contract de limites espaciais usado pelo crop de nuvens de pontos.

Issue: #5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Representa uma caixa alinhada aos eixos em metros, usada para recortar uma
# nuvem de pontos. Existe local a este módulo (e não em
# ``contextual_mapping_contracts``) porque um recorte de treino não é um
# conceito de posse do repositório inteiro, ao contrário de ``FrameId``/``Pose``.
@dataclass(frozen=True)
class AxisAlignedBounds:
    """Limites ``[min_m, max_m]`` por eixo, em metros.

    Argumentos:
        min_m: limite inferior ``(x, y, z)``.
        max_m: limite superior ``(x, y, z)``.
    """

    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]

    # Rejeita limites com dimensão errada, valores não-finitos, ou onde o
    # mínimo excede o máximo em algum eixo — uma caixa assim nunca conteria
    # pontos e é sinal de configuração invertida, não uma região vazia válida.
    def __post_init__(self) -> None:
        """Valida dimensão, finitude e consistência ``min <= max`` por eixo."""
        for name, value in (("min_m", self.min_m), ("max_m", self.max_m)):
            if len(value) != 3 or not all(math.isfinite(component) for component in value):
                raise ValueError(f"AxisAlignedBounds.{name} must be 3 finite values, got {value!r}.")
        if any(lo > hi for lo, hi in zip(self.min_m, self.max_m, strict=True)):
            raise ValueError(
                f"AxisAlignedBounds.min_m must not exceed max_m on any axis: "
                f"min_m={self.min_m}, max_m={self.max_m}."
            )
