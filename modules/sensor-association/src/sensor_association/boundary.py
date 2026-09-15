"""Distância explícita à fronteira e incerteza da associação em pixels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from scipy import ndimage


# Modela a folga em pixels da associação sem apagar a máscara. A rasterização
# arredonda cada eixo em até meio pixel; o raio euclidiano máximo é sqrt(0,5).
# Erro de registro adicional só entra se for declarado pela composição.
@dataclass(frozen=True)
class BoundaryPolicy:
    """Incerteza projetada: arredondamento mais margem de registro declarada.

    ``registration_sigma_px=0`` significa que não há estimativa adicional
    disponível, não que pose/calibração sejam perfeitas. ``enabled=False``
    isola o efeito desta política na ablação do grounding.
    """

    enabled: bool = True
    rasterization_radius_px: float = sqrt(0.5)
    registration_sigma_px: float = 0.0
    sigma_multiplier: float = 2.0
    allow_legacy_discovery: bool = False

    # Rejeita medidas de incerteza não físicas antes de indexar pixels.
    def __post_init__(self) -> None:
        """Valida raio, sigma, multiplicador e controles de ablação."""
        for name in ("rasterization_radius_px", "registration_sigma_px", "sigma_multiplier"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"boundary.{name} must be finite and non-negative.")
        if type(self.enabled) is not bool or type(self.allow_legacy_discovery) is not bool:
            raise TypeError("Boundary policy switches must be boolean.")

    # Expõe a margem efetiva para constar em cada associação e nos diagnostics.
    @property
    def margin_px(self) -> float:
        """Retorna o limite explícito de distância usado para marcar boundary."""
        return self.rasterization_radius_px + self.sigma_multiplier * self.registration_sigma_px


# Marca pixels fronteiriços em zero e mede distância euclidiana aos seus
# centros. Bordas do frame e buracos contam como fronteira; a máscara não muda.
def distance_to_mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Retorna distâncias em pixels, com zero na borda e fora da máscara.

    A fronteira contém pixels com ao menos um vizinho 8 fora da máscara.
    O mapa de distâncias sempre corresponde ao footprint após ownership.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Boundary distance requires a two-dimensional boolean mask.")
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float64)
    return np.where(mask, ndimage.distance_transform_edt(~boundary), 0.0)


# Publica as contagens da mesma regra aplicada aos pontos, evitando diagnostics
# que usem outra margem ou uma erosão distinta da decisão de associação.
def boundary_diagnostics(mask: np.ndarray, policy: BoundaryPolicy) -> dict[str, float | int | bool]:
    """Mede boundary e safe interior sem modificar os pixels de entrada."""
    distance = distance_to_mask_boundary(mask)
    boundary = mask & (distance <= policy.margin_px) if policy.enabled else np.zeros_like(mask)
    area, count = int(mask.sum()), int(boundary.sum())
    return {
        "boundary_policy_enabled": policy.enabled,
        "boundary_margin_px": policy.margin_px,
        "rasterization_radius_px": policy.rasterization_radius_px,
        "registration_sigma_px": policy.registration_sigma_px,
        "boundary_pixel_count": count,
        "safe_interior_pixel_count": area - count,
        "safe_interior_ratio": (area - count) / area if area else 0.0,
    }
