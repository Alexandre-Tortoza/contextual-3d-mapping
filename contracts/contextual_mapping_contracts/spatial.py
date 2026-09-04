"""Valores compartilhados de identidade espacial."""

from __future__ import annotations

from dataclasses import dataclass


# Nome estável de um frame de coordenadas (ex: "lidar", "map", "camera_rgb").
# Existe como um contract mínimo e comparável para que módulos referenciem o
# mesmo frame sem acoplar-se à semântica do transform em si, que fica a
# cargo dos adapters.
@dataclass(frozen=True, order=True)
class FrameId:
    """Nome estável de frame de coordenadas; a semântica do transform continua a cargo do adapter."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("frame id must not be empty.")

    def __str__(self) -> str:
        return self.value
