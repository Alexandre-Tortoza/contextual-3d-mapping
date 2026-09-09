"""Valores compartilhados de identidade espacial."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import TypeAlias


Vector3: TypeAlias = tuple[float, float, float]
Quaternion: TypeAlias = tuple[float, float, float, float]


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


# Valida vetores espaciais compartilhados antes que atravessem contratos de
# módulo. Existe para impedir que NaN, infinitos ou dimensões ambíguas sejam
# aceitos por geometria, estimação ou associação.
def _vector3(value: Vector3, field: str) -> Vector3:
    """Valida e normaliza um vetor tridimensional finito.

    Argumentos:
        value: valores x, y e z do vetor.
        field: nome usado na mensagem de erro.
    Retorna:
        vetor convertido para valores ``float``.
    """
    if len(value) != 3:
        raise ValueError(f"{field} must contain exactly three values.")
    normalized = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in normalized):
        raise ValueError(f"{field} must contain only finite values.")
    return normalized  # type: ignore[return-value]


# Representa um transform rígido entre frames que módulos distintos precisam
# interpretar da mesma forma. A direção é sempre ``source_frame`` para
# ``target_frame``; usado por pose, calibração e inserção no mapa.
@dataclass(frozen=True)
class RigidTransform:
    """Transform rígido de ``source_frame`` para ``target_frame``.

    Argumentos:
        source_frame: frame no qual a entrada é expressa.
        target_frame: frame no qual a saída é expressa.
        translation_m: translação em metros.
        rotation_xyzw: rotação unitária no formato x, y, z, w.
    """

    source_frame: FrameId
    target_frame: FrameId
    translation_m: Vector3
    rotation_xyzw: Quaternion

    def __post_init__(self) -> None:
        if self.source_frame == self.target_frame:
            raise ValueError("source_frame and target_frame must differ.")
        object.__setattr__(self, "translation_m", _vector3(self.translation_m, "translation_m"))
        if len(self.rotation_xyzw) != 4:
            raise ValueError("rotation_xyzw must contain exactly four values.")
        rotation = tuple(float(item) for item in self.rotation_xyzw)
        if not all(isfinite(item) for item in rotation):
            raise ValueError("rotation_xyzw must contain only finite values.")
        norm = sqrt(sum(item * item for item in rotation))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("rotation_xyzw must be unit length.")
        object.__setattr__(self, "rotation_xyzw", rotation)


# Identifica uma pose em um instante sem introduzir um tipo específico de
# estimator. Existe como entrada comum de geometric-map e sensor-association.
@dataclass(frozen=True)
class Pose:
    """Pose rígida e timestampada de um frame em relação a outro.

    Argumentos:
        transform: transform que expressa a pose.
        timestamp_ns: instante de validade em nanossegundos.
    """

    transform: RigidTransform
    timestamp_ns: int

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer.")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")
