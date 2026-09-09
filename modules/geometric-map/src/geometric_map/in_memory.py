"""Implementação determinística em memória do mapa geométrico."""

from __future__ import annotations

from dataclasses import dataclass, field

from contextual_mapping_contracts import FrameId, MapId, Provenance
from state_estimation import MotionCorrectedLidarFrame

from .models import Bounds3D, GeometryPoint, GeometryReference


# Rotaciona um vetor pelo quaternion xyzw para registrar LiDAR em mapa. Existe
# localmente porque geometric-map é dono da aplicação da pose à geometria.
def _rotate(point: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Rotaciona um ponto tridimensional por quaternion unitário."""
    x, y, z = point
    qx, qy, qz, qw = quaternion
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    return (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )


# Mantém a primeira versão simples de geometria persistente, suficiente para
# slice e testes. Índices e reconstrução avançada podem substituí-la pelo contract.
@dataclass
class InMemoryGeometricMap:
    """Mapa de pontos determinístico no frame global configurado."""

    map_id: MapId
    frame_id: FrameId
    _points: dict[str, GeometryPoint] = field(default_factory=dict)

    # Insere todos os pontos de um scan usando a pose publicada pelo estimator.
    # É chamada pelo mapping-runtime para cada frame LiDAR corrigido.
    def insert(self, frame: MotionCorrectedLidarFrame) -> tuple[GeometryReference, ...]:
        """Registra a nuvem corrigida no frame do mapa e retorna suas referências."""
        transform = frame.pose.transform
        if transform.target_frame != self.frame_id:
            raise ValueError("motion-corrected pose target frame must match map frame.")
        references: list[GeometryReference] = []
        for index, point in enumerate(frame.observation.points_m):
            rotated = _rotate(point, transform.rotation_xyzw)
            coordinates = tuple(
                rotated[axis] + transform.translation_m[axis] for axis in range(3)
            )
            geometry_id = f"{frame.observation.reference.observation_id}:{index}"
            reference = GeometryReference(self.map_id, geometry_id)
            if geometry_id in self._points:
                raise ValueError(f"duplicate geometry id {geometry_id!r}.")
            self._points[geometry_id] = GeometryPoint(
                reference, coordinates, point, frame.observation.reference, frame.provenance
            )
            references.append(reference)
        return tuple(references)

    # Expõe somente pontos dentro do volume solicitado, em ordem estável de ID.
    # É consumida por serviços de entrega de geometria e por testes de contrato.
    def lookup(self, bounds: Bounds3D) -> tuple[GeometryPoint, ...]:
        """Retorna pontos do mapa que interceptam bounds no frame do mapa."""
        if bounds.frame_id != self.frame_id:
            raise ValueError("bounds frame must match map frame.")
        return tuple(point for _, point in sorted(self._points.items()) if bounds.contains(point.coordinates_m))

    # Calcula bounds atuais sem expor o dicionário interno. Um mapa vazio não
    # inventa bounds; clientes tratam o resultado ausente explicitamente.
    def bounds(self) -> Bounds3D | None:
        """Retorna os bounds atuais ou ``None`` quando o mapa está vazio."""
        if not self._points:
            return None
        coordinates = tuple(point.coordinates_m for point in self._points.values())
        return Bounds3D(
            self.frame_id,
            tuple(min(point[axis] for point in coordinates) for axis in range(3)),
            tuple(max(point[axis] for point in coordinates) for axis in range(3)),
        )
