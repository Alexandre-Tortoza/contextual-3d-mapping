"""Composição do slice geométrico e visual do M1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geometric_map import InMemoryGeometricMap
from sensor_association import (
    CameraLidarCalibration,
    RgbFrame,
    VisualRegionEvidence,
    associate_points,
)
from state_estimation import MotionCorrectedLidarFrame

from contextual_mapping_contracts import FrameId, MapId

from .slice_export import export_associated_slice


# Agrupa as entradas que cruzam a fronteira do composition root. Existe para
# tornar frames, tolerância temporal e identidade do mapa explícitos em uma
# única chamada reproduzível.
@dataclass(frozen=True)
class SliceBuildRequest:
    """Entradas públicas para construir um slice RGB–LiDAR.

    Argumentos:
        map_id: identidade do mapa de destino.
        map_frame: frame global que recebe a geometria.
        lidar_frame: scan LiDAR deskewed e sua pose.
        rgb_frame: frame RGB sincronizado.
        calibration: calibração LiDAR→câmera usada na projeção.
        regions: evidências visuais opcionais no frame RGB.
        max_time_delta_ns: tolerância de sincronização LiDAR–RGB.
    """

    map_id: MapId
    map_frame: FrameId
    lidar_frame: MotionCorrectedLidarFrame
    rgb_frame: RgbFrame
    calibration: CameraLidarCalibration
    regions: tuple[VisualRegionEvidence, ...] = ()
    max_time_delta_ns: int = 50_000_000

    # Valida a configuração de composição antes de criar estado parcial.
    def __post_init__(self) -> None:
        """Valida frame global e tolerância temporal do request."""
        if self.lidar_frame.pose.transform.target_frame != self.map_frame:
            raise ValueError("LiDAR pose target frame must match map_frame.")
        if isinstance(self.max_time_delta_ns, bool) or not isinstance(self.max_time_delta_ns, int):
            raise TypeError("max_time_delta_ns must be an integer.")
        if self.max_time_delta_ns < 0:
            raise ValueError("max_time_delta_ns must be non-negative.")


# Compõe mapa geométrico, associação RGB–LiDAR e persistência do artifact.
# É o caminho executável mínimo usado pela CLI e pelos testes end-to-end do M1.
def build_associated_slice(request: SliceBuildRequest, destination: Path) -> Path:
    """Constrói e exporta um slice RGB–LiDAR completo.

    Argumentos:
        request: entradas validadas da composição M1.
        destination: arquivo JSON de destino.
    Retorna:
        caminho do artifact exportado atomicamente.
    """
    geometric_map = InMemoryGeometricMap(request.map_id, request.map_frame)
    geometric_map.insert(request.lidar_frame)
    bounds = geometric_map.bounds()
    if bounds is None:  # pragma: no cover - LidarObservation já impede scans vazios.
        raise RuntimeError("geometric map unexpectedly remained empty.")
    points = geometric_map.lookup(bounds)
    associations = associate_points(
        points,
        request.rgb_frame,
        request.calibration,
        request.regions,
        max_time_delta_ns=request.max_time_delta_ns,
    )
    return export_associated_slice(
        destination,
        str(request.map_id),
        str(request.map_frame),
        points,
        associations,
    )
