"""Projeção calibrada e associação de pontos LiDAR a RGB."""

from __future__ import annotations

from math import atan2, hypot

from geometric_map import GeometryPoint

from .models import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    PointVisualAssociation,
    RgbFrame,
    VisualRegionEvidence,
)


# Aplica uma extrínseca LiDAR→câmera a um ponto. Existe localmente porque a
# transformação de correspondência multimodal é responsabilidade da associação.
def _to_camera(point: tuple[float, float, float], calibration: CameraLidarCalibration) -> tuple[float, float, float]:
    """Transforma um ponto LiDAR para o frame da câmera."""
    x, y, z = point
    qx, qy, qz, qw = calibration.lidar_to_camera.rotation_xyzw
    ix, iy, iz, iw = qw*x + qy*z - qz*y, qw*y + qz*x - qx*z, qw*z + qx*y - qy*x, -qx*x - qy*y - qz*z
    rotated = (ix*qw + iw*-qx + iy*-qz - iz*-qy, iy*qw + iw*-qy + iz*-qx - ix*-qz, iz*qw + iw*-qz + ix*-qy - iy*-qx)
    return tuple(rotated[axis] + calibration.lidar_to_camera.translation_m[axis] for axis in range(3))


# Projeta em pinhole ou fisheye equidistante, sempre retornando pixel inteiro
# por arredondamento determinístico. Pontos atrás da câmera não produzem pixel.
def _project(camera_point: tuple[float, float, float], calibration: CameraLidarCalibration) -> tuple[int, int] | None:
    """Projeta coordenadas de câmera no modelo definido pela calibração."""
    x, y, z = camera_point
    if z <= 0:
        return None
    if calibration.model is CameraModel.PINHOLE:
        u, v = calibration.fx * x / z + calibration.cx, calibration.fy * y / z + calibration.cy
    else:
        radius = hypot(x, y)
        theta = atan2(radius, z)
        scale = theta / radius if radius else 1.0
        u, v = calibration.fx * x * scale + calibration.cx, calibration.fy * y * scale + calibration.cy
    return round(u), round(v)


# Associa um scan de pontos persistidos a uma imagem RGB e regiões visuais.
# É chamada pelo mapping-runtime após inserção geométrica para produzir evidência map-anchored.
def associate_points(
    points: tuple[GeometryPoint, ...],
    rgb: RgbFrame,
    calibration: CameraLidarCalibration,
    regions: tuple[VisualRegionEvidence, ...] = (),
) -> tuple[PointVisualAssociation, ...]:
    """Projeta, filtra visibilidade e associa RGB/região a pontos persistidos."""
    candidates: list[tuple[GeometryPoint, tuple[int, int], float]] = []
    rejected: dict[str, AssociationStatus] = {}
    for point in points:
        camera_point = _to_camera(point.source_coordinates_m, calibration)
        pixel = _project(camera_point, calibration)
        if pixel is None:
            rejected[point.reference.geometry_id] = AssociationStatus.BEHIND_CAMERA
        elif not (0 <= pixel[0] < rgb.width and 0 <= pixel[1] < rgb.height):
            rejected[point.reference.geometry_id] = AssociationStatus.OUTSIDE_IMAGE
        elif pixel not in rgb.valid_pixels:
            rejected[point.reference.geometry_id] = AssociationStatus.OUTSIDE_VALID_SUPPORT
        else:
            candidates.append((point, pixel, camera_point[2]))
    visible: dict[tuple[int, int], tuple[GeometryPoint, float]] = {}
    for point, pixel, depth in sorted(candidates, key=lambda item: (item[2], item[0].reference.geometry_id)):
        if pixel in visible:
            rejected[point.reference.geometry_id] = AssociationStatus.OCCLUDED
        else:
            visible[pixel] = (point, depth)
    region_by_pixel = {pixel: region for region in regions for pixel in region.pixels}
    results: list[PointVisualAssociation] = []
    for point in points:
        status = rejected.get(point.reference.geometry_id)
        if status is not None:
            results.append(PointVisualAssociation(point.reference, point.source_observation, rgb.reference, calibration, status))
            continue
        pixel = next(pixel for pixel, value in visible.items() if value[0] == point)
        region = region_by_pixel.get(pixel)
        results.append(PointVisualAssociation(point.reference, point.source_observation, rgb.reference, calibration, AssociationStatus.ASSOCIATED, pixel, rgb.color_at(pixel), region.region_id if region else None, region.label if region else None, region.feature_reference if region else None))
    return tuple(results)
