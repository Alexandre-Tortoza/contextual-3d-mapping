"""Projeção calibrada e associação de pontos LiDAR a RGB."""

from __future__ import annotations

from math import atan2, hypot, sqrt

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
def _to_camera(
    point: tuple[float, float, float], calibration: CameraLidarCalibration
) -> tuple[float, float, float]:
    """Transforma um ponto LiDAR para o frame da câmera."""
    x, y, z = point
    qx, qy, qz, qw = calibration.lidar_to_camera.rotation_xyzw
    ix, iy, iz, iw = (
        qw * x + qy * z - qz * y,
        qw * y + qz * x - qx * z,
        qw * z + qx * y - qy * x,
        -qx * x - qy * y - qz * z,
    )
    rotated = (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )
    return tuple(rotated[axis] + calibration.lidar_to_camera.translation_m[axis] for axis in range(3))


# Projeta nos modelos pinhole, fisheye equidistante ou unificado MEI, sempre
# retornando pixel inteiro por arredondamento determinístico. A formulação
# MEI segue o contract CamOdoCal usado pelos arquivos de calibração do dataset.
def _project(
    camera_point: tuple[float, float, float], calibration: CameraLidarCalibration
) -> tuple[int, int] | None:
    """Projeta coordenadas de câmera no modelo definido pela calibração."""
    x, y, z = camera_point
    if calibration.model is CameraModel.PINHOLE:
        if z <= 0:
            return None
        u, v = calibration.fx * x / z + calibration.cx, calibration.fy * y / z + calibration.cy
    elif calibration.model is CameraModel.EQUIDISTANT_FISHEYE:
        if z <= 0:
            return None
        radius = hypot(x, y)
        theta = atan2(radius, z)
        scale = theta / radius if radius else 1.0
        u, v = calibration.fx * x * scale + calibration.cx, calibration.fy * y * scale + calibration.cy
    else:
        length = sqrt(x * x + y * y + z * z)
        if length == 0.0:  # Um ponto na origem não define um raio óptico.
            return None
        normalized_x, normalized_y, normalized_z = x / length, y / length, z / length
        denominator = normalized_z + float(calibration.mirror_xi)
        if denominator <= 0.0:
            return None
        model_x, model_y = normalized_x / denominator, normalized_y / denominator
        radius_squared = model_x * model_x + model_y * model_y
        radial = (
            1.0
            + calibration.distortion_k1 * radius_squared
            + calibration.distortion_k2 * radius_squared * radius_squared
        )
        delta_x = (
            2.0 * calibration.distortion_p1 * model_x * model_y
            + calibration.distortion_p2 * (radius_squared + 2.0 * model_x * model_x)
        )
        delta_y = (
            calibration.distortion_p1 * (radius_squared + 2.0 * model_y * model_y)
            + 2.0 * calibration.distortion_p2 * model_x * model_y
        )
        distorted_x = radial * model_x + delta_x
        distorted_y = radial * model_y + delta_y
        u = calibration.fx * distorted_x + calibration.cx
        v = calibration.fy * distorted_y + calibration.cy
    return round(u), round(v)


# Associa um scan de pontos persistidos a uma imagem RGB e regiões visuais.
# É chamada pelo mapping-runtime após inserção geométrica para produzir evidência map-anchored.
def associate_points(
    points: tuple[GeometryPoint, ...],
    rgb: RgbFrame,
    calibration: CameraLidarCalibration,
    regions: tuple[VisualRegionEvidence, ...] = (),
    *,
    max_time_delta_ns: int = 50_000_000,
) -> tuple[PointVisualAssociation, ...]:
    """Projeta, filtra visibilidade e associa RGB/região a pontos persistidos.

    Argumentos:
        points: pontos persistidos derivados de um ou mais scans LiDAR.
        rgb: frame RGB sincronizado usado para colorização.
        calibration: intrínsecos e extrínseca LiDAR→câmera.
        regions: evidências visuais opcionais no mesmo frame RGB.
        max_time_delta_ns: tolerância máxima entre LiDAR e RGB em nanossegundos.
    Retorna:
        uma tentativa de associação por ponto, na ordem de entrada.
    """
    if isinstance(max_time_delta_ns, bool) or not isinstance(max_time_delta_ns, int):
        raise TypeError("max_time_delta_ns must be an integer.")
    if max_time_delta_ns < 0:
        raise ValueError("max_time_delta_ns must be non-negative.")
    geometry_ids = [point.reference.geometry_id for point in points]
    if len(geometry_ids) != len(set(geometry_ids)):
        raise ValueError("points must have unique geometry ids.")
    if calibration.lidar_to_camera.target_frame != rgb.reference.frame_id:
        raise ValueError("calibration target frame must match the RGB frame.")
    if rgb.reference.calibration_id not in (None, calibration.calibration_id):
        raise ValueError("RGB observation calibration_id does not match calibration.")
    occupied_region_pixels: set[tuple[int, int]] = set()
    region_ids: set[str] = set()
    for region in regions:
        if region.region_id in region_ids:
            raise ValueError("regions must have unique region ids.")
        region_ids.add(region.region_id)
        if any(x >= rgb.width or y >= rgb.height for x, y in region.pixels):
            raise ValueError("region pixels must stay inside the RGB frame.")
        if occupied_region_pixels.intersection(region.pixels):
            raise ValueError("visual regions must not overlap.")
        occupied_region_pixels.update(region.pixels)
    candidates: list[tuple[GeometryPoint, tuple[int, int], float]] = []
    rejected: dict[str, AssociationStatus] = {}
    for point in points:
        if calibration.lidar_to_camera.source_frame != point.source_observation.frame_id:
            raise ValueError("calibration source frame must match the LiDAR frame.")
        lidar_timestamp = point.source_observation.timestamp
        rgb_timestamp = rgb.reference.timestamp
        if lidar_timestamp.clock_id != rgb_timestamp.clock_id:
            raise ValueError("LiDAR and RGB observations must use the same clock_id.")
        if abs(lidar_timestamp.nanoseconds - rgb_timestamp.nanoseconds) > max_time_delta_ns:
            raise ValueError("LiDAR and RGB observations exceed max_time_delta_ns.")
        camera_point = _to_camera(point.source_coordinates_m, calibration)
        pixel = _project(camera_point, calibration)
        if pixel is None:
            rejected[point.reference.geometry_id] = AssociationStatus.BEHIND_CAMERA
        elif not (0 <= pixel[0] < rgb.width and 0 <= pixel[1] < rgb.height):
            rejected[point.reference.geometry_id] = AssociationStatus.OUTSIDE_IMAGE
        elif pixel not in rgb.valid_pixels:
            rejected[point.reference.geometry_id] = AssociationStatus.OUTSIDE_VALID_SUPPORT
        else:
            depth = sqrt(sum(coordinate * coordinate for coordinate in camera_point))
            candidates.append((point, pixel, depth))
    visible: dict[tuple[int, int], tuple[GeometryPoint, float]] = {}
    visible_pixel_by_geometry: dict[str, tuple[int, int]] = {}
    for point, pixel, depth in sorted(candidates, key=lambda item: (item[2], item[0].reference.geometry_id)):
        if pixel in visible:
            rejected[point.reference.geometry_id] = AssociationStatus.OCCLUDED
        else:
            visible[pixel] = (point, depth)
            visible_pixel_by_geometry[point.reference.geometry_id] = pixel
    region_by_pixel = {pixel: region for region in regions for pixel in region.pixels}
    results: list[PointVisualAssociation] = []
    for point in points:
        status = rejected.get(point.reference.geometry_id)
        if status is not None:
            results.append(
                PointVisualAssociation(
                    point.reference, point.source_observation, rgb.reference, calibration, status
                )
            )
            continue
        pixel = visible_pixel_by_geometry[point.reference.geometry_id]
        region = region_by_pixel.get(pixel)
        results.append(
            PointVisualAssociation(
                point.reference,
                point.source_observation,
                rgb.reference,
                calibration,
                AssociationStatus.ASSOCIATED,
                pixel,
                rgb.color_at(pixel),
                region.region_id if region else None,
                region.label if region else None,
                region.feature_reference if region else None,
            )
        )
    return tuple(results)
