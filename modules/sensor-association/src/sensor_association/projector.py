"""Projeção calibrada e associação de pontos LiDAR a RGB."""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, hypot, sqrt

from geometric_map import GeometryPoint, GeometryReference

from contextual_mapping_contracts import ObservationReference, RigidTransform

from .models import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    PointVisualAssociation,
    RgbFrame,
    VisualRegionEvidence,
)


# Aplica um transform rígido a um ponto. Existe localmente porque a
# transformação de correspondência multimodal é responsabilidade da associação,
# e é compartilhada pela associação por scan e pela ancorada no mapa, que
# diferem apenas no frame de partida.
def _to_camera(
    point: tuple[float, float, float], transform: RigidTransform
) -> tuple[float, float, float]:
    """Transforma um ponto para o frame de destino do transform."""
    x, y, z = point
    qx, qy, qz, qw = transform.rotation_xyzw
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
    return tuple(rotated[axis] + transform.translation_m[axis] for axis in range(3))


# Projeta nos modelos pinhole, fisheye equidistante ou unificado MEI, sempre
# retornando pixel inteiro por arredondamento determinístico. A formulação
# MEI segue o contract CamOdoCal usado pelos arquivos de calibração do dataset.
def _project(
    camera_point: tuple[float, float, float], calibration: CameraLidarCalibration
) -> tuple[int, int] | None:
    """Projeta coordenadas de câmera no modelo definido pela calibração."""
    x, y, z = camera_point
    if calibration.front_hemisphere_only and z <= 0:
        return None
    if calibration.model is CameraModel.PINHOLE:
        u, v = calibration.fx * x / z + calibration.cx, calibration.fy * y / z + calibration.cy
    elif calibration.model is CameraModel.EQUIDISTANT_FISHEYE:
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


# Valida as regiões visuais e indexa cada pixel coberto. Existe para que as duas
# fronteiras de associação apliquem exatamente a mesma regra de unicidade,
# limites e não sobreposição, sem duplicar a validação.
def _region_index(
    regions: tuple[VisualRegionEvidence, ...], rgb: RgbFrame
) -> dict[tuple[int, int], VisualRegionEvidence]:
    """Valida as regiões visuais e devolve o índice pixel→região."""
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
    return {pixel: region for region in regions for pixel in region.pixels}


# Confere a tolerância temporal antes de qualquer projeção, porque um valor
# inválido aqui produziria associações silenciosamente fora de sincronia.
def _validate_time_delta(max_time_delta_ns: int) -> None:
    """Valida o tipo e o sinal da tolerância temporal."""
    if isinstance(max_time_delta_ns, bool) or not isinstance(max_time_delta_ns, int):
        raise TypeError("max_time_delta_ns must be an integer.")
    if max_time_delta_ns < 0:
        raise ValueError("max_time_delta_ns must be non-negative.")


# Garante que geometria e imagem descrevem o mesmo instante e a mesma
# calibração antes de aceitar qualquer correspondência entre elas.
def _validate_pairing(
    lidar_observation: ObservationReference,
    rgb: RgbFrame,
    calibration: CameraLidarCalibration,
    max_time_delta_ns: int,
) -> None:
    """Valida relógio, sincronização e calibração entre LiDAR e RGB."""
    rgb_timestamp = rgb.reference.timestamp
    lidar_timestamp = lidar_observation.timestamp
    if lidar_timestamp.clock_id != rgb_timestamp.clock_id:
        raise ValueError("LiDAR and RGB observations must use the same clock_id.")
    if abs(lidar_timestamp.nanoseconds - rgb_timestamp.nanoseconds) > max_time_delta_ns:
        raise ValueError("LiDAR and RGB observations exceed max_time_delta_ns.")
    if rgb.reference.calibration_id not in (None, calibration.calibration_id):
        raise ValueError("RGB observation calibration_id does not match calibration.")


# Descarta pontos que uma superfície mais próxima já cobre, tolerando a
# esparsidade do mapa. Existe porque o z-buffer por pixel exato só é suficiente
# quando os pontos vêm de um único scan, que é naturalmente livre de oclusão a
# partir do seu próprio viewpoint. Em um mapa acumulado e subamostrado, um ponto
# de outro cômodo passa pelos vazios entre os pixels ocupados pela parede da
# frente e recebe o label do que está na imagem — na prática, o mapa é
# classificado através das paredes.
def _occluded_by_surface(
    candidates: Sequence[tuple[str, tuple[int, int], float, float]],
    cell_px: int,
    relative_tolerance: float,
    absolute_margin_m: float,
) -> set[str]:
    """Retorna as geometrias cobertas por uma superfície mais próxima.

    O buffer é agregado em células de ``cell_px`` pixels e consultado na
    vizinhança 3×3 da célula do ponto. A agregação aproxima a área que cada
    amostra de um mapa esparso realmente cobre; consultar os vizinhos evita que
    um ponto escape por estar do outro lado da fronteira de uma célula.

    A profundidade comparada é a do eixo óptico, e não a distância radial ao
    centro da câmera. Em um campo de visão largo a distância radial cresce em
    direção às bordas mesmo sobre uma superfície frontal, o que faria a
    periferia ocluir a si mesma; no eixo óptico, uma superfície frontal tem
    profundidade constante e não se autocobre.
    """
    nearest_by_cell: dict[tuple[int, int], float] = {}
    for _, (x, y), _, axial_depth in candidates:
        cell = (x // cell_px, y // cell_px)
        current = nearest_by_cell.get(cell)
        if current is None or axial_depth < current:
            nearest_by_cell[cell] = axial_depth
    occluded: set[str] = set()
    for geometry_id, (x, y), _, axial_depth in candidates:
        cell_x, cell_y = x // cell_px, y // cell_px
        nearest = axial_depth
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                neighbour = nearest_by_cell.get((cell_x + offset_x, cell_y + offset_y))
                if neighbour is not None and neighbour < nearest:
                    nearest = neighbour
        if axial_depth > nearest * (1.0 + relative_tolerance) + absolute_margin_m:
            occluded.add(geometry_id)
    return occluded


# Projeta pontos já expressos no frame da câmera, resolve oclusão e devolve o
# estado de cada geometria. É o núcleo compartilhado pelas duas fronteiras
# públicas: entre elas mudam o frame de partida dos pontos e a densidade
# esperada, que governa o teste de oclusão.
def _resolve_visibility(
    camera_points: Sequence[tuple[str, tuple[float, float, float]]],
    rgb: RgbFrame,
    calibration: CameraLidarCalibration,
    *,
    occlusion_cell_px: int = 0,
    occlusion_relative_tolerance: float = 0.0,
    occlusion_absolute_margin_m: float = 0.0,
) -> tuple[dict[str, AssociationStatus], dict[str, tuple[int, int]]]:
    """Separa pontos rejeitados de pontos visíveis com o pixel resolvido."""
    rejected: dict[str, AssociationStatus] = {}
    candidates: list[tuple[str, tuple[int, int], float, float]] = []
    for geometry_id, camera_point in camera_points:
        pixel = _project(camera_point, calibration)
        if pixel is None:
            rejected[geometry_id] = AssociationStatus.BEHIND_CAMERA
        elif not (0 <= pixel[0] < rgb.width and 0 <= pixel[1] < rgb.height):
            rejected[geometry_id] = AssociationStatus.OUTSIDE_IMAGE
        elif pixel not in rgb.valid_pixels:
            rejected[geometry_id] = AssociationStatus.OUTSIDE_VALID_SUPPORT
        else:
            depth = sqrt(sum(coordinate * coordinate for coordinate in camera_point))
            candidates.append((geometry_id, pixel, depth, camera_point[2]))
    if occlusion_cell_px > 0:
        covered = _occluded_by_surface(
            candidates,
            occlusion_cell_px,
            occlusion_relative_tolerance,
            occlusion_absolute_margin_m,
        )
        for geometry_id in covered:
            rejected[geometry_id] = AssociationStatus.OCCLUDED
        candidates = [item for item in candidates if item[0] not in covered]
    occupied: set[tuple[int, int]] = set()
    visible_pixel_by_geometry: dict[str, tuple[int, int]] = {}
    for geometry_id, pixel, _, _ in sorted(candidates, key=lambda item: (item[2], item[0])):
        if pixel in occupied:
            rejected[geometry_id] = AssociationStatus.OCCLUDED
        else:
            occupied.add(pixel)
            visible_pixel_by_geometry[geometry_id] = pixel
    return rejected, visible_pixel_by_geometry


# Monta o resultado público de um ponto. Centraliza a regra de que rejeições
# não carregam evidência visual residual e associações sempre trazem pixel, cor
# e a região que cobre aquele pixel.
def _result(
    geometry: GeometryReference,
    lidar_observation: ObservationReference,
    rgb: RgbFrame,
    calibration: CameraLidarCalibration,
    status: AssociationStatus | None,
    pixel: tuple[int, int] | None,
    region: VisualRegionEvidence | None,
) -> PointVisualAssociation:
    """Converte o estado resolvido de um ponto no contract público."""
    if status is not None:
        return PointVisualAssociation(
            geometry, lidar_observation, rgb.reference, calibration, status
        )
    assert pixel is not None
    return PointVisualAssociation(
        geometry,
        lidar_observation,
        rgb.reference,
        calibration,
        AssociationStatus.ASSOCIATED,
        pixel,
        rgb.color_at(pixel),
        region.region_id if region else None,
        region.label if region else None,
        region.feature_reference if region else None,
    )


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
    _validate_time_delta(max_time_delta_ns)
    geometry_ids = [point.reference.geometry_id for point in points]
    if len(geometry_ids) != len(set(geometry_ids)):
        raise ValueError("points must have unique geometry ids.")
    if calibration.lidar_to_camera.target_frame != rgb.reference.frame_id:
        raise ValueError("calibration target frame must match the RGB frame.")
    region_by_pixel = _region_index(regions, rgb)
    camera_points: list[tuple[str, tuple[float, float, float]]] = []
    for point in points:
        if calibration.lidar_to_camera.source_frame != point.source_observation.frame_id:
            raise ValueError("calibration source frame must match the LiDAR frame.")
        _validate_pairing(point.source_observation, rgb, calibration, max_time_delta_ns)
        camera_points.append(
            (
                point.reference.geometry_id,
                _to_camera(point.source_coordinates_m, calibration.lidar_to_camera),
            )
        )
    rejected, visible_pixel_by_geometry = _resolve_visibility(camera_points, rgb, calibration)
    results: list[PointVisualAssociation] = []
    for point in points:
        geometry_id = point.reference.geometry_id
        status = rejected.get(geometry_id)
        pixel = visible_pixel_by_geometry.get(geometry_id)
        results.append(
            _result(
                point.reference,
                point.source_observation,
                rgb,
                calibration,
                status,
                pixel,
                region_by_pixel.get(pixel) if pixel is not None else None,
            )
        )
    return tuple(results)


# Associa pontos do mapa persistente — e não um scan recém-chegado — a uma
# imagem RGB. Existe porque colorir o scan e anexá-lo ao mapa cria duas
# amostragens da mesma superfície lado a lado, das quais só uma carrega
# contexto; ancorando no mapa, cada ponto persistido recebe ou não um label.
def associate_map_points(
    points: tuple[MapAnchoredPoint, ...],
    rgb: RgbFrame,
    calibration: CameraLidarCalibration,
    map_to_camera: RigidTransform,
    regions: tuple[VisualRegionEvidence, ...] = (),
    *,
    lidar_observation: ObservationReference,
    max_time_delta_ns: int = 50_000_000,
    occlusion_cell_px: int = 3,
    occlusion_relative_tolerance: float = 0.05,
    occlusion_absolute_margin_m: float = 0.10,
) -> tuple[PointVisualAssociation, ...]:
    """Projeta o mapa persistente em um frame RGB e associa cor e região.

    O teste de oclusão agrega profundidade em células de pixels porque o mapa
    é esparso: sem isso, pontos atrás de uma superfície passam pelos vazios
    entre as amostras da superfície da frente e recebem o label dela.

    Argumentos:
        points: pontos do mapa expressos no frame do mapa.
        rgb: frame RGB usado para colorização e classificação.
        calibration: intrínsecos e calibração de origem da projeção.
        map_to_camera: transform do frame do mapa para o frame da câmera,
            derivado da pose estimada no instante da observação.
        regions: evidências visuais opcionais no mesmo frame RGB.
        lidar_observation: observação LiDAR que ancora temporalmente a pose.
        max_time_delta_ns: tolerância máxima entre a âncora e o RGB.
        occlusion_cell_px: lado da célula do buffer de profundidade, em pixels;
            zero desliga o teste e volta ao z-buffer por pixel exato.
        occlusion_relative_tolerance: folga proporcional à distância, que
            acomoda a espessura aparente de uma superfície.
        occlusion_absolute_margin_m: folga fixa somada à tolerância relativa.
    Retorna:
        uma tentativa de associação por ponto, na ordem de entrada.
    Levanta:
        ValueError: se identidades, frames, relógio ou calibração divergirem.
    """
    _validate_time_delta(max_time_delta_ns)
    geometry_ids = [point.geometry.geometry_id for point in points]
    if len(geometry_ids) != len(set(geometry_ids)):
        raise ValueError("points must have unique geometry ids.")
    if map_to_camera.target_frame != rgb.reference.frame_id:
        raise ValueError("map_to_camera target frame must match the RGB frame.")
    _validate_pairing(lidar_observation, rgb, calibration, max_time_delta_ns)
    region_by_pixel = _region_index(regions, rgb)
    camera_points = [
        (point.geometry.geometry_id, _to_camera(point.coordinates_m, map_to_camera))
        for point in points
    ]
    rejected, visible_pixel_by_geometry = _resolve_visibility(
        camera_points,
        rgb,
        calibration,
        occlusion_cell_px=occlusion_cell_px,
        occlusion_relative_tolerance=occlusion_relative_tolerance,
        occlusion_absolute_margin_m=occlusion_absolute_margin_m,
    )
    results: list[PointVisualAssociation] = []
    for point in points:
        geometry_id = point.geometry.geometry_id
        status = rejected.get(geometry_id)
        pixel = visible_pixel_by_geometry.get(geometry_id)
        results.append(
            _result(
                point.geometry,
                lidar_observation,
                rgb,
                calibration,
                status,
                pixel,
                region_by_pixel.get(pixel) if pixel is not None else None,
            )
        )
    return tuple(results)
