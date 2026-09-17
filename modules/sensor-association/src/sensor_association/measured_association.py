"""Associação ao primeiro suporte medido e vínculo conservador de regiões."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import binary_dilation

from contextual_mapping_contracts import ObservationReference, RigidTransform

from .boundary import BoundaryPolicy
from .camera_geometry import pixel_rays, project_coordinates, transform_coordinates
from .dense_features import DenseFeatureMap
from .models import (
    AssociationStatus,
    CameraLidarCalibration,
    MapAnchoredPoint,
    PointVisualAssociation,
    RgbFrame,
    SemanticAssociationStatus,
    SurfaceAssociationEvidence,
    VisualRegionEvidence,
)
from .projector import _region_index, _result, _validate_pairing, _validate_time_delta
from .region_surface import bind_region_surface
from .surface_visibility import MeasuredSurfaceModel


# Expõe decisões por ponto e por região sem deixar caches ou objetos de runtime
# atravessarem a fronteira para a composição e para o viewer.
@dataclass(frozen=True)
class SurfaceAssociationResult:
    """Associações na ordem de entrada e diagnostics auditáveis da observação."""

    associations: tuple[PointVisualAssociation, ...]
    regions: dict[str, dict]
    visibility_counts: dict[str, int]


# Separa o raio de visibilidade do vínculo do objeto. Nenhum ponto disputado
# por subamostragem substitui a geometria completa usada nas interseções.
def associate_measured_map_points(
    points: tuple[MapAnchoredPoint, ...], rgb: RgbFrame,
    calibration: CameraLidarCalibration, map_to_camera: RigidTransform,
    surfaces: MeasuredSurfaceModel, regions: tuple[VisualRegionEvidence, ...] = (),
    *, lidar_observation: ObservationReference, max_time_delta_ns: int = 50_000_000,
    boundary_policy: BoundaryPolicy | None = None,
    dense_feature_map: DenseFeatureMap | None = None,
) -> SurfaceAssociationResult:
    """Associa RGB à superfície visível e labels somente ao suporte ancorado.

    A geometria completa e os pontos candidatos usam o mesmo mapa, frame e
    unidade (metro). A ordem, quantidade e seleção dos candidatos não mudam
    as decisões de um ponto comum. Ausência de patch provoca abstenção.
    """
    _validate_time_delta(max_time_delta_ns)
    _validate_pairing(lidar_observation, rgb, calibration, max_time_delta_ns)
    if map_to_camera.target_frame != rgb.reference.frame_id:
        raise ValueError("map_to_camera target frame must match the RGB frame.")
    if calibration.lidar_to_camera.target_frame != rgb.reference.frame_id:
        raise ValueError("calibration target frame must match the RGB frame.")
    if calibration.lidar_to_camera.source_frame != lidar_observation.frame_id:
        raise ValueError("calibration source frame must match the LiDAR frame.")
    if any(point.geometry.map_id != surfaces.geometry.map_id for point in points):
        raise ValueError("Candidates and visibility geometry must use the same map_id.")
    identifiers = [point.geometry.geometry_id for point in points]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("points must have unique geometry ids.")
    indexed = _region_index(regions, rgb)
    view = surfaces.camera_view(map_to_camera)
    policy = boundary_policy or BoundaryPolicy()
    coordinates, _ = transform_coordinates(np.array([p.coordinates_m for p in points]).reshape(-1, 3), map_to_camera)
    depth = np.linalg.norm(coordinates, axis=1)
    rays = np.divide(coordinates, depth[:, None], out=np.full_like(coordinates, np.nan), where=depth[:, None] > 0)
    uv = project_coordinates(coordinates, calibration)
    finite = np.isfinite(uv).all(axis=1)
    pixels = np.rint(np.where(finite[:, None], uv, -1)).astype(int)
    within = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < rgb.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < rgb.height)
    valid = np.zeros((rgb.height, rgb.width), dtype=bool)
    if rgb.valid_pixels:
        vx, vy = np.array(tuple(rgb.valid_pixels)).T
        valid[vy, vx] = True
    eligible = np.zeros(len(points), dtype=bool)
    eligible[within] = valid[pixels[within, 1], pixels[within, 0]]
    queried = rays.copy()
    queried[~eligible] = np.nan
    hits = view.intersect(queried)

    # Apenas a máscara e seu contorno externo precisam do raster. Este suporte
    # depende da imagem e do mapa completo, nunca da seleção do viewer.
    masks = {}
    union = np.zeros_like(valid)
    for region in regions:
        mask = np.zeros_like(valid)
        if region.pixels:
            rx, ry = np.array(tuple(region.pixels)).T
            mask[ry, rx] = True
        masks[region.region_id] = mask & valid
        union |= mask
    raster_support = binary_dilation(union, iterations=surfaces.config.contour_width_px) & valid
    yy, xx = np.nonzero(raster_support)
    raster_rays = np.full((*valid.shape, 3), np.nan)
    raster_rays[yy, xx] = pixel_rays(np.column_stack((xx, yy)), calibration)
    raster_hits = view.intersect(raster_rays[yy, xx])
    ranges = np.full(valid.shape, np.inf)
    patches = np.full(valid.shape, -1, dtype=np.int64)
    ranges[yy, xx], patches[yy, xx] = raster_hits.ranges_m, raster_hits.patch_indices
    bindings = {key: bind_region_surface(mask, ranges, patches, raster_rays, view) for key, mask in masks.items()}

    associations = []
    for number, point in enumerate(points):
        pixel = tuple(int(value) for value in pixels[number])
        status = None
        evidence = None
        if not finite[number]:
            status = AssociationStatus.BEHIND_CAMERA
        elif not within[number]:
            status = AssociationStatus.OUTSIDE_IMAGE
        elif not eligible[number]:
            status = AssociationStatus.OUTSIDE_VALID_SUPPORT
        else:
            patch = int(hits.patch_indices[number])
            first = float(hits.ranges_m[number]) if patch >= 0 else None
            tolerance = float(hits.tolerances_m[number])
            if first is None:
                status, reason = AssociationStatus.VISIBILITY_UNCONFIRMED, "no_measured_surface"
            elif depth[number] > first + tolerance:
                status, reason = AssociationStatus.OCCLUDED, "behind_first_surface"
            elif depth[number] < first - tolerance:
                status, reason = AssociationStatus.VISIBILITY_UNCONFIRMED, "point_before_measured_surface"
            else:
                reason = "first_surface_match"
            evidence = SurfaceAssociationEvidence(float(depth[number]), first, tolerance,
                int(surfaces.source_indices[patch]) if patch >= 0 else None, reason)
        result = _result(point.geometry, lidar_observation, rgb, calibration, status,
                         pixel if status is None else None, indexed.get(pixel) if status is None else None, policy,
                         dense_feature_map)
        if result.region_id is not None and evidence is not None:
            binding = bindings[result.region_id]
            x, y = pixel
            component = int(binding.components[y, x])
            supported = bool(binding.supported[y, x])
            # O pixel arredondado também deve atingir a superfície do ponto.
            # Impede transferir o suporte de um aro para o fundo na sua borda.
            raster_patch = patches[y, x]
            same_surface = False
            if raster_patch >= 0:
                normal = view.normals[raster_patch]
                incidence = abs(float(normal @ rays[number]))
                if incidence >= surfaces.config.minimum_incidence_cosine:
                    expected = float(normal @ view.centers[raster_patch]) / float(normal @ rays[number])
                    tolerance = surfaces.config.depth_tolerance_m + surfaces.config.residual_sigma_multiplier * surfaces.residuals[raster_patch] / incidence
                    same_surface = expected > 0 and abs(expected - depth[number]) <= tolerance
            supported &= bool(same_surface)
            evidence = replace(evidence, surface_support="supported" if supported else "uncertain",
                component_id=component if component >= 0 else None,
                support_reason=(binding.diagnostics["components"][component]["reason"] if same_surface and component >= 0
                                else "no_matching_raster_surface"))
            if result.label is not None and not supported:
                result = replace(result, label=None, tentative_label=result.label,
                    semantic_status=SemanticAssociationStatus.SURFACE_UNSUPPORTED)
        associations.append(replace(result, surface_evidence=evidence))
    return SurfaceAssociationResult(tuple(associations),
        {key: binding.diagnostics for key, binding in bindings.items()},
        dict(Counter(item.status.value for item in associations)))
