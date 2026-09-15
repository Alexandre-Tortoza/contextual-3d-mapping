"""Projeção vetorizada e raios ópticos para consultas à geometria medida."""

from __future__ import annotations

import numpy as np

from .models import CameraLidarCalibration, CameraModel


# Aplica Brown–Conrady no plano unificado MEI. A mesma expressão é usada na
# projeção e na inversão numérica, preservando uma única convenção de pixels.
def _distort(xy: np.ndarray, calibration: CameraLidarCalibration) -> np.ndarray:
    """Retorna coordenadas distorcidas normalizadas, sem arredondar pixels."""
    x, y = xy.T
    r2 = x * x + y * y
    radial = 1 + calibration.distortion_k1 * r2 + calibration.distortion_k2 * r2 * r2
    return np.column_stack((
        x * radial + 2 * calibration.distortion_p1 * x * y + calibration.distortion_p2 * (r2 + 2 * x * x),
        y * radial + calibration.distortion_p1 * (r2 + 2 * y * y) + 2 * calibration.distortion_p2 * x * y,
    ))


# Mantém a transformação matricial local à associação. Evita construir um
# objeto Python por ponto quando a origem tem milhões de amostras.
def transform_coordinates(coordinates: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Transforma XYZ e retorna também a rotação, para transformar normais."""
    x, y, z, w = transform.rotation_xyzw
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return np.asarray(coordinates) @ rotation.T + np.asarray(transform.translation_m), rotation


# Produz a mesma projeção do caminho escalar para a nuvem completa. Pixels
# não físicos ficam em NaN e nunca são usados como índices de raster.
def project_coordinates(coordinates: np.ndarray, calibration: CameraLidarCalibration) -> np.ndarray:
    """Projeta XYZ de câmera em UV contínuo, usando metros e pixels top-left."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    length = np.linalg.norm(xyz, axis=1)
    usable = np.isfinite(xyz).all(axis=1) & (length > 0)
    if calibration.front_hemisphere_only:
        usable &= xyz[:, 2] > 0
    uv = np.full((len(xyz), 2), np.nan)
    selected = xyz[usable]
    if calibration.model is CameraModel.PINHOLE:
        plane = selected[:, :2] / selected[:, 2, None]
    elif calibration.model is CameraModel.EQUIDISTANT_FISHEYE:
        radius = np.linalg.norm(selected[:, :2], axis=1)
        scale = np.divide(np.arctan2(radius, selected[:, 2]), radius,
                          out=np.ones_like(radius), where=radius > 0)
        plane = selected[:, :2] * scale[:, None]
    else:
        normal = selected / length[usable, None]
        denominator = normal[:, 2] + calibration.mirror_xi
        with np.errstate(divide="ignore", invalid="ignore"):
            plane = _distort(normal[:, :2] / denominator[:, None], calibration)
        plane[denominator <= 0] = np.nan
    uv[usable] = plane * (calibration.fx, calibration.fy) + (calibration.cx, calibration.cy)
    return uv


# Constrói o raio do pixel no modelo calibrado. A inversão MEI resolve a
# distorção numericamente e escolhe a raiz frontal da esfera unificada.
def pixel_rays(pixels: np.ndarray, calibration: CameraLidarCalibration) -> np.ndarray:
    """Retorna direções unitárias em câmera; inversões inválidas ficam em NaN."""
    normalized = (np.asarray(pixels, dtype=np.float64) - (calibration.cx, calibration.cy)) / (calibration.fx, calibration.fy)
    if calibration.model is CameraModel.PINHOLE:
        rays = np.column_stack((normalized, np.ones(len(normalized))))
    elif calibration.model is CameraModel.EQUIDISTANT_FISHEYE:
        theta = np.linalg.norm(normalized, axis=1)
        scale = np.divide(np.sin(theta), theta, out=np.ones_like(theta), where=theta > 0)
        rays = np.column_stack((normalized * scale[:, None], np.cos(theta)))
    else:
        plane = normalized.copy()
        epsilon = 1e-6
        for _ in range(20):
            value = _distort(plane, calibration)
            residual = value - normalized
            if np.max(np.abs(residual), initial=0) < 1e-10:
                break
            dx = (_distort(plane + (epsilon, 0), calibration) - value) / epsilon
            dy = (_distort(plane + (0, epsilon), calibration) - value) / epsilon
            determinant = dx[:, 0] * dy[:, 1] - dy[:, 0] * dx[:, 1]
            stable = np.abs(determinant) > 1e-12
            update = np.zeros_like(plane)
            update[stable, 0] = (residual[stable, 0] * dy[stable, 1] - residual[stable, 1] * dy[stable, 0]) / determinant[stable]
            update[stable, 1] = (dx[stable, 0] * residual[stable, 1] - dx[stable, 1] * residual[stable, 0]) / determinant[stable]
            plane -= np.clip(update, -0.25, 0.25)
        r2 = np.sum(plane * plane, axis=1)
        xi = float(calibration.mirror_xi)
        discriminant = 1 + (1 - xi * xi) * r2
        factor = (xi + np.sqrt(np.maximum(discriminant, 0))) / (1 + r2)
        rays = np.column_stack((plane * factor[:, None], factor - xi))
        invalid = (discriminant < 0) | (np.max(np.abs(_distort(plane, calibration) - normalized), axis=1) > 1e-7)
        rays[invalid] = np.nan
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    if calibration.front_hemisphere_only:
        rays[rays[:, 2] <= 0] = np.nan
    return rays
