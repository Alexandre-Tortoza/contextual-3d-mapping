"""Visibilidade por primeira interseção com patches de geometria medida."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from geometric_map import PointCloudGeometry
from scipy.spatial import cKDTree

from .camera_geometry import transform_coordinates


# Reúne parâmetros físicos e numéricos, independentes de labels, frames e
# distância ao objeto. A composição registra a configuração inteira na run.
@dataclass(frozen=True)
class SurfaceVisibilityConfig:
    """Política reproduzível para patches e suporte de regiões.

    Raios e resíduos são medidos em metros; ângulos usam cossenos.
    Os defaults representam suporte local, não alcance máximo do sensor.
    """

    neighbors: int = 16
    radius_scale: float = 0.55
    max_patch_radius_m: float = 0.15
    max_plane_residual_m: float = 0.02
    max_planarity_ratio: float = 0.12
    depth_tolerance_m: float = 0.03
    residual_sigma_multiplier: float = 3.0
    minimum_incidence_cosine: float = 0.02
    connectivity_gap_m: float = 0.10
    connectivity_normal_cosine: float = 0.85
    contour_width_px: int = 3
    minimum_anchor_pixels: int = 4
    minimum_anchor_fraction: float = 0.75

    # Impede que uma configuração inválida amplie superfícies indefinidamente.
    def __post_init__(self) -> None:
        """Valida dimensões, limites físicos e frações da política."""
        for value, minimum in ((self.neighbors, 6), (self.contour_width_px, 1), (self.minimum_anchor_pixels, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError("Contagens da política de superfície são inválidas.")
        for value in (self.radius_scale, self.max_patch_radius_m, self.max_plane_residual_m,
                      self.depth_tolerance_m, self.residual_sigma_multiplier, self.connectivity_gap_m):
            if not isfinite(value) or value <= 0:
                raise ValueError("Escalas da política de superfície devem ser finitas e positivas.")
        for value in (self.max_planarity_ratio, self.minimum_incidence_cosine,
                      self.connectivity_normal_cosine, self.minimum_anchor_fraction):
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError("Frações da política de superfície devem estar em (0, 1].")


# Expõe somente o resultado da consulta; árvores de busca e parâmetros de
# rasterização permanecem privados ao dono da visibilidade.
@dataclass(frozen=True)
class SurfaceHits:
    """Primeira superfície por raio, com infinito/-1 quando não há suporte."""

    ranges_m: np.ndarray
    patch_indices: np.ndarray
    tolerances_m: np.ndarray


# Estima orientação local uma única vez no mapa completo, sem usar os pontos
# selecionados para renderização. Cada patch é ancorado em uma amostra real.
class MeasuredSurfaceModel:
    """Patches locais limitados pela densidade e planicidade medidas."""

    # Ajusta PCA em blocos para limitar memória durante a leitura de mapas grandes.
    def __init__(self, geometry: PointCloudGeometry, config: SurfaceVisibilityConfig | None = None) -> None:
        """Estima normais e raios; vizinhanças sem plano sustentado são excluídas."""
        self.geometry = geometry
        self.config = config or SurfaceVisibilityConfig()
        xyz = geometry.coordinates_m
        tree = cKDTree(xyz)
        centers, normals, radii, residuals, indices = [], [], [], [], []
        neighbors = min(self.config.neighbors, len(xyz))
        if neighbors >= 6:
            for start in range(0, len(xyz), 16384):
                stop = min(start + 16384, len(xyz))
                distances, nearby = tree.query(xyz[start:stop], k=neighbors, workers=1)
                samples = xyz[nearby]
                centered = samples - samples.mean(axis=1, keepdims=True)
                covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                residual = np.sqrt(np.maximum(eigenvalues[:, 0], 0))
                normal = eigenvectors[:, :, 0]
                # A densidade muito desigual não autoriza cobrir todo o espaço
                # até o 16º vizinho. O terceiro vizinho limita o suporte local.
                radius = np.minimum(np.minimum(distances[:, -1] * self.config.radius_scale,
                                               distances[:, 3]), self.config.max_patch_radius_m)
                valid = ((eigenvalues[:, 1] > 1e-10)
                         & (eigenvalues[:, 0] <= self.config.max_planarity_ratio * eigenvalues.sum(axis=1))
                         & (residual <= self.config.max_plane_residual_m) & (radius > 1e-6))
                centers.append(xyz[start:stop][valid])
                normals.append(normal[valid])
                radii.append(radius[valid])
                residuals.append(residual[valid])
                indices.append(geometry.source_indices[start:stop][valid])
        self.centers = np.concatenate(centers) if centers else np.empty((0, 3))
        self.normals = np.concatenate(normals) if normals else np.empty((0, 3))
        self.radii = np.concatenate(radii) if radii else np.empty(0)
        self.residuals = np.concatenate(residuals) if residuals else np.empty(0)
        self.source_indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)

    # Publica a contagem necessária à proveniência sem expor buffers de patches
    # ao composition root que registra o custo e a densidade da representação.
    @property
    def patch_count(self) -> int:
        """Retorna quantas medições sustentam patches nesta geometria."""
        return len(self.centers)

    # Constrói uma consulta por pose sem recalcular os planos locais do mapa.
    def camera_view(self, map_to_camera) -> CameraSurfaceView:
        """Expressa patches na câmera, validando o frame de origem."""
        if map_to_camera.source_frame != self.geometry.frame_id:
            raise ValueError("O frame da geometria de oclusão difere do transform mapa→câmera.")
        return CameraSurfaceView(self, map_to_camera)


# Indexa discos por abertura angular em faixas. A consulta encontra todos os
# patches que podem intersectar o raio, mesmo com fundo angularmente mais denso.
class CameraSurfaceView:
    """Índice de raios para uma pose RGB sobre patches medidos."""

    # Mantém índices determinísticos da origem para desempatar interseções iguais.
    def __init__(self, model: MeasuredSurfaceModel, transform) -> None:
        """Transforma centros/normais e constrói índices angulares privados."""
        self.model = model
        self.frame_id = transform.target_frame
        self.centers, rotation = transform_coordinates(model.centers, transform)
        self.normals = model.normals @ rotation.T
        self.groups = []
        ranges = np.linalg.norm(self.centers, axis=1)
        valid = ranges > model.radii
        directions = np.divide(self.centers, ranges[:, None], out=np.zeros_like(self.centers), where=ranges[:, None] > 0)
        angular_radius = np.zeros(len(ranges))
        angular_radius[valid] = 2 * np.sin(np.arcsin(model.radii[valid] / ranges[valid]) / 2)
        buckets = np.ceil(np.log2(np.maximum(angular_radius, 1e-8))).astype(int)
        for bucket in sorted(set(buckets[valid].tolist())):
            selected = np.flatnonzero(valid & (buckets == bucket))
            self.groups.append((cKDTree(directions[selected]), selected, 2.0 ** bucket))

    # Usa interseção raio/plano e pertença ao disco, em vez de tomar o menor
    # depth de pixels vizinhos; planos inclinados mantêm sua profundidade correta.
    def intersect(self, rays: np.ndarray) -> SurfaceHits:
        """Retorna a primeira interseção positiva sustentada por cada raio unitário."""
        directions = np.asarray(rays, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1:] != (3,):
            raise ValueError("Raios devem formar um array N×3.")
        finite = np.isfinite(directions).all(axis=1)
        if not np.allclose(np.linalg.norm(directions[finite], axis=1), 1, atol=1e-6):
            raise ValueError("As direções de raio devem ser unitárias.")
        first = np.full(len(directions), np.inf)
        patch = np.full(len(directions), -1, dtype=np.int64)
        tolerance = np.full(len(directions), self.model.config.depth_tolerance_m)
        for start in range(0, len(directions), 1024):
            selected_rays = np.flatnonzero(finite[start:start + 1024]) + start
            if not len(selected_rays):
                continue
            for tree, selected_patches, radius in self.groups:
                neighbors = tree.query_ball_point(directions[selected_rays], radius, return_sorted=True)
                counts = np.fromiter((len(items) for items in neighbors), dtype=int, count=len(neighbors))
                if not counts.sum():
                    continue
                ray_ids = np.repeat(selected_rays, counts)
                patch_ids = selected_patches[np.concatenate([items for items in neighbors if len(items)])]
                normals = self.normals[patch_ids]
                centers = self.centers[patch_ids]
                denominator = np.einsum("ij,ij->i", normals, directions[ray_ids])
                stable = np.abs(denominator) >= self.model.config.minimum_incidence_cosine
                depth = np.divide(np.einsum("ij,ij->i", normals, centers), denominator,
                                  out=np.full(len(ray_ids), np.inf), where=stable)
                with np.errstate(invalid="ignore"):
                    offset = directions[ray_ids] * depth[:, None] - centers
                hit = stable & (depth > 0) & (np.sum(offset * offset, axis=1) <= self.model.radii[patch_ids] ** 2)
                ray_ids, patch_ids, depth = ray_ids[hit], patch_ids[hit], depth[hit]
                incidence = np.abs(denominator[hit])
                order = np.lexsort((self.model.source_indices[patch_ids], depth, ray_ids))
                ray_ids, patch_ids, depth, incidence = ray_ids[order], patch_ids[order], depth[order], incidence[order]
                unique = np.r_[True, np.diff(ray_ids) != 0] if len(ray_ids) else np.empty(0, dtype=bool)
                ray_ids, patch_ids, depth, incidence = ray_ids[unique], patch_ids[unique], depth[unique], incidence[unique]
                improve = depth < first[ray_ids]
                ray_ids, patch_ids, depth, incidence = ray_ids[improve], patch_ids[improve], depth[improve], incidence[improve]
                first[ray_ids], patch[ray_ids] = depth, patch_ids
                tolerance[ray_ids] = (self.model.config.depth_tolerance_m
                    + self.model.config.residual_sigma_multiplier * self.model.residuals[patch_ids] / incidence)
        return SurfaceHits(first, patch, tolerance)
