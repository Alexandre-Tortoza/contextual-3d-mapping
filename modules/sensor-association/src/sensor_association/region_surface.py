"""Suporte 3D de uma região, independente de sua máscara e de visibilidade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .surface_visibility import CameraSurfaceView


# Mantém o resultado do vínculo no módulo dono. Componentes sem âncora não
# herdam a identidade do objeto por serem o único retorno dentro da máscara.
@dataclass(frozen=True)
class RegionSurfaceBinding:
    """Pixels sustentados e diagnostics dos componentes medidos da região."""

    supported: np.ndarray
    components: np.ndarray
    diagnostics: dict


# Agrupa patches por continuidade e confronta seu contorno com superfícies
# vizinhas. O plano vizinho é usado para rejeitar o fundo, nunca para criar
# geometria numa abertura ou para declarar o vidro como medido.
def bind_region_surface(
    mask: np.ndarray, ranges: np.ndarray, patch_indices: np.ndarray,
    rays: np.ndarray, view: CameraSurfaceView,
) -> RegionSurfaceBinding:
    """Vincula a região apenas a componentes com âncora geométrica de contorno.

    Argumentos:
        mask: footprint semântico 2D preservado.
        ranges: primeira interseção medida por pixel, em metros.
        patch_indices: índice do patch ou -1, por pixel.
        rays: direções unitárias de câmera por pixel.
        view: patches medidos no mesmo frame da câmera.
    Retorna:
        suporte semântico por pixel, separado da visibilidade da paisagem.
    """
    config = view.model.config
    height, width = mask.shape
    valid = mask & (patch_indices >= 0) & np.isfinite(ranges)
    positions = np.flatnonzero(valid)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    supported = np.zeros_like(mask)
    if not len(positions):
        return RegionSurfaceBinding(supported, labels, {"status": "uncertain", "reason": "no_measured_surface", "components": []})
    index = np.full(mask.size, -1, dtype=np.int32)
    index[positions] = np.arange(len(positions))
    ids = index.reshape(mask.shape)
    normal = np.zeros((*mask.shape, 3))
    has_hit = patch_indices >= 0
    normal[has_hit] = view.normals[patch_indices[has_hit]]
    xyz = np.zeros_like(normal)
    xyz[has_hit] = rays[has_hit] * ranges[has_hit, None]
    rows, cols = [], []
    farther_neighbors = np.zeros(len(positions), dtype=np.int32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y1, y2 = slice(0, height - dy), slice(dy, height)
        x1 = slice(max(0, -dx), min(width, width - dx))
        x2 = slice(max(0, dx), min(width, width + dx))
        paired = valid[y1, x1] & valid[y2, x2]
        a, b = ids[y1, x1][paired], ids[y2, x2][paired]
        delta = xyz[y1, x1][paired] - xyz[y2, x2][paired]
        na, nb = normal[y1, x1][paired], normal[y2, x2][paired]
        distance = np.maximum(np.abs(np.einsum("ij,ij->i", delta, na)), np.abs(np.einsum("ij,ij->i", delta, nb)))
        coherent = (distance <= config.connectivity_gap_m) & (np.abs(np.einsum("ij,ij->i", na, nb)) >= config.connectivity_normal_cosine)
        rows.extend((a[coherent], b[coherent]))
        cols.extend((b[coherent], a[coherent]))
        ra, rb = ranges[y1, x1][paired], ranges[y2, x2][paired]
        np.add.at(farther_neighbors, a[~coherent & (rb > ra + config.connectivity_gap_m)], 1)
        np.add.at(farther_neighbors, b[~coherent & (ra > rb + config.connectivity_gap_m)], 1)
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(positions), len(positions))).tocsr()
    count, component = connected_components(graph, directed=False)
    labels.flat[positions] = component
    outside = ~mask & has_hit
    anchor_count = np.zeros(count, dtype=int)
    favorable_count = np.zeros(count, dtype=int)
    contrast_count = np.zeros(count, dtype=int)
    if outside.any():
        distance, nearest = ndimage.distance_transform_edt(~outside, return_indices=True)
        contour = valid & (distance <= config.contour_width_px)
        ys, xs = np.nonzero(contour)
        oy, ox = nearest[:, ys, xs]
        outer_normal = normal[oy, ox]
        denominator = np.einsum("ij,ij->i", outer_normal, rays[ys, xs])
        stable = np.abs(denominator) >= config.minimum_incidence_cosine
        expected = np.divide(np.einsum("ij,ij->i", outer_normal, xyz[oy, ox]), denominator,
                             out=np.full(len(ys), np.nan), where=stable)
        stable &= expected > 0
        component_ids = labels[ys[stable], xs[stable]]
        gap = ranges[ys[stable], xs[stable]] - expected[stable]
        compatible = gap <= config.connectivity_gap_m
        angle = np.abs(np.einsum("ij,ij->i", normal[ys[stable], xs[stable]], outer_normal[stable]))
        foreground = compatible & ((gap < -config.connectivity_gap_m) | (angle < config.connectivity_normal_cosine))
        np.add.at(anchor_count, component_ids, 1)
        np.add.at(favorable_count, component_ids[compatible], 1)
        np.add.at(contrast_count, component_ids[foreground], 1)
    interior_contrast = np.bincount(component, weights=farther_neighbors, minlength=count)
    anchored = ((anchor_count >= config.minimum_anchor_pixels)
                & (favorable_count >= config.minimum_anchor_fraction * anchor_count)
                & ((contrast_count >= config.minimum_anchor_pixels) | (interior_contrast >= config.minimum_anchor_pixels)))
    supported.flat[positions] = anchored[component]
    component_sizes = np.bincount(component, minlength=count)
    records = []
    for number in range(count):
        if anchored[number]:
            reason = "contour_anchored_surface"
        elif anchor_count[number] and favorable_count[number] < config.minimum_anchor_fraction * anchor_count[number]:
            reason = "background_behind_contour"
        else:
            reason = "insufficient_surface_anchor"
        records.append({"component_id": number, "pixel_count": int(component_sizes[number]),
                        "anchor_pixels": int(anchor_count[number]), "favorable_anchor_pixels": int(favorable_count[number]),
                        "foreground_anchor_pixels": int(contrast_count[number]),
                        "interior_depth_edges": int(interior_contrast[number]),
                        "status": "supported" if anchored[number] else "uncertain", "reason": reason})
    return RegionSurfaceBinding(supported, labels, {
        "status": "supported" if supported.any() else "uncertain",
        "reason": "contour_anchored_surface" if supported.any() else "insufficient_surface_anchor",
        "measured_pixels": len(positions), "supported_pixels": int(supported.sum()), "components": records,
    })
