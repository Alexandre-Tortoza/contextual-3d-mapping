"""Diagnóstico geométrico de plausibilidade para claims semânticos 2D."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np
from scipy.spatial import cKDTree


# Mantém a natureza ampla declarada pelo produtor distinta do noun bruto. Ela
# é suficiente para checar plausibilidade sem transformar geometria em parser.
class SemanticNature(StrEnum):
    """Natureza ampla de um claim semântico."""

    SURFACE = "surface"
    OBJECT = "object"
    PART = "part"
    UNKNOWN = "unknown"


# Separa medição de política para que o default diagnóstico nunca reescreva o
# resultado 2D enquanto a referência anotada ainda está sendo construída.
class GeometryConsistencyPolicy(StrEnum):
    """Política de uso do diagnóstico geométrico na fusão."""

    DISABLED = "disabled"
    DIAGNOSTIC = "diagnostic"
    ABSTAIN = "abstain"
    DOWNRANK_CONTRADICTIONS = "downrank_contradictions"


# Reúne parâmetros reproduzíveis para a medição local sobre o mapa integral.
@dataclass(frozen=True)
class GeometryConsistencyConfig:
    """Limiares da verificação local de continuidade geométrica."""

    neighbours: int = 16
    radius_m: float = 0.35
    max_planarity_ratio: float = 0.08
    minimum_extent_m: float = 0.30
    maximum_plane_residual_m: float = 0.035
    minimum_protrusion_m: float = 0.08

    # Falha cedo para que nenhum threshold físico inválido entre em artifacts.
    def __post_init__(self) -> None:
        """Valida vizinhança, comprimentos e frações da configuração."""
        if type(self.neighbours) is not int or self.neighbours < 6:
            raise ValueError("neighbours must be an integer of at least six.")
        for name in ("radius_m", "minimum_extent_m", "maximum_plane_residual_m", "minimum_protrusion_m"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isfinite(self.max_planarity_ratio) or not 0.0 < self.max_planarity_ratio <= 1.0:
            raise ValueError("max_planarity_ratio must be in (0, 1].")


# Define uma entrada mínima que não depende do artifact, do viewer ou do VLM.
@dataclass(frozen=True)
class GeometricSemanticPoint:
    """Ponto persistente e a natureza do claim que ele carrega."""

    geometry_id: str
    coordinates_m: tuple[float, float, float]
    nature: SemanticNature
    is_boundary: bool = False

    # Recusa entradas ambíguas antes de montar a árvore espacial.
    def __post_init__(self) -> None:
        """Valida identidade, coordenadas e estado de boundary."""
        if not self.geometry_id.strip():
            raise ValueError("geometry_id must not be empty.")
        if len(self.coordinates_m) != 3 or not all(isfinite(float(value)) for value in self.coordinates_m):
            raise ValueError("coordinates_m must contain three finite values.")
        if not isinstance(self.is_boundary, bool):
            raise TypeError("is_boundary must be a boolean.")


# Publica uma conclusão explicável sem substituir o claim de origem.
@dataclass(frozen=True)
class GeometricSupport:
    """Evidência geométrica local que sustenta, contradiz ou deixa um claim em aberto."""

    state: str
    reason: str
    nature: SemanticNature
    neighbour_count: int
    planarity_ratio: float | None = None
    normal_consistency: float | None = None
    extent_m: float | None = None
    protrusion_m: float | None = None

    # Mantém `unresolved` distinto de uma contradição mensurada.
    def __post_init__(self) -> None:
        """Valida estado, contagem e medidas opcionais do diagnóstico."""
        if self.state not in {"coherent", "contradictory", "unresolved"}:
            raise ValueError("GeometricSupport.state is invalid.")
        if not self.reason.strip() or self.neighbour_count < 0:
            raise ValueError("GeometricSupport requires a reason and non-negative neighbour count.")
        for name in ("planarity_ratio", "normal_consistency", "extent_m", "protrusion_m"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when present.")


# Mede PCA local no mapa integral para derivar plano, extensão, normal e
# desvio da superfície. É chamada pelo runtime após a fusão de labels.
def measure_geometric_support(
    points: tuple[GeometricSemanticPoint, ...], config: GeometryConsistencyConfig | None = None,
) -> dict[str, GeometricSupport]:
    """Diagnostica a compatibilidade entre natureza semântica e geometria local.

    Argumentos:
        points: pontos contextuais do mesmo mapa persistente.
        config: limiares reproduzíveis para a medição local.
    Retorna:
        diagnóstico por identidade geométrica, independente da ordem de entrada.
    """
    settings = config or GeometryConsistencyConfig()
    if len({point.geometry_id for point in points}) != len(points):
        raise ValueError("points must have unique geometry ids.")
    if not points:
        return {}
    coordinates = np.asarray([point.coordinates_m for point in points], dtype=np.float64)
    tree = cKDTree(coordinates)
    normals: list[np.ndarray | None] = []
    descriptors: list[tuple[tuple[int, ...], float | None, float | None, float | None]] = []
    for coordinate in coordinates:
        nearby = tree.query_ball_point(coordinate, settings.radius_m)
        nearby = sorted(nearby, key=lambda item: (float(np.linalg.norm(coordinates[item] - coordinate)), item))[:settings.neighbours]
        if len(nearby) < 6:
            normals.append(None)
            descriptors.append((tuple(nearby), None, None, None))
            continue
        sample = coordinates[nearby]
        centered = sample - sample.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(sample))
        total = float(eigenvalues.sum())
        normal = eigenvectors[:, 0]
        planarity = float(eigenvalues[0] / total) if total > 1e-12 else None
        residual = float(np.sqrt(max(float(eigenvalues[0]), 0.0)))
        extent = float(np.max(np.linalg.norm(sample - coordinate, axis=1)))
        normals.append(normal)
        descriptors.append((tuple(nearby), planarity, extent, residual))
    result: dict[str, GeometricSupport] = {}
    for index, point in enumerate(points):
        nearby, planarity, extent, residual = descriptors[index]
        count = len(nearby)
        if point.nature is SemanticNature.UNKNOWN:
            result[point.geometry_id] = GeometricSupport("unresolved", "semantic_nature_unknown", point.nature, count)
            continue
        if point.is_boundary:
            result[point.geometry_id] = GeometricSupport("unresolved", "boundary_association", point.nature, count, planarity, None, extent, residual)
            continue
        if planarity is None or extent is None or residual is None:
            result[point.geometry_id] = GeometricSupport("unresolved", "sparse_geometry", point.nature, count)
            continue
        comparable = [
            abs(float(normals[index] @ normals[neighbour]))
            for neighbour in nearby
            if neighbour != index and normals[index] is not None and normals[neighbour] is not None
        ]
        normal_consistency = float(np.mean(comparable)) if comparable else None
        plane = planarity <= settings.max_planarity_ratio and extent >= settings.minimum_extent_m and residual <= settings.maximum_plane_residual_m
        if point.nature is SemanticNature.SURFACE:
            state, reason = ("coherent", "continuous_planar_surface") if plane else ("unresolved", "surface_not_measurably_planar")
        elif point.nature is SemanticNature.OBJECT:
            if plane:
                state, reason = "contradictory", "object_claim_on_continuous_plane"
            elif residual >= settings.minimum_protrusion_m:
                state, reason = "coherent", "local_geometric_protrusion"
            else:
                state, reason = "unresolved", "object_geometry_ambiguous"
        else:
            state, reason = ("coherent", "part_on_local_protrusion") if residual >= settings.minimum_protrusion_m else ("unresolved", "part_geometry_ambiguous")
        result[point.geometry_id] = GeometricSupport(state, reason, point.nature, count, planarity, normal_consistency, extent, residual)
    return result
