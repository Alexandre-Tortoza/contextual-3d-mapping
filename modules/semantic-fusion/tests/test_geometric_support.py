"""Testes da validação geométrica pós-percepção 2D."""

from __future__ import annotations

import pytest
from semantic_fusion import (
    GeometricSemanticPoint,
    GeometryConsistencyConfig,
    SemanticNature,
    measure_geometric_support,
)


# Cria uma parede plana densa para distinguir continuidade de uma nuvem esparsa.
def _plane(nature: SemanticNature, *, boundary: bool = False) -> tuple[GeometricSemanticPoint, ...]:
    """Cria pontos coplanares com um ponto central de natureza parametrizada."""
    points = []
    for y in range(-2, 3):
        for z in range(-2, 3):
            identifier = f"p-{y}-{z}"
            points.append(GeometricSemanticPoint(
                identifier, (0.0, y * 0.1, z * 0.1),
                nature if (y, z) == (0, 0) else SemanticNature.SURFACE,
                is_boundary=boundary and (y, z) == (0, 0),
            ))
    return tuple(points)


# Um noun de objeto sobre uma parede plana contínua deve ser explicitamente contestado.
def test_object_claim_on_continuous_plane_is_contradictory() -> None:
    """Marca objeto sem protrusão sobre parede como contraditório."""
    result = measure_geometric_support(_plane(SemanticNature.OBJECT), GeometryConsistencyConfig(minimum_extent_m=0.20))
    support = result["p-0-0"]
    assert support.state == "contradictory"
    assert support.reason == "object_claim_on_continuous_plane"


# A superfície estrutural não é promovida a objeto; o mesmo plano a sustenta.
def test_surface_claim_on_continuous_plane_is_coherent() -> None:
    """Marca uma superfície sobre plano contínuo como coerente."""
    support = measure_geometric_support(_plane(SemanticNature.SURFACE), GeometryConsistencyConfig(minimum_extent_m=0.20))["p-0-0"]
    assert support.state == "coherent"
    assert support.reason == "continuous_planar_surface"


# Um ponto deslocado para fora do plano é evidência local de geometria de objeto.
def test_protruding_object_is_coherent() -> None:
    """Aceita objeto quando a vizinhança mede protrusão suficiente."""
    points = list(_plane(SemanticNature.SURFACE))
    points = [
        GeometricSemanticPoint("p-0-0", (0.15, 0.0, 0.0), SemanticNature.OBJECT)
        if point.geometry_id == "p-0-0" else point
        for point in points
    ]
    support = measure_geometric_support(
        tuple(points), GeometryConsistencyConfig(minimum_extent_m=0.20, minimum_protrusion_m=0.02)
    )["p-0-0"]
    assert support.state == "coherent"
    assert support.reason == "local_geometric_protrusion"


# Boundary e baixa densidade não autorizam uma conclusão geométrica forte.
@pytest.mark.parametrize(
    "points, identifier, reason",
    [(_plane(SemanticNature.OBJECT, boundary=True), "p-0-0", "boundary_association"),
     ((GeometricSemanticPoint("sparse", (0.0, 0.0, 0.0), SemanticNature.OBJECT),), "sparse", "sparse_geometry")],
)
def test_boundary_and_sparse_geometry_remain_unresolved(
    points: tuple[GeometricSemanticPoint, ...], identifier: str, reason: str,
) -> None:
    """Preserva ausência de evidência como unresolved, não como contradição."""
    support = measure_geometric_support(points)[identifier]
    assert support.state == "unresolved"
    assert support.reason == reason


# Entrada inválida não pode montar um diagnóstico aparentemente confiável.
def test_invalid_geometry_config_is_rejected() -> None:
    """Recusa thresholds e vizinhanças fisicamente inválidos."""
    with pytest.raises(ValueError, match="neighbours"):
        GeometryConsistencyConfig(neighbours=5)
    with pytest.raises(ValueError, match="radius_m"):
        GeometryConsistencyConfig(radius_m=0.0)
