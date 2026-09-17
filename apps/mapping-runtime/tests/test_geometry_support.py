"""Integração do diagnóstico geométrico ao artifact contextual."""

from __future__ import annotations

from mapping_runtime.corridor02_context import _apply_geometric_support
from semantic_fusion import GeometryConsistencyConfig


# Monta uma parede mínima no formato persistido pelo runtime, sem depender de bag.
def _plane_points() -> list[dict]:
    """Cria contexts de plano contínuo com um objeto 2D contraditório no centro."""
    points = []
    for y in range(-2, 3):
        for z in range(-2, 3):
            centre = (y, z) == (0, 0)
            points.append({
                "geometry_id": f"p-{y}-{z}",
                "coordinates_m": (0.0, y * 0.1, z * 0.1),
                "context": {
                    "label": "cabinet" if centre else "wall",
                    "semantic_nature": "object" if centre else "surface",
                    "semantic_status": "interior",
                },
            })
    return points


# Garante que o runtime publica diagnóstico sem alterar o claim bruto no modo default.
def test_runtime_persists_geometric_contradiction_without_rewriting_label() -> None:
    """Anexa evidência geométrica compacta e preserva o label 2D."""
    points = _plane_points()
    counts = _apply_geometric_support(points, GeometryConsistencyConfig(minimum_extent_m=0.20))
    context = next(point["context"] for point in points if point["geometry_id"] == "p-0-0")
    assert context["label"] == "cabinet"
    assert context["geometric_support"]["state"] == "contradictory"
    assert counts["contradictory"] == 1
