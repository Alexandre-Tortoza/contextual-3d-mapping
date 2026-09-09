"""Testes do mapa geométrico determinístico em memória."""

from __future__ import annotations

import pytest
from geometric_map import Bounds3D, InMemoryGeometricMap
from state_estimation import LidarObservation, MotionCorrectedLidarFrame

from contextual_mapping_contracts import (
    FrameId,
    MapId,
    ObservationReference,
    Pose,
    Provenance,
    RigidTransform,
    Timestamp,
)


# Monta um scan com rotação de 180 graus em Z e translação conhecida para
# exercitar o registro geométrico sem depender de backend externo.
def _corrected_frame() -> MotionCorrectedLidarFrame:
    """Cria uma nuvem deskewed sintética com pose válida."""
    lidar_frame = FrameId("lidar")
    reference = ObservationReference(
        "lidar-1",
        "dataset",
        "sequence",
        "lidar",
        0,
        Timestamp(10, "clock"),
        lidar_frame,
    )
    observation = LidarObservation(reference, ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    pose = Pose(
        RigidTransform(lidar_frame, FrameId("map"), (10.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)),
        10,
    )
    provenance = Provenance("fixture", (reference,))
    return MotionCorrectedLidarFrame(observation, pose, lidar_frame, provenance)


# Verifica transformação, identidade estável e lookup inclusivo no menor nível útil.
def test_insert_registers_points_and_lookup_preserves_source() -> None:
    """Registra pontos no mapa e preserva coordenadas/proveniência de origem."""
    geometric_map = InMemoryGeometricMap(MapId("map-1"), FrameId("map"))

    references = geometric_map.insert(_corrected_frame())
    points = geometric_map.lookup(Bounds3D(FrameId("map"), (8.9, -2.1, 0.9), (10.1, 0.1, 1.1)))

    assert tuple(item.geometry_id for item in references) == ("lidar-1:0", "lidar-1:1")
    assert tuple(point.coordinates_m for point in points) == (
        pytest.approx((9.0, 0.0, 1.0)),
        pytest.approx((10.0, -2.0, 1.0)),
    )
    assert points[0].source_coordinates_m == (1.0, 0.0, 0.0)


# Confirma que o mapa rejeita poses destinadas a outro frame global.
def test_insert_rejects_wrong_map_frame() -> None:
    """Rejeita scan cuja pose não termina no frame do mapa."""
    geometric_map = InMemoryGeometricMap(MapId("map-1"), FrameId("world"))

    with pytest.raises(ValueError, match="target frame"):
        geometric_map.insert(_corrected_frame())
