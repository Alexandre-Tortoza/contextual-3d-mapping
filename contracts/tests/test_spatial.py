"""Testes dos contracts espaciais compartilhados."""

import pytest

from contextual_mapping_contracts import FrameId, MapId, Pose, RigidTransform


# Garante que contratos espaciais aceitem um transform não trivial e
# preservem sua direção; protege consumidores de ambiguidade de frame.
def test_rigid_transform_preserves_direction_and_pose_timestamp() -> None:
    """Aceita transform rígido unitário e pose timestampada."""
    transform = RigidTransform(FrameId("lidar"), FrameId("map"), (1, 2, 3), (0, 0, 0, 1))

    assert Pose(transform, 42).timestamp_ns == 42


# Rejeita rotações que não preservam distância, pois elas não são transforms
# rígidos válidos e corromperiam projeção ou inserção no mapa.
def test_rigid_transform_rejects_non_unit_rotation() -> None:
    """Rejeita quaternion não normalizado."""
    with pytest.raises(ValueError, match="unit length"):
        RigidTransform(FrameId("lidar"), FrameId("map"), (0, 0, 0), (0, 0, 0, 2))


# Confirma que a identidade de mapa não aceita valores vazios antes de virar
# chave de persistência ou parâmetro público de application service.
def test_map_id_rejects_empty_value() -> None:
    """Rejeita identidade de mapa vazia."""
    with pytest.raises(ValueError, match="must not be empty"):
        MapId(" ")
