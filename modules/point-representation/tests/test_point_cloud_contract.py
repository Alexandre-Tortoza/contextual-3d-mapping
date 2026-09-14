"""Testes de contract de PointCloud (#1, #3)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import PointCloud


def _coords(n: int) -> np.ndarray:
    return np.arange(n * 3, dtype=np.float64).reshape(n, 3)


def test_coordinate_only_point_cloud_is_valid() -> None:
    cloud = PointCloud(_coords(4), FrameId("lidar"))

    assert len(cloud) == 4
    assert cloud.channels == {}


def test_point_cloud_with_auxiliary_channels_is_valid() -> None:
    cloud = PointCloud(
        _coords(3),
        FrameId("lidar"),
        channels={"color": np.ones((3, 3)), "intensity": np.zeros((3, 1))},
    )

    assert set(cloud.channels) == {"color", "intensity"}
    assert cloud.channels["intensity"].shape == (3, 1)


def test_wrong_coordinate_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        PointCloud(np.zeros((4, 2)), FrameId("lidar"))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_coordinates_are_rejected(bad_value: float) -> None:
    coords = _coords(3)
    coords[1, 0] = bad_value

    with pytest.raises(ValueError):
        PointCloud(coords, FrameId("lidar"))


def test_misaligned_channel_length_is_rejected() -> None:
    with pytest.raises(ValueError):
        PointCloud(_coords(4), FrameId("lidar"), channels={"color": np.ones((3, 3))})


def test_channel_with_non_finite_values_is_rejected() -> None:
    color = np.ones((4, 3))
    color[2, 1] = np.nan

    with pytest.raises(ValueError):
        PointCloud(_coords(4), FrameId("lidar"), channels={"color": color})


def test_point_cloud_arrays_are_immutable() -> None:
    cloud = PointCloud(_coords(2), FrameId("lidar"))

    with pytest.raises(ValueError):
        cloud.coordinates[0, 0] = 99.0
