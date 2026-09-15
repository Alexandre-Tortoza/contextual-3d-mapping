"""Testes de recorte espacial determinístico (#5)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import AxisAlignedBounds, CropConfig, PointCloud, crop_points


def _cloud() -> PointCloud:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [-1.0, -1.0, -1.0]],
    )
    return PointCloud(coordinates, FrameId("lidar"), channels={"intensity": np.array([[1.0], [2.0], [3.0], [4.0]])})


def test_crop_keeps_only_points_inside_bounds() -> None:
    bounds = AxisAlignedBounds((-0.5, -0.5, -0.5), (2.0, 2.0, 2.0))

    cropped, indices = crop_points(_cloud(), CropConfig(bounds))

    assert len(cropped) == 2
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(cropped.coordinates, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_crop_keeps_channels_aligned_with_coordinates() -> None:
    bounds = AxisAlignedBounds((-0.5, -0.5, -0.5), (2.0, 2.0, 2.0))

    cropped, _ = crop_points(_cloud(), CropConfig(bounds))

    np.testing.assert_array_equal(cropped.channels["intensity"], [[1.0], [2.0]])


def test_crop_supports_coordinate_only_samples() -> None:
    cloud = PointCloud(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]), FrameId("lidar"))
    bounds = AxisAlignedBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    cropped, indices = crop_points(cloud, CropConfig(bounds))

    assert len(cropped) == 1
    np.testing.assert_array_equal(indices, [0])


def test_crop_with_seeded_subsample_is_reproducible() -> None:
    bounds = AxisAlignedBounds((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0))
    config = CropConfig(bounds, max_points=2, seed=42)

    first_indices = crop_points(_cloud(), config)[1]
    second_indices = crop_points(_cloud(), config)[1]

    np.testing.assert_array_equal(first_indices, second_indices)
    assert len(first_indices) == 2


def test_max_points_without_seed_is_rejected() -> None:
    bounds = AxisAlignedBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    with pytest.raises(ValueError):
        CropConfig(bounds, max_points=1)


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        AxisAlignedBounds((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
