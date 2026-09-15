"""Testes de normalização reversível de coordenadas (#7)."""

from __future__ import annotations

import numpy as np
from contextual_mapping_contracts import FrameId

from point_representation import (
    NormalizationConfig,
    PointCloud,
    denormalize_coordinates,
    normalize_coordinates,
)


def _cloud() -> PointCloud:
    coordinates = np.array([[10.0, 20.0, 30.0], [12.0, 24.0, 36.0], [8.0, 16.0, 24.0]])
    return PointCloud(coordinates, FrameId("lidar"), channels={"intensity": np.array([[1.0], [2.0], [3.0]])})


def test_normalized_coordinates_are_centered_and_bounded() -> None:
    normalized, _ = normalize_coordinates(_cloud(), NormalizationConfig(center=True, scale=True))

    assert np.abs(normalized.coordinates.mean(axis=0)).max() < 1e-9
    assert np.abs(normalized.coordinates).max() <= 1.0 + 1e-9


def test_normalize_then_denormalize_recovers_original_within_tolerance() -> None:
    cloud = _cloud()

    normalized, transform = normalize_coordinates(cloud, NormalizationConfig(center=True, scale=True))
    restored = denormalize_coordinates(normalized.coordinates, transform)

    np.testing.assert_allclose(restored, cloud.coordinates, atol=1e-9)


def test_disabled_normalization_leaves_coordinates_unchanged() -> None:
    cloud = _cloud()

    normalized, transform = normalize_coordinates(cloud, NormalizationConfig(center=False, scale=False))

    np.testing.assert_array_equal(normalized.coordinates, cloud.coordinates)
    assert transform.scale == 1.0
    assert transform.centroid == (0.0, 0.0, 0.0)


def test_auxiliary_channels_are_left_unchanged() -> None:
    cloud = _cloud()

    normalized, _ = normalize_coordinates(cloud, NormalizationConfig())

    np.testing.assert_array_equal(normalized.channels["intensity"], cloud.channels["intensity"])


def test_degenerate_single_point_cloud_does_not_produce_nan_or_inf() -> None:
    cloud = PointCloud(np.array([[5.0, 5.0, 5.0]]), FrameId("lidar"))

    normalized, transform = normalize_coordinates(cloud, NormalizationConfig(center=True, scale=True))

    assert np.isfinite(normalized.coordinates).all()
    assert transform.scale == 1.0
    restored = denormalize_coordinates(normalized.coordinates, transform)
    np.testing.assert_allclose(restored, cloud.coordinates)


def test_degenerate_duplicate_points_do_not_produce_nan_or_inf() -> None:
    cloud = PointCloud(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), FrameId("lidar"))

    normalized, transform = normalize_coordinates(cloud, NormalizationConfig(center=True, scale=True))

    assert np.isfinite(normalized.coordinates).all()
    assert transform.scale == 1.0
