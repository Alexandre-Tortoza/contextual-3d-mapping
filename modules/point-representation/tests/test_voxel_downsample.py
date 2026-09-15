"""Testes de downsampling por voxel com proveniência (#6)."""

from __future__ import annotations

import numpy as np
from contextual_mapping_contracts import FrameId

from point_representation import PointCloud, VoxelDownsampleConfig, voxel_downsample


def test_duplicate_points_collapse_into_one_voxel() -> None:
    cloud = PointCloud(np.array([[0.05, 0.05, 0.05], [0.06, 0.06, 0.06]]), FrameId("lidar"))

    downsampled, groups = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=1.0))

    assert len(downsampled) == 1
    assert len(groups) == 1
    np.testing.assert_array_equal(sorted(groups[0]), [0, 1])


def test_sparse_points_far_apart_stay_separate() -> None:
    cloud = PointCloud(np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]), FrameId("lidar"))

    downsampled, groups = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=1.0))

    assert len(downsampled) == 2
    assert {len(group) for group in groups} == {1}


def test_larger_voxel_size_produces_fewer_or_equal_points() -> None:
    coordinates = np.array([[float(i), 0.0, 0.0] for i in range(20)])
    cloud = PointCloud(coordinates, FrameId("lidar"))

    small_voxels, _ = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=0.5))
    large_voxels, _ = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=50.0))

    assert len(small_voxels) == 20
    assert len(large_voxels) == 1


def test_output_coordinates_are_the_voxel_centroid() -> None:
    cloud = PointCloud(np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]), FrameId("lidar"))

    downsampled, _ = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=1.0))

    np.testing.assert_allclose(downsampled.coordinates[0], [0.1, 0.1, 0.1])


def test_auxiliary_channels_are_averaged_alongside_coordinates() -> None:
    cloud = PointCloud(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]),
        FrameId("lidar"),
        channels={"intensity": np.array([[1.0], [3.0]])},
    )

    downsampled, _ = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=1.0))

    np.testing.assert_allclose(downsampled.channels["intensity"], [[2.0]])


def test_coordinate_only_input_is_supported() -> None:
    cloud = PointCloud(np.array([[0.0, 0.0, 0.0]]), FrameId("lidar"))

    downsampled, groups = voxel_downsample(cloud, VoxelDownsampleConfig(voxel_size_m=1.0))

    assert len(downsampled) == 1
    assert downsampled.channels == {}
