"""Testes de lineage de índice através do pré-processamento (#8)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import (
    CropConfig,
    PointCloud,
    PointLineage,
    VoxelDownsampleConfig,
    compose_lineage,
    crop_points,
    voxel_downsample,
)
from point_representation.domain.spatial_bounds import AxisAlignedBounds


def test_one_to_one_lineage_from_crop_indices() -> None:
    lineage = PointLineage.from_indices(np.array([3, 1, 4]))

    assert lineage.point_count == 3
    assert lineage.resolve(0) == (3,)
    assert lineage.resolve(2) == (4,)


def test_many_to_one_lineage_from_groups() -> None:
    lineage = PointLineage.from_groups([np.array([0, 2]), np.array([1])])

    assert lineage.resolve(0) == (0, 2)
    assert lineage.resolve(1) == (1,)


@pytest.mark.parametrize(
    "groups", [((),), ((-1,),), ((0, 0),)],
)
def test_invalid_groups_are_rejected(groups: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError):
        PointLineage(groups)


def test_compose_lineage_of_crop_then_downsample_resolves_to_original_indices() -> None:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [5.0, 5.0, 5.0], [0.02, 0.02, 0.02]]
    )
    cloud = PointCloud(coordinates, FrameId("lidar"))

    cropped, crop_indices = crop_points(cloud, CropConfig(AxisAlignedBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))))
    crop_lineage = PointLineage.from_indices(crop_indices)

    downsampled, groups = voxel_downsample(cropped, VoxelDownsampleConfig(voxel_size_m=1.0))
    downsample_lineage = PointLineage.from_groups(groups)

    composed = compose_lineage(crop_lineage, downsample_lineage)

    assert len(downsampled) == 1
    # Os três pontos próximos da origem (índices 0, 1, 3 na nuvem original)
    # sobrevivem ao crop e colapsam no mesmo voxel.
    assert composed.resolve(0) == (0, 1, 3)


def test_compose_lineage_rejects_out_of_range_reference() -> None:
    earlier = PointLineage.from_indices(np.array([0, 1]))
    later = PointLineage.from_groups([np.array([0, 5])])

    with pytest.raises(ValueError):
        compose_lineage(earlier, later)
