"""Testes de colagem (collate) de nuvens de tamanho variável em lote (#9)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import PointCloud, PointCloudValidationError, PointLineage, collate_point_clouds


def _cloud(n: int, offset: float = 0.0, with_channels: bool = True) -> PointCloud:
    coordinates = np.arange(n * 3, dtype=np.float64).reshape(n, 3) + offset
    channels = {"intensity": np.full((n, 1), offset)} if with_channels else {}
    return PointCloud(coordinates, FrameId("lidar"), channels=channels)


def test_heterogeneous_point_counts_share_one_batch() -> None:
    batch = collate_point_clouds([_cloud(2), _cloud(5, offset=100.0)])

    assert batch.sample_count == 2
    assert len(batch.coordinates) == 7
    assert batch.sample_offsets == (0, 2, 7)


def test_each_batched_point_resolves_to_its_source_sample_and_point() -> None:
    batch = collate_point_clouds([_cloud(2), _cloud(3, offset=100.0)])

    assert batch.resolve_source(0) == (0, 0)
    assert batch.resolve_source(1) == (0, 1)
    assert batch.resolve_source(2) == (1, 0)
    assert batch.resolve_source(4) == (1, 2)


def test_coordinate_only_batches_are_supported() -> None:
    batch = collate_point_clouds([_cloud(2, with_channels=False), _cloud(3, with_channels=False)])

    assert batch.channels == {}
    assert len(batch.coordinates) == 5


def test_channels_are_preserved_and_aligned() -> None:
    batch = collate_point_clouds([_cloud(2, offset=1.0), _cloud(2, offset=9.0)])

    np.testing.assert_array_equal(batch.channels["intensity"], [[1.0], [1.0], [9.0], [9.0]])


def test_incompatible_channel_layouts_are_rejected() -> None:
    with pytest.raises(PointCloudValidationError):
        collate_point_clouds([_cloud(2, with_channels=True), _cloud(2, with_channels=False)])


def test_empty_cloud_list_is_rejected() -> None:
    with pytest.raises(PointCloudValidationError):
        collate_point_clouds([])


def test_lineage_is_preserved_per_sample() -> None:
    lineage_a = PointLineage.from_indices(np.array([0, 1]))
    lineage_b = PointLineage.from_groups([np.array([0, 1]), np.array([2]), np.array([3, 4])])

    batch = collate_point_clouds([_cloud(2), _cloud(3, offset=100.0)], lineages=[lineage_a, lineage_b])

    assert batch.lineages == (lineage_a, lineage_b)


def test_lineage_with_wrong_point_count_is_rejected() -> None:
    mismatched_lineage = PointLineage.from_indices(np.array([0]))

    with pytest.raises(ValueError):
        collate_point_clouds([_cloud(2)], lineages=[mismatched_lineage])
