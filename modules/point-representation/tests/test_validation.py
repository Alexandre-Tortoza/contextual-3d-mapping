"""Testes da política de fronteira sobre nuvens vazias (#3)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import PointCloud, PointCloudValidationError, validate_point_cloud


def test_non_empty_cloud_passes_through() -> None:
    cloud = PointCloud(np.ones((2, 3)), FrameId("lidar"))

    assert validate_point_cloud(cloud) is cloud


def test_empty_cloud_is_rejected_by_default() -> None:
    empty = PointCloud(np.zeros((0, 3)), FrameId("lidar"))

    with pytest.raises(PointCloudValidationError):
        validate_point_cloud(empty)


def test_empty_cloud_is_accepted_when_explicitly_allowed() -> None:
    empty = PointCloud(np.zeros((0, 3)), FrameId("lidar"))

    assert validate_point_cloud(empty, allow_empty=True) is empty
