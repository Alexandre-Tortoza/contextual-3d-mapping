"""Testes de contract de PointTrainingSample (#2)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import PointCloud, PointTrainingSample


def _cloud(n: int) -> PointCloud:
    return PointCloud(np.ones((n, 3)), FrameId("lidar"))


def test_coordinate_only_sample_is_valid() -> None:
    sample = PointTrainingSample(_cloud(3))

    assert sample.teacher_features is None
    assert sample.correspondence_indices is None


def test_sample_with_teacher_features_and_correspondence_is_valid() -> None:
    sample = PointTrainingSample(
        _cloud(3),
        teacher_features=np.ones((3, 5)),
        correspondence_indices=np.array([10, 11, 12]),
    )

    assert sample.teacher_features.shape == (3, 5)
    assert list(sample.correspondence_indices) == [10, 11, 12]


def test_misaligned_teacher_features_are_rejected() -> None:
    with pytest.raises(ValueError):
        PointTrainingSample(_cloud(3), teacher_features=np.ones((2, 5)))


def test_misaligned_correspondence_indices_are_rejected() -> None:
    with pytest.raises(ValueError):
        PointTrainingSample(_cloud(3), correspondence_indices=np.array([0, 1]))


def test_negative_correspondence_indices_are_rejected() -> None:
    with pytest.raises(ValueError):
        PointTrainingSample(_cloud(2), correspondence_indices=np.array([0, -1]))


def test_non_finite_teacher_features_are_rejected() -> None:
    features = np.ones((2, 3))
    features[0, 0] = np.nan

    with pytest.raises(ValueError):
        PointTrainingSample(_cloud(2), teacher_features=features)
