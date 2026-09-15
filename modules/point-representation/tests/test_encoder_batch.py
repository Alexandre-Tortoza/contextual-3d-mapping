"""Testes de colagem de EncoderInput por amostra em um lote (#19)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import (
    FeatureSelectionConfig,
    PointCloud,
    PointCloudValidationError,
    collate_encoder_inputs,
    select_features,
)


def _encoder_input(n: int, offset: float = 0.0):
    cloud = PointCloud(
        np.full((n, 3), offset), FrameId("lidar"), channels={"intensity": np.full((n, 1), offset)}
    )
    return select_features(cloud, FeatureSelectionConfig(channels=("intensity",)))


def test_collate_preserves_sample_boundaries() -> None:
    batch = collate_encoder_inputs([_encoder_input(2), _encoder_input(3, offset=5.0)])

    assert batch.sample_count == 2
    assert batch.sample_offsets == (0, 2, 5)
    assert batch.feature_dimension == 1


def test_collate_rejects_mismatched_selected_channels() -> None:
    with_channel = _encoder_input(2)
    without_channel = select_features(
        PointCloud(np.zeros((2, 3)), FrameId("lidar")), FeatureSelectionConfig()
    )

    with pytest.raises(PointCloudValidationError):
        collate_encoder_inputs([with_channel, without_channel])


def test_collate_rejects_empty_input_list() -> None:
    with pytest.raises(PointCloudValidationError):
        collate_encoder_inputs([])
