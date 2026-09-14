"""Testes de seleção configurável de canais de entrada do encoder (#4)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import FeatureSelectionConfig, FeatureSelectionError, PointCloud, select_features


def _cloud() -> PointCloud:
    return PointCloud(
        np.ones((4, 3)),
        FrameId("lidar"),
        channels={"color": np.full((4, 3), 2.0), "intensity": np.full((4, 1), 3.0)},
    )


def test_coordinate_only_selection_produces_zero_width_features() -> None:
    result = select_features(_cloud(), FeatureSelectionConfig())

    assert result.feature_dimension == 0
    assert result.features.shape == (4, 0)
    assert result.coordinates.shape == (4, 3)


def test_single_channel_selection() -> None:
    result = select_features(_cloud(), FeatureSelectionConfig(channels=("intensity",)))

    assert result.feature_dimension == 1
    assert np.all(result.features == 3.0)


def test_multiple_channels_are_concatenated_in_configured_order() -> None:
    result = select_features(_cloud(), FeatureSelectionConfig(channels=("intensity", "color")))

    assert result.feature_dimension == 4
    assert np.all(result.features[:, 0] == 3.0)
    assert np.all(result.features[:, 1:] == 2.0)


def test_missing_channel_raises_explicit_error() -> None:
    with pytest.raises(FeatureSelectionError):
        select_features(_cloud(), FeatureSelectionConfig(channels=("normals",)))
