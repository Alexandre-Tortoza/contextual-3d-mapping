"""Testes de contract do port PointEncoder, via o encoder fake (#19)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import (
    FeatureSelectionConfig,
    PointCloud,
    PointEncoder,
    collate_encoder_inputs,
    select_features,
)
from point_representation.domain.errors import FeatureSelectionError
from point_representation.infrastructure.fakes.fake_point_encoder import FakePointEncoder


def test_fake_encoder_satisfies_the_point_encoder_protocol() -> None:
    assert isinstance(FakePointEncoder(), PointEncoder)


def test_training_and_inference_code_can_depend_on_the_port_type_only() -> None:
    def run_through_port(encoder: PointEncoder, cloud: PointCloud) -> int:
        batch = collate_encoder_inputs([select_features(cloud, FeatureSelectionConfig())])
        return encoder.encode(batch).embedding_dimension

    dimension = run_through_port(FakePointEncoder(embedding_dimension=6), PointCloud(np.zeros((3, 3)), FrameId("lidar")))

    assert dimension == 6


def test_coordinate_only_input_is_supported_when_declared() -> None:
    encoder = FakePointEncoder(required_channels=frozenset())
    cloud = PointCloud(np.ones((2, 3)), FrameId("lidar"))
    batch = collate_encoder_inputs([select_features(cloud, FeatureSelectionConfig())])

    output = encoder.encode(batch)

    assert len(output.features) == 2


def test_output_rows_correspond_to_canonical_input_point_order() -> None:
    encoder = FakePointEncoder()
    cloud = PointCloud(np.arange(9, dtype=np.float64).reshape(3, 3), FrameId("lidar"))
    batch = collate_encoder_inputs([select_features(cloud, FeatureSelectionConfig())])

    output = encoder.encode(batch)

    assert [group for group in output.point_lineage.groups] == [(0,), (1,), (2,)]


def test_missing_required_channel_raises_explicit_error() -> None:
    encoder = FakePointEncoder(required_channels=frozenset({"color"}))
    cloud = PointCloud(np.ones((2, 3)), FrameId("lidar"))
    batch = collate_encoder_inputs([select_features(cloud, FeatureSelectionConfig())])

    with pytest.raises(FeatureSelectionError):
        encoder.encode(batch)


def test_output_preserves_batch_boundaries() -> None:
    encoder = FakePointEncoder()
    clouds = [PointCloud(np.zeros((2, 3)), FrameId("lidar")), PointCloud(np.ones((3, 3)), FrameId("lidar"))]
    batch = collate_encoder_inputs([select_features(cloud, FeatureSelectionConfig()) for cloud in clouds])

    output = encoder.encode(batch)

    assert output.sample_offsets == batch.sample_offsets
