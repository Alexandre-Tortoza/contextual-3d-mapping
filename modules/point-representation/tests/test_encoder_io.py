"""Testes de contract de EncoderInputBatch/EncoderOutput (#19)."""

from __future__ import annotations

import numpy as np
import pytest

from point_representation import EncoderInputBatch, EncoderOutput, PointLineage


def _valid_input_batch() -> EncoderInputBatch:
    return EncoderInputBatch(
        coordinates=np.zeros((4, 3)),
        features=np.zeros((4, 2)),
        feature_dimension=2,
        selected_channels=("intensity", "color"),
        sample_offsets=(0, 2, 4),
    )


def test_valid_encoder_input_batch_round_trips_fields() -> None:
    batch = _valid_input_batch()

    assert batch.sample_count == 2
    assert batch.feature_dimension == 2


def test_encoder_input_batch_rejects_misaligned_features() -> None:
    with pytest.raises(ValueError):
        EncoderInputBatch(
            coordinates=np.zeros((4, 3)),
            features=np.zeros((3, 2)),
            feature_dimension=2,
            selected_channels=(),
            sample_offsets=(0, 4),
        )


def test_encoder_input_batch_rejects_feature_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        EncoderInputBatch(
            coordinates=np.zeros((4, 3)),
            features=np.zeros((4, 2)),
            feature_dimension=3,
            selected_channels=(),
            sample_offsets=(0, 4),
        )


def test_encoder_output_rejects_lineage_point_count_mismatch() -> None:
    lineage = PointLineage.from_indices(np.array([0, 1, 2]))

    with pytest.raises(ValueError):
        EncoderOutput(features=np.zeros((2, 8)), point_lineage=lineage, sample_offsets=(0, 2))


def test_encoder_output_rejects_non_finite_features() -> None:
    features = np.array([[1.0, np.nan]])

    with pytest.raises(ValueError):
        EncoderOutput(features=features, point_lineage=PointLineage.from_indices(np.array([0])), sample_offsets=(0, 1))


def test_encoder_output_embedding_dimension_matches_features() -> None:
    lineage = PointLineage.from_indices(np.array([0, 1]))
    output = EncoderOutput(features=np.zeros((2, 5)), point_lineage=lineage, sample_offsets=(0, 2))

    assert output.embedding_dimension == 5
