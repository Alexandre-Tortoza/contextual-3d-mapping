"""Testes do baseline concreto de destilação 2D->3D por ponto."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId

from point_representation import (
    FeatureSelectionConfig,
    PointCloud,
    PointCloudValidationError,
    PointEncoder,
    PointTrainingSample,
    RidgeDistillationConfig,
    RidgeDistilledPointEncoder,
    encode_point_clouds,
    fit_ridge_distilled_encoder,
)
from point_representation.domain.errors import FeatureSelectionError


# Cria uma amostra em que o professor é uma função linear conhecida das
# coordenadas e da intensidade, permitindo verificar treino e inferência sem
# GPU nem checkpoint externo.
def _sample(values: np.ndarray) -> PointTrainingSample:
    """Monta uma amostra supervisionada com um professor linear determinístico."""
    coordinates = np.column_stack([values, values * 2.0, np.ones_like(values)])
    intensity = (values * 3.0).reshape(-1, 1)
    cloud = PointCloud(coordinates, FrameId("map"), channels={"intensity": intensity})
    combined = np.concatenate([coordinates, intensity], axis=1)
    teacher = np.column_stack([combined @ np.array([2.0, -1.0, 0.5, 3.0]) + 4.0, values - 2.0])
    return PointTrainingSample(cloud, teacher_features=teacher)


def test_ridge_encoder_satisfies_the_public_port() -> None:
    encoder = fit_ridge_distilled_encoder(
        [_sample(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))],
        FeatureSelectionConfig(("intensity",)),
        RidgeDistillationConfig(0.0),
    )

    assert isinstance(encoder, PointEncoder)
    assert encoder.embedding_dimension == 2
    assert encoder.required_channels == frozenset({"intensity"})


def test_training_reconstructs_the_teacher_for_seen_linear_data() -> None:
    selection = FeatureSelectionConfig(("intensity",))
    encoder = fit_ridge_distilled_encoder(
        [_sample(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))], selection, RidgeDistillationConfig(0.0)
    )
    cloud = _sample(np.array([1.5, 2.5])).point_cloud

    result = encode_point_clouds(encoder, [cloud], selection)

    expected = _sample(np.array([1.5, 2.5])).teacher_features
    assert np.allclose(np.asarray([embedding.vector for embedding in result.embeddings]), expected)
    assert all(embedding.valid for embedding in result.embeddings)


def test_inference_preserves_multiple_cloud_boundaries() -> None:
    selection = FeatureSelectionConfig(("intensity",))
    encoder = fit_ridge_distilled_encoder([_sample(np.arange(5, dtype=np.float64))], selection)

    result = encode_point_clouds(
        encoder,
        [_sample(np.array([1.0])).point_cloud, _sample(np.array([2.0, 3.0])).point_cloud],
        selection,
    )

    assert result.sample_offsets == (0, 1, 3)
    assert [embedding.coordinates for embedding in result.embeddings] == [(1.0, 2.0, 1.0), (2.0, 4.0, 1.0), (3.0, 6.0, 1.0)]


def test_training_rejects_missing_or_incompatible_teacher_features() -> None:
    selection = FeatureSelectionConfig(("intensity",))
    unsupervised = PointTrainingSample(_sample(np.array([1.0])).point_cloud)
    incompatible = PointTrainingSample(_sample(np.array([2.0])).point_cloud, teacher_features=np.ones((1, 3)))

    with pytest.raises(PointCloudValidationError, match="teacher_features"):
        fit_ridge_distilled_encoder([unsupervised], selection)
    with pytest.raises(PointCloudValidationError, match="same embedding dimension"):
        fit_ridge_distilled_encoder([_sample(np.array([1.0])), incompatible], selection)


def test_encoder_rejects_channel_order_different_from_training() -> None:
    selection = FeatureSelectionConfig(("intensity",))
    encoder = fit_ridge_distilled_encoder([_sample(np.arange(4, dtype=np.float64))], selection)
    cloud = _sample(np.array([1.0])).point_cloud

    with pytest.raises(FeatureSelectionError, match="selected channels"):
        encode_point_clouds(encoder, [cloud], FeatureSelectionConfig())


def test_config_and_serialized_state_validation_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RidgeDistillationConfig(-1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        RidgeDistillationConfig(float("nan"))
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeDistilledPointEncoder(
            ("intensity",),
            np.zeros(4),
            np.array([1.0, 1.0, 1.0, 0.0]),
            np.ones((5, 2)),
        )
