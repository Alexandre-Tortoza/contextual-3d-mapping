"""Testes da fronteira de extração de features visuais densas (#161)."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import payload_with_blobs
from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor


# Garante que a resolução pedida na config é respeitada no shape da grade de saída.
def test_feature_map_shape_matches_requested_resolution() -> None:
    payload = payload_with_blobs(width=32, height=32)
    feature_map = FakeDenseFeatureExtractor().extract(payload, FeatureExtractionConfig(feature_resolution=8))
    assert (feature_map.grid_height, feature_map.grid_width) == (8, 8)


# Garante que o fake nunca produz NaN/Inf — um feature map inválido quebraria todo
# estágio downstream que consome esses valores.
def test_feature_map_is_finite() -> None:
    payload = payload_with_blobs(width=16, height=16)
    feature_map = FakeDenseFeatureExtractor().extract(payload, FeatureExtractionConfig(feature_resolution=4))
    assert np.isfinite(feature_map.data).all()


# Confirma que múltiplas resoluções de feature grid são suportadas pelo mesmo extractor.
def test_multiple_resolutions_are_supported() -> None:
    payload = payload_with_blobs(width=32, height=32)
    for resolution in (4, 8, 16):
        feature_map = FakeDenseFeatureExtractor().extract(
            payload, FeatureExtractionConfig(feature_resolution=resolution)
        )
        assert feature_map.grid_width == resolution


# Protege o invariante de FeatureMap de nunca aceitar dados com NaN.
def test_feature_map_rejects_nan() -> None:
    data = np.full((2, 2, 2), np.nan)
    with pytest.raises(ValueError):
        FeatureMap(data, stride_x=1.0, stride_y=1.0, dimension=2, model_id="fake")


# Protege o invariante de que a dimensão declarada (`dimension`) deve bater com o shape
# real dos dados.
def test_feature_map_rejects_dimension_mismatch() -> None:
    data = np.zeros((2, 2, 3))
    with pytest.raises(ValueError):
        FeatureMap(data, stride_x=1.0, stride_y=1.0, dimension=4, model_id="fake")
