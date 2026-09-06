"""Testes de contract da evidência densa pixel-aligned (#191) e do seu upsampling (#192)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.application.dense_evidence import (
    BILINEAR,
    NEAREST,
    upsample_feature_map,
    upsampled_bytes,
)
from visual_perception.application.pooling import BASELINE, PIXEL_BILINEAR, pool_region_evidence
from visual_perception.domain.feature_map import (
    FeatureMap,
    FeatureRepresentation,
    SamplingRule,
    feature_map_spec_from_dict,
    feature_map_spec_to_dict,
    sample_feature_map,
)
from visual_perception.domain.geometry import BoundingBox, Mask


# Constrói um feature map de gradiente com valores previsíveis, para que os
# testes possam comparar amostragens contra números calculados à mão.
def _gradient_map(
    grid: int = 4,
    *,
    stride: float = 8.0,
    origin: tuple[float, float] = (0.0, 0.0),
    support: np.ndarray | None = None,
) -> FeatureMap:
    data = np.zeros((grid, grid, 2), dtype=np.float64)
    for row in range(grid):
        for column in range(grid):
            data[row, column] = (float(column), float(row))
    return FeatureMap(
        data=data,
        stride_x=stride,
        stride_y=stride,
        dimension=2,
        model_id="test-backbone",
        origin_x=origin[0],
        origin_y=origin[1],
        checkpoint="test/ckpt",
        valid_support=support,
    )


def test_patch_grid_and_pixel_aligned_maps_stay_distinguishable() -> None:
    patch_grid = _gradient_map()
    pixel_aligned = upsample_feature_map(patch_grid, method=NEAREST, width=32, height=32)

    assert patch_grid.representation is FeatureRepresentation.PATCH_GRID
    assert pixel_aligned.representation is FeatureRepresentation.PIXEL_ALIGNED
    assert (pixel_aligned.stride_x, pixel_aligned.stride_y) == (1.0, 1.0)
    assert pixel_aligned.upsampling_method == NEAREST


def test_a_pixel_aligned_map_must_declare_unit_strides_and_its_upsampler() -> None:
    data = np.zeros((2, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="unit strides"):
        FeatureMap(
            data=data,
            stride_x=4.0,
            stride_y=4.0,
            dimension=2,
            model_id="test",
            representation=FeatureRepresentation.PIXEL_ALIGNED,
            upsampling_method=NEAREST,
        )
    with pytest.raises(ValueError, match="upsampling_method"):
        FeatureMap(
            data=data,
            stride_x=1.0,
            stride_y=1.0,
            dimension=2,
            model_id="test",
            representation=FeatureRepresentation.PIXEL_ALIGNED,
        )


def test_an_original_image_coordinate_maps_to_the_cell_that_covers_it() -> None:
    feature_map = _gradient_map()  # 4x4 grid, stride 8 -> cobre 32x32 pixels

    values, valid = sample_feature_map(
        feature_map, np.array([4.0, 12.0, 20.0, 28.0]), np.array([4.0, 4.0, 4.0, 4.0])
    )

    assert valid.all()
    assert values[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert values[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_a_tiled_feature_map_reports_its_own_origin_and_covered_region() -> None:
    tile_map = _gradient_map(origin=(32.0, 16.0))

    assert tile_map.covered_region == (32.0, 16.0, 64.0, 48.0)
    inside, valid_inside = sample_feature_map(tile_map, np.array([36.0]), np.array([20.0]))
    outside, valid_outside = sample_feature_map(tile_map, np.array([4.0]), np.array([20.0]))

    assert valid_inside.tolist() == [True]
    assert inside[0].tolist() == [0.0, 0.0]
    assert valid_outside.tolist() == [False]
    assert outside[0].tolist() == [0.0, 0.0]


def test_a_resized_map_keeps_an_invertible_transform() -> None:
    feature_map = _gradient_map(stride=8.0, origin=(5.0, 7.0))
    transform = feature_map.transform

    box = transform.box_to_global(transform.box_to_local(BoundingBox(10.0, 12.0, 20.0, 24.0)))

    assert (transform.scale_x, transform.offset_x) == (8.0, 5.0)
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (10.0, 12.0, 20.0, 24.0)


def test_unsupported_border_coordinates_are_invalid_under_bilinear_sampling() -> None:
    feature_map = _gradient_map()

    # O centro da primeira célula fica em x=4; qualquer coisa à esquerda dele
    # não tem vizinhança completa para interpolar.
    values, valid = sample_feature_map(
        feature_map,
        np.array([1.0, 8.0, 30.0]),
        np.array([8.0, 8.0, 8.0]),
        interpolation=SamplingRule.BILINEAR,
    )

    assert valid.tolist() == [False, True, False]
    assert values[0].tolist() == [0.0, 0.0]


def test_cells_without_valid_support_are_never_extrapolated() -> None:
    support = np.ones((4, 4), dtype=np.bool_)
    support[0, 0] = False
    feature_map = _gradient_map(support=support)

    values, valid = sample_feature_map(feature_map, np.array([4.0, 12.0]), np.array([4.0, 4.0]))

    assert valid.tolist() == [False, True]
    assert values[0].tolist() == [0.0, 0.0]
    assert feature_map.support_ratio == pytest.approx(15 / 16)


def test_bilinear_sampling_interpolates_between_neighbouring_cells() -> None:
    feature_map = _gradient_map()

    values, valid = sample_feature_map(
        feature_map, np.array([8.0]), np.array([4.0]), interpolation=SamplingRule.BILINEAR
    )

    assert valid.tolist() == [True]
    # x=8 fica exatamente no meio entre os centros das colunas 0 (x=4) e 1 (x=12).
    assert values[0].tolist() == [0.5, 0.0]


def test_the_dense_spec_round_trips_and_rejects_malformed_metadata() -> None:
    feature_map = _gradient_map()

    spec = feature_map_spec_to_dict(feature_map)
    restored = feature_map_spec_from_dict(spec)

    assert restored["representation"] == FeatureRepresentation.PATCH_GRID.value
    assert restored["interpolation"] == SamplingRule.NEAREST.value
    assert restored["dimension"] == 2
    assert restored["checkpoint"] == "test/ckpt"
    with pytest.raises(ValueError, match="missing required fields"):
        feature_map_spec_from_dict({"representation": "patch_grid"})
    with pytest.raises(ValueError, match="positive integer"):
        feature_map_spec_from_dict({**spec, "grid_width": 0})


def test_upsampling_and_on_demand_sampling_agree_on_the_same_coordinates() -> None:
    feature_map = _gradient_map()
    upsampled = upsample_feature_map(feature_map, method=BILINEAR, width=32, height=32)

    xs, ys = np.array([8.5, 16.5, 24.5]), np.array([8.5, 16.5, 24.5])
    sampled, sampled_valid = sample_feature_map(
        feature_map, xs, ys, interpolation=SamplingRule.BILINEAR
    )
    materialized, materialized_valid = sample_feature_map(upsampled, xs, ys)

    assert sampled_valid.tolist() == materialized_valid.tolist()
    assert materialized == pytest.approx(sampled, abs=1e-6)


def test_materializing_a_map_above_the_memory_budget_is_refused_with_an_alternative() -> None:
    feature_map = _gradient_map()

    with pytest.raises(ValueError, match="sample_feature_map"):
        upsample_feature_map(feature_map, method=NEAREST, width=1280, height=720, max_bytes=1024)
    assert upsampled_bytes(feature_map, 1280, 720) == 1280 * 720 * 2 * 4


def test_pooling_reports_how_much_dense_support_a_region_actually_had() -> None:
    support = np.ones((4, 4), dtype=np.bool_)
    support[0, :] = False
    feature_map = _gradient_map(support=support)
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[0:16, 0:8] = True  # metade das linhas caem na faixa sem suporte

    pooled = pool_region_evidence(Mask(mask, 32, 32), feature_map, PIXEL_BILINEAR)

    assert 0.0 < pooled.support_ratio < 1.0
    assert pooled.supported_pixels < int(mask.sum())


def test_the_patch_grid_baseline_still_rejects_regions_smaller_than_one_cell() -> None:
    feature_map = _gradient_map()
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[9:11, 9:11] = True  # nenhum centro de célula (4, 12, 20, 28) cai aqui

    with pytest.raises(ValueError, match=BASELINE):
        pool_region_evidence(Mask(mask, 32, 32), feature_map, BASELINE)
