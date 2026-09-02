"""Region, mask, and image-coordinate invariant tests (#155)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.domain.geometry import BoundingBox, CoordinateTransform, Mask


def _mask(data: list[list[bool]]) -> Mask:
    array = np.array(data, dtype=np.bool_)
    return Mask(array, image_width=array.shape[1], image_height=array.shape[0])


def test_bounding_box_rejects_non_positive_area() -> None:
    with pytest.raises(ValueError):
        BoundingBox(0, 0, 0, 5)


def test_bounding_box_clips_to_image_bounds() -> None:
    box = BoundingBox(-5, -5, 15, 15).clipped_to(width=10, height=10)
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0, 0, 10, 10)


def test_mask_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError):
        Mask(np.zeros((2, 2), dtype=np.int32), image_width=2, image_height=2)


def test_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        Mask(np.zeros((2, 2), dtype=np.bool_), image_width=3, image_height=2)


def test_empty_mask_bounding_box_raises() -> None:
    mask = _mask([[False, False], [False, False]])
    assert mask.is_empty
    with pytest.raises(ValueError):
        mask.bounding_box()


def test_mask_bounding_box_for_edge_pixel() -> None:
    mask = _mask([[True, False], [False, False]])
    box = mask.bounding_box()
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0.0, 0.0, 1.0, 1.0)


def test_mask_iou_and_containment() -> None:
    a = _mask([[True, True], [False, False]])
    b = _mask([[True, False], [False, False]])
    assert a.iou(b) == pytest.approx(0.5)
    assert a.containment_ratio(b) == pytest.approx(1.0)
    assert b.containment_ratio(a) == pytest.approx(0.5)


def test_identity_transform_is_a_no_op() -> None:
    transform = CoordinateTransform.identity()
    box = BoundingBox(1, 2, 3, 4)
    assert transform.box_to_global(box) == box


def test_transform_box_round_trip() -> None:
    transform = CoordinateTransform(scale_x=2.0, scale_y=2.0, offset_x=10.0, offset_y=5.0)
    local_box = BoundingBox(1, 1, 4, 4)
    global_box = transform.box_to_global(local_box)
    recovered = transform.box_to_local(global_box)
    assert recovered.x_min == pytest.approx(local_box.x_min)
    assert recovered.y_min == pytest.approx(local_box.y_min)
    assert recovered.x_max == pytest.approx(local_box.x_max)
    assert recovered.y_max == pytest.approx(local_box.y_max)


def test_tile_mask_remaps_exactly_for_integer_offset() -> None:
    local_mask = _mask([[True, False], [False, True]])
    transform = CoordinateTransform(scale_x=1.0, scale_y=1.0, offset_x=2.0, offset_y=3.0)
    global_mask = transform.mask_to_global(local_mask, global_width=6, global_height=6)
    expected = np.zeros((6, 6), dtype=np.bool_)
    expected[3, 2] = True
    expected[4, 3] = True
    assert np.array_equal(global_mask.data, expected)
