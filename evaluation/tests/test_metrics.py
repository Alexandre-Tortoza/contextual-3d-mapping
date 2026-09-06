"""Testes das primitivas de métrica, verificados contra valores calculados à mão (#198)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception_evaluation.masks import (
    boundary,
    boundary_f1,
    decode_mask,
    dilate,
    intersection_over_union,
)
from visual_perception_evaluation.metrics import (
    MetricValue,
    RegionMatch,
    RegionMatching,
    abstention_report,
    bootstrap_interval,
    boundary_metrics,
    brier_score,
    detection_metrics,
    expected_calibration_error,
    match_regions,
    open_vocabulary_scores,
    reliability_table,
    set_metrics,
)


# Constrói uma máscara retangular em uma imagem 8x8, para que os testes
# possam calcular áreas e interseções mentalmente.
def _square(x_min: int, y_min: int, size: int = 3, canvas: int = 8) -> np.ndarray:
    mask = np.zeros((canvas, canvas), dtype=np.bool_)
    mask[y_min : y_min + size, x_min : x_min + size] = True
    return mask


def test_rle_decoding_matches_the_written_encoding() -> None:
    # 4x4 = 16 pixels: 5 de fundo, 3 ocupados, 8 de fundo.
    mask = decode_mask(4, 4, (5, 3, 8))

    assert mask.sum() == 3
    assert mask.reshape(-1)[5:8].all()
    with pytest.raises(ValueError, match="cover exactly 16 pixels"):
        decode_mask(4, 4, (5, 3))


def test_intersection_over_union_matches_a_hand_computed_overlap() -> None:
    # Dois quadrados 3x3 deslocados de uma coluna: interseção 3x2 = 6,
    # união 3x4 = 12, IoU = 0.5.
    assert intersection_over_union(_square(2, 2), _square(3, 2)) == pytest.approx(0.5)
    assert intersection_over_union(_square(0, 0), _square(5, 5)) == 0.0


def test_boundary_extraction_keeps_only_the_ring_of_a_solid_square() -> None:
    # Um quadrado 3x3 tem 8 pixels de contorno e 1 de interior.
    assert boundary(_square(2, 2)).sum() == 8
    # Um quadrado 5x5 dilatado de 1 cobre 5x5 = 25 pixels a partir de 3x3.
    assert dilate(_square(2, 2), 1).sum() == 25


def test_dilation_does_not_wrap_around_the_image_border() -> None:
    corner = np.zeros((5, 5), dtype=np.bool_)
    corner[0, 0] = True

    dilated = dilate(corner, 1)

    assert dilated.sum() == 4  # apenas o bloco 2x2 do canto, sem reaparecer embaixo
    assert not dilated[-1].any()


def test_boundary_f1_rewards_a_close_contour_and_punishes_a_far_one() -> None:
    reference = _square(2, 2)

    assert boundary_f1(reference, reference) == 1.0
    assert boundary_f1(reference, _square(3, 2), tolerance=2) == 1.0
    assert boundary_f1(reference, _square(0, 0, canvas=8), tolerance=0) < 0.5
    assert boundary_f1(np.zeros((8, 8), bool), np.zeros((8, 8), bool)) == 1.0
    assert boundary_f1(reference, np.zeros((8, 8), bool)) == 0.0


def test_region_matching_is_greedy_deterministic_and_respects_ignored_annotations() -> None:
    references = [_square(0, 0), _square(4, 4)]
    predictions = [_square(4, 4), _square(0, 0), _square(0, 5, size=2)]

    matching = match_regions(references, predictions, ignored=[False, True])

    assert [(match.reference_index, match.prediction_index) for match in matching.matches] == [(0, 1)]
    assert matching.ignored_prediction == (0,)  # absorvida pela anotação ignorada
    assert matching.unmatched_prediction == (2,)
    assert matching.unmatched_reference == ()


def test_detection_metrics_match_a_hand_computed_confusion() -> None:
    # 2 acertos, 1 falso positivo, 1 falso negativo.
    matching = RegionMatching(
        matches=(RegionMatch(0, 0, 0.8), RegionMatch(1, 1, 0.6)),
        unmatched_reference=(2,),
        unmatched_prediction=(2,),
    )

    metrics = detection_metrics(matching)

    assert metrics["region_precision"].value == pytest.approx(2 / 3)
    assert metrics["region_recall"].value == pytest.approx(2 / 3)
    assert metrics["region_f1"].value == pytest.approx(2 / 3)
    assert metrics["matched_mask_iou"].value == pytest.approx(0.7)


def test_metrics_without_support_are_never_reported_as_zero() -> None:
    metrics = detection_metrics(RegionMatching((), (), ()))

    assert metrics["region_precision"].value is None
    assert metrics["region_precision"].measured is False
    with pytest.raises(ValueError, match="cannot carry a value"):
        MetricValue("x", 0.0, 0)
    with pytest.raises(ValueError, match="has support but no value"):
        MetricValue("x", None, 3)


def test_boundary_metrics_average_only_the_matched_regions() -> None:
    references = [_square(0, 0), _square(4, 4)]
    predictions = [_square(0, 0), _square(4, 4)]
    matching = match_regions(references, predictions)

    metric = boundary_metrics(references, predictions, matching, tolerance=1)

    assert metric.value == pytest.approx(1.0)
    assert metric.support == 2
    assert metric.detail == "tolerance=1px"


def test_open_vocabulary_scoring_accepts_any_annotated_label() -> None:
    assert open_vocabulary_scores(("door", "gate"), ("door", "window")) == (1.0, 1.0)
    assert open_vocabulary_scores(("door", "gate"), ("window", "door")) == (0.0, 0.5)
    assert open_vocabulary_scores(("door",), ("window", "wall")) == (0.0, 0.0)
    assert open_vocabulary_scores(("Door",), ("door",)) == (1.0, 1.0)


def test_set_metrics_match_a_hand_computed_example() -> None:
    metrics = set_metrics("attribute", ["wet", "closed"], ["wet", "open", "dirty"])

    assert metrics["attribute_precision"].value == pytest.approx(1 / 3)
    assert metrics["attribute_recall"].value == pytest.approx(1 / 2)
    assert metrics["attribute_f1"].value == pytest.approx(0.4)


def test_a_task_without_annotations_is_unsupported_rather_than_zero() -> None:
    metrics = set_metrics("hazard", [], ["fire"])

    assert metrics["hazard_precision"].measured is False
    assert metrics["hazard_recall"].value is None


def test_expected_calibration_error_matches_a_hand_computed_example() -> None:
    # Bin (0.5, 1.0]: conf média 0.9, acurácia 0.5 -> 0.4, peso 2/4 -> 0.20
    # Bin [0.0, 0.5]: conf média 0.1, acurácia 0.0 -> 0.1, peso 2/4 -> 0.05
    metric = expected_calibration_error([0.9, 0.9, 0.1, 0.1], [True, False, False, False], bins=2)

    assert metric.value == pytest.approx(0.25)
    assert metric.support == 4
    assert metric.detail == "bins=2"


def test_brier_score_matches_a_hand_computed_example() -> None:
    # ((0.9-1)^2 + (0.9-0)^2 + (0.1-0)^2 + (0.1-0)^2) / 4 = 0.84 / 4
    metric = brier_score([0.9, 0.9, 0.1, 0.1], [True, False, False, False])

    assert metric.value == pytest.approx(0.21)


def test_calibration_metrics_without_scores_stay_unmeasured() -> None:
    assert expected_calibration_error([], []).measured is False
    assert brier_score([], []).value is None
    with pytest.raises(ValueError, match="must all lie in"):
        brier_score([1.4], [True])


def test_the_reliability_table_reports_every_bin_including_the_empty_ones() -> None:
    table = reliability_table([0.9, 0.9, 0.1], [True, False, False], bins=2)

    assert table[0]["count"] == 1 and table[0]["accuracy"] == pytest.approx(0.0)
    assert table[1]["count"] == 2 and table[1]["accuracy"] == pytest.approx(0.5)
    assert reliability_table([], [], bins=2)[0]["count"] == 0


def test_the_abstention_report_keeps_unscored_decisions_visible() -> None:
    metrics = abstention_report(
        [True, False, True, False], [True, True, False, False], abstained=[False, False, True, False]
    )

    assert metrics["coverage"].value == pytest.approx(0.5)
    assert metrics["abstention_rate"].value == pytest.approx(0.25)
    assert metrics["unscored_rate"].value == pytest.approx(0.5)
    assert metrics["selective_risk"].value == pytest.approx(0.5)


def test_abstaining_from_everything_has_zero_coverage_and_no_selective_risk() -> None:
    metrics = abstention_report([True, False], [False, False], abstained=[True, True])

    assert metrics["coverage"].value == 0.0
    assert metrics["selective_risk"].measured is False


def test_bootstrap_intervals_are_reproducible_and_bracket_the_mean() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 1.0]

    first = bootstrap_interval(values, seed=7, resamples=200)
    second = bootstrap_interval(values, seed=7, resamples=200)

    assert first == second
    assert first is not None and first[0] <= 0.6 <= first[1]
    assert bootstrap_interval([0.5]) is None
