"""Testes da avaliação ponta a ponta contra o conjunto de referência (#198)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from contextual_mapping_datasets import (
    AnnotationCertainty,
    AnnotationProvenance,
    MaskAnnotation,
    ReferenceManifest,
    RegionAnnotation,
    RelationAnnotation,
    SampleAnnotation,
    SampleArtifact,
    Split,
)
from visual_perception_evaluation import (
    EvaluationConfig,
    PredictedRegion,
    PredictedRelation,
    PredictionSet,
    SamplePrediction,
    ScoredValue,
    SupportState,
    evaluate_split,
    load_predictions,
    prediction_set_from_mapping,
    prediction_set_to_mapping,
    render_comparison,
    render_report,
    save_predictions,
)

_CANVAS = 16
_PROVENANCE = AnnotationProvenance("reviewer", "manual", "annotation-policy/1")


# Codifica uma máscara booleana no RLE compartilhado pelos dois contracts,
# para que os testes construam anotações e predições da mesma geometria.
def _runs(mask: np.ndarray) -> tuple[int, ...]:
    flat = mask.reshape(-1)
    runs: list[int] = []
    current = False
    count = 0
    for value in flat:
        if bool(value) == current:
            count += 1
            continue
        runs.append(count)
        current = bool(value)
        count = 1
    runs.append(count)
    return tuple(runs)


# Constrói um quadrado no canvas de teste, o bloco de geometria de todos os
# casos deste arquivo.
def _square(x_min: int, y_min: int, size: int) -> np.ndarray:
    mask = np.zeros((_CANVAS, _CANVAS), dtype=np.bool_)
    mask[y_min : y_min + size, x_min : x_min + size] = True
    return mask


# Constrói uma anotação de região a partir de uma máscara.
def _annotation(
    annotation_id: str, mask: np.ndarray, labels: tuple[str, ...] = ("door",), **overrides: object
) -> RegionAnnotation:
    return RegionAnnotation(
        annotation_id=annotation_id,
        mask=MaskAnnotation(_CANVAS, _CANVAS, _runs(mask)),
        labels=labels,
        **overrides,  # type: ignore[arg-type]
    )


# Constrói uma região prevista a partir de uma máscara e de labels pontuados.
def _predicted(
    region_id: str, mask: np.ndarray, labels: tuple[ScoredValue, ...] = (), **overrides: object
) -> PredictedRegion:
    return PredictedRegion(
        region_id=region_id,
        width=_CANVAS,
        height=_CANVAS,
        runs=_runs(mask),
        labels=labels,
        **overrides,  # type: ignore[arg-type]
    )


# Constrói uma amostra anotada do conjunto de referência.
def _sample(
    sample_id: str,
    regions: tuple[RegionAnnotation, ...],
    *,
    split: Split = Split.TEST,
    relations: tuple[RelationAnnotation, ...] = (),
    strata: tuple[str, ...] = (),
) -> SampleAnnotation:
    return SampleAnnotation(
        sample_id=sample_id,
        artifact=SampleArtifact(
            f"frames/{sample_id}.png", "image/png", hashlib.sha256(sample_id.encode()).hexdigest()
        ),
        width=_CANVAS,
        height=_CANVAS,
        split=split,
        provenance=_PROVENANCE,
        regions=regions,
        relations=relations,
        strata=strata,
    )


# Constrói o manifest de referência com as amostras dadas.
def _manifest(samples: tuple[SampleAnnotation, ...]) -> ReferenceManifest:
    return ReferenceManifest(
        reference_id="test-reference",
        dataset_id="corridor-02",
        policy_uri="docs/annotation-policy.md",
        samples=samples,
    )


# Constrói o conjunto de predições com as amostras dadas.
def _predictions(samples: tuple[SamplePrediction, ...], name: str = "baseline") -> PredictionSet:
    return PredictionSet(
        run_id="run-1",
        configuration=name,
        config_fingerprint="fingerprint",
        code_revision="abc1234",
        samples=samples,
    )


def test_a_perfect_prediction_scores_one_on_every_supported_metric() -> None:
    manifest = _manifest((_sample("s1", (_annotation("a1", _square(2, 2, 6)),)),))
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted(
                        "r1",
                        _square(2, 2, 6),
                        (ScoredValue("door", 0.9, 0.8, SupportState.CALIBRATED),),
                    ),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.metrics["region_precision"].value == 1.0
    assert report.metrics["region_recall"].value == 1.0
    assert report.metrics["matched_mask_iou"].value == pytest.approx(1.0)
    assert report.metrics["label_top1"].value == 1.0
    assert report.metrics["sample_failure_rate"].value == 0.0


def test_a_failed_sample_enters_the_failure_rate_and_contributes_no_quality() -> None:
    manifest = _manifest(
        (
            _sample("s1", (_annotation("a1", _square(2, 2, 6)),)),
            _sample("s2", (_annotation("a2", _square(2, 2, 6)),)),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(_predicted("r1", _square(2, 2, 6), (ScoredValue("door", 0.9),)),),
            ),
            SamplePrediction("s2", _CANVAS, _CANVAS, failed=True, failure_reason="backend OOM"),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.sample_count == 2
    assert report.failed_sample_count == 1
    assert report.metrics["sample_failure_rate"].value == 0.5
    assert report.metrics["region_recall"].value == 1.0  # só a amostra válida entra


def test_a_sample_without_any_prediction_counts_as_a_failure() -> None:
    manifest = _manifest(
        (
            _sample("s1", (_annotation("a1", _square(2, 2, 6)),)),
            _sample("s2", (_annotation("a2", _square(2, 2, 6)),)),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(_predicted("r1", _square(2, 2, 6), (ScoredValue("door", 0.9),)),),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.failed_sample_count == 1
    assert report.outcomes[1].failure_reason == "no prediction was produced for this sample"


def test_an_ignored_annotation_absorbs_its_prediction_without_scoring_it() -> None:
    manifest = _manifest(
        (
            _sample(
                "s1",
                (
                    _annotation("a1", _square(1, 1, 4)),
                    _annotation("a2", _square(9, 9, 4), (), ignored=True),
                ),
            ),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted("r1", _square(1, 1, 4), (ScoredValue("door", 0.9),)),
                    _predicted("r2", _square(9, 9, 4), (ScoredValue("debris", 0.9),)),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.ignored_annotation_count == 1
    assert report.metrics["region_precision"].value == 1.0  # a predição ignorada não é falso positivo
    assert report.metrics["region_recall"].value == 1.0
    assert report.unsupported_claims == ()


def test_an_unknown_annotation_counts_for_detection_and_not_for_semantics() -> None:
    manifest = _manifest(
        (
            _sample(
                "s1",
                (
                    _annotation("a1", _square(1, 1, 4)),
                    _annotation("a2", _square(9, 9, 4), (), certainty=AnnotationCertainty.UNKNOWN),
                ),
            ),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted("r1", _square(1, 1, 4), (ScoredValue("door", 0.9),)),
                    _predicted("r2", _square(9, 9, 4), (ScoredValue("anything", 0.9),)),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.metrics["region_recall"].value == 1.0
    assert report.metrics["label_top1"].support == 1  # só a anotação com label é pontuável
    assert report.metrics["label_top1"].value == 1.0


def test_a_prediction_with_no_annotated_counterpart_is_reported_as_unsupported() -> None:
    manifest = _manifest((_sample("s1", (_annotation("a1", _square(1, 1, 4)),)),))
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted("r1", _square(1, 1, 4), (ScoredValue("door", 0.9),)),
                    _predicted("r2", _square(10, 10, 4), (ScoredValue("fire", 0.9),)),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.metrics["unsupported_claim_rate"].value == pytest.approx(0.5)
    assert [claim.value for claim in report.unsupported_claims] == ["fire"]
    assert "matched no annotated region" in report.unsupported_claims[0].reason


def test_ambiguity_is_not_counted_as_a_hallucination() -> None:
    manifest = _manifest(
        (
            _sample(
                "s1",
                (
                    _annotation(
                        "a1", _square(1, 1, 4), ("door", "gate"), certainty=AnnotationCertainty.AMBIGUOUS
                    ),
                ),
            ),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(_predicted("r1", _square(1, 1, 4), (ScoredValue("gate", 0.9),)),),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.unsupported_claims == ()
    assert report.metrics["label_top1"].value == 1.0
    assert "ambiguous_annotation" in report.strata


def test_metrics_are_broken_down_by_region_size_stratum() -> None:
    manifest = _manifest(
        (
            _sample(
                "s1",
                (_annotation("small", _square(1, 1, 2)), _annotation("large", _square(6, 6, 8))),
                strata=("low_light",),
            ),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted("r1", _square(1, 1, 2), (ScoredValue("door", 0.9),)),
                    _predicted("r2", _square(6, 6, 8), (ScoredValue("wall", 0.9),)),
                ),
            ),
        )
    )

    report = evaluate_split(
        manifest, predictions, Split.TEST, EvaluationConfig(small_region_max_area_px=16)
    )

    assert report.strata["small_region"]["region_recall"].value == 1.0
    assert report.strata["small_region"]["matched_mask_iou"].support == 1
    assert report.strata["ordinary_region"]["label_top1"].value == 0.0  # "wall" não é "door"
    assert report.strata["low_light"]["region_recall"].support == 2


def test_raw_and_calibrated_reliability_are_reported_separately() -> None:
    manifest = _manifest(
        (
            _sample("s1", (_annotation("a1", _square(1, 1, 4)),)),
            _sample("s2", (_annotation("a2", _square(1, 1, 4)),)),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted(
                        "r1", _square(1, 1, 4), (ScoredValue("door", 0.9, 0.6, SupportState.CALIBRATED),)
                    ),
                ),
            ),
            SamplePrediction(
                "s2",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted(
                        "r2", _square(1, 1, 4), (ScoredValue("wall", 0.9, 0.6, SupportState.CALIBRATED),)
                    ),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST, EvaluationConfig(calibration_bins=2))

    # Acurácia real 0.5: o score bruto 0.9 erra por 0.4, o calibrado 0.6 por 0.1.
    assert report.calibration["raw"]["expected_calibration_error"].value == pytest.approx(0.4)
    assert report.calibration["calibrated"]["expected_calibration_error"].value == pytest.approx(0.1)
    assert report.calibration["raw"]["brier_score"].value == pytest.approx(0.41)


def test_abstained_claims_stay_visible_in_coverage_and_risk() -> None:
    manifest = _manifest(
        (
            _sample("s1", (_annotation("a1", _square(1, 1, 4)),)),
            _sample("s2", (_annotation("a2", _square(1, 1, 4)),)),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted(
                        "r1", _square(1, 1, 4), (ScoredValue("door", 0.9, 0.7, SupportState.CALIBRATED),)
                    ),
                ),
            ),
            SamplePrediction(
                "s2",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted(
                        "r2",
                        _square(1, 1, 4),
                        (
                            ScoredValue(
                                "wall", 0.9, None, SupportState.ABSTAINED, "insufficient support"
                            ),
                        ),
                    ),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.abstention["coverage"].value == pytest.approx(0.5)
    assert report.abstention["abstention_rate"].value == pytest.approx(0.5)
    assert report.abstention["selective_risk"].value == pytest.approx(0.0)
    assert report.calibration["calibrated"]["scored_decisions"].value == 1.0


def test_relation_metrics_use_the_region_matching_to_align_identities() -> None:
    manifest = _manifest(
        (
            _sample(
                "s1",
                (_annotation("a1", _square(1, 1, 4)), _annotation("a2", _square(9, 9, 4), ("floor",))),
                relations=(RelationAnnotation("rel-1", "a1", "near", "a2"),),
            ),
        )
    )
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted("r1", _square(1, 1, 4), (ScoredValue("door", 0.9),)),
                    _predicted("r2", _square(9, 9, 4), (ScoredValue("floor", 0.9),)),
                ),
                relations=(PredictedRelation("p-rel-1", "r1", "near", "r2", 0.5),),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)

    assert report.metrics["relation_precision"].value == 1.0
    assert report.metrics["relation_recall"].value == 1.0


def test_tasks_without_annotations_stay_unmeasured_in_the_report() -> None:
    manifest = _manifest((_sample("s1", (_annotation("a1", _square(1, 1, 4)),)),))
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(
                    _predicted(
                        "r1",
                        _square(1, 1, 4),
                        (ScoredValue("door", 0.9),),
                        hazards=(ScoredValue("fire", 0.9),),
                    ),
                ),
            ),
        )
    )

    report = evaluate_split(manifest, predictions, Split.TEST)
    rendered = render_report(report)

    assert report.metrics["hazard_precision"].measured is False
    assert "hazard_precision" not in report.measured_metrics()
    assert "não medida (sem suporte)" in rendered


def test_the_comparison_refuses_to_rank_configurations_on_an_unmeasured_metric() -> None:
    manifest = _manifest((_sample("s1", (_annotation("a1", _square(1, 1, 4)),)),))
    baseline = evaluate_split(
        manifest,
        _predictions(
            (
                SamplePrediction(
                    "s1",
                    _CANVAS,
                    _CANVAS,
                    regions=(_predicted("r1", _square(1, 1, 4), (ScoredValue("door", 0.9),)),),
                ),
            ),
            "baseline",
        ),
        Split.TEST,
    )
    candidate = evaluate_split(
        manifest,
        _predictions(
            (
                SamplePrediction(
                    "s1",
                    _CANVAS,
                    _CANVAS,
                    regions=(_predicted("r1", _square(2, 2, 4), (ScoredValue("wall", 0.9),)),),
                ),
            ),
            "high_resolution",
        ),
        Split.TEST,
    )

    rendered = render_comparison(baseline, [candidate])

    assert "não comparável" in rendered
    assert "`label_top1`" in rendered


def test_the_prediction_contract_round_trips_and_rejects_a_foreign_version(tmp_path: Path) -> None:
    predictions = _predictions(
        (
            SamplePrediction(
                "s1",
                _CANVAS,
                _CANVAS,
                regions=(_predicted("r1", _square(1, 1, 4), (ScoredValue("door", 0.9, 0.7),)),),
                latency_s=1.5,
                peak_vram_bytes=1024,
                model_calls=3,
            ),
        )
    )
    path = tmp_path / "predictions.json"

    save_predictions(predictions, path)

    assert load_predictions(path) == predictions
    assert prediction_set_from_mapping(prediction_set_to_mapping(predictions)) == predictions
    document = prediction_set_to_mapping(predictions)
    document["schema_version"] = "visual-prediction/99"
    (tmp_path / "bad.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported prediction schema_version"):
        load_predictions(tmp_path / "bad.json")


def test_a_failed_prediction_may_not_smuggle_regions_through_the_contract() -> None:
    with pytest.raises(ValueError, match="marked failed but carries regions"):
        SamplePrediction(
            "s1",
            _CANVAS,
            _CANVAS,
            regions=(_predicted("r1", _square(1, 1, 4)),),
            failed=True,
            failure_reason="OOM",
        )
    with pytest.raises(ValueError, match="failed without a reason"):
        SamplePrediction("s1", _CANVAS, _CANVAS, failed=True)


def test_evaluating_a_split_without_samples_fails_instead_of_reporting_nothing() -> None:
    manifest = _manifest((_sample("s1", (_annotation("a1", _square(1, 1, 4)),), split=Split.TEST),))
    predictions = _predictions((SamplePrediction("s1", _CANVAS, _CANVAS),))

    with pytest.raises(ValueError, match="no samples in split 'development'"):
        evaluate_split(manifest, predictions, Split.DEVELOPMENT)
