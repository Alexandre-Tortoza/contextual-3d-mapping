"""Reproducible perception quality benchmark and ablation harness.

Issue: #175.

Metrics separate segmentation/localization quality (region coverage, mask
IoU) from semantics (label match rate), relations, and hallucination, and
every run is defined by a versioned :class:`DatasetReference` plus an
:class:`AblationConfig` so a report is reproducible from its own fields.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from visual_perception.application.pipeline import PerceptionPorts, run_canonical_pipeline
from visual_perception.config import ModuleConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.visual_observation import VisualObservation


@dataclass(frozen=True)
class DatasetReference:
    """A versioned pointer to the dataset/annotations a report was computed against."""

    name: str
    version: str
    annotation_uri: str


@dataclass(frozen=True)
class AblationConfig:
    """Which capabilities are enabled for one benchmark run.

    Disabling one capability must not change the behavior of unrelated
    stages: this is enforced by callers composing ``ModuleConfig`` directly
    rather than by a hidden global switch.
    """

    name: str
    module_config: ModuleConfig


@dataclass(frozen=True)
class GroundTruthRegion:
    """One annotated region against which predictions are scored."""

    mask: Mask
    label: str | None = None


@dataclass(frozen=True)
class QualityMetrics:
    region_coverage: float
    mean_matched_mask_iou: float
    label_match_rate: float
    hallucination_rate: float

    def __post_init__(self) -> None:
        for name in ("region_coverage", "mean_matched_mask_iou", "label_match_rate", "hallucination_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")


@dataclass(frozen=True)
class BenchmarkReport:
    dataset: DatasetReference
    ablation: AblationConfig
    metrics: QualityMetrics
    runtime_s: float
    model_calls: int = field(default=0)


_MATCH_IOU_THRESHOLD = 0.5


def score_observation(
    observation: VisualObservation, ground_truth: tuple[GroundTruthRegion, ...]
) -> QualityMetrics:
    """Score one canonical observation against its ground-truth regions."""
    if not ground_truth:
        hallucinated = 1.0 if observation.regions else 0.0
        return QualityMetrics(0.0 if observation.regions else 1.0, 0.0, 0.0, hallucinated)

    matched_gt: set[int] = set()
    matched_predictions: list[tuple[float, bool]] = []  # (iou, label_matched)

    for region in observation.regions:
        best_iou, best_index, best_label_match = 0.0, None, False
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            iou = region.mask.iou(gt.mask)
            if iou > best_iou:
                labels = {claim.value for claim in region.claims if claim.kind.value == "label"}
                best_iou, best_index, best_label_match = iou, gt_index, bool(gt.label and gt.label in labels)
        if best_iou >= _MATCH_IOU_THRESHOLD and best_index is not None:
            matched_gt.add(best_index)
            matched_predictions.append((best_iou, best_label_match))

    region_coverage = len(matched_gt) / len(ground_truth)
    match_count = len(matched_predictions)
    mean_iou = sum(iou for iou, _ in matched_predictions) / match_count if match_count else 0.0
    label_matches = [matched for _, matched in matched_predictions if matched]
    label_match_rate = len(label_matches) / match_count if match_count else 0.0
    hallucinations = len(observation.regions) - len(matched_predictions)
    hallucination_rate = hallucinations / len(observation.regions) if observation.regions else 0.0

    return QualityMetrics(region_coverage, mean_iou, label_match_rate, hallucination_rate)


def run_ablation(
    dataset: DatasetReference,
    ablation: AblationConfig,
    samples: tuple[tuple[ImageObservation, ImagePayload, tuple[GroundTruthRegion, ...]], ...],
    ports: PerceptionPorts,
) -> BenchmarkReport:
    """Run the canonical pipeline under one ablation over a fixed sample set."""
    start = time.monotonic()
    per_sample_metrics = []
    for image, payload, ground_truth in samples:
        result = run_canonical_pipeline(image, payload, ablation.module_config, ports)
        per_sample_metrics.append(score_observation(result.observation, ground_truth))
    runtime_s = time.monotonic() - start

    count = len(per_sample_metrics) or 1
    aggregate = QualityMetrics(
        region_coverage=sum(m.region_coverage for m in per_sample_metrics) / count,
        mean_matched_mask_iou=sum(m.mean_matched_mask_iou for m in per_sample_metrics) / count,
        label_match_rate=sum(m.label_match_rate for m in per_sample_metrics) / count,
        hallucination_rate=sum(m.hallucination_rate for m in per_sample_metrics) / count,
    )
    return BenchmarkReport(
        dataset=dataset, ablation=ablation, metrics=aggregate, runtime_s=runtime_s, model_calls=len(samples)
    )
