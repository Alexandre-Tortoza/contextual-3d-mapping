"""Harness reproduzível de benchmark de qualidade de percepção e ablation.

Issue: #175.

As métricas separam qualidade de segmentação/localização (region coverage,
mask IoU) de semântica (label match rate), relações, e hallucination, e toda
execução é definida por um :class:`DatasetReference` versionado mais um
:class:`AblationConfig`, para que um report seja reproduzível a partir de
seus próprios campos.
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


# Identifica de forma versionada qual dataset e anotações geraram um
# BenchmarkReport, para que o report seja rastreável e reproduzível.
@dataclass(frozen=True)
class DatasetReference:
    """Um ponteiro versionado para o dataset/anotações contra o qual um report foi calculado."""

    name: str
    version: str
    annotation_uri: str


# Agrupa um nome de ablation com o ModuleConfig completo que a implementa,
# para que run_ablation saiba exatamente qual configuração produziu um
# BenchmarkReport.
@dataclass(frozen=True)
class AblationConfig:
    """Quais capacidades estão habilitadas para uma execução de benchmark.

    Desabilitar uma capacidade não deve mudar o comportamento de estágios não
    relacionados: isso é garantido por quem chama compor o ``ModuleConfig``
    diretamente, em vez de um switch global oculto.
    """

    name: str
    module_config: ModuleConfig


# Representa uma região de ground-truth (mask + label opcional) usada por
# score_observation para avaliar as predições do pipeline canônico.
@dataclass(frozen=True)
class GroundTruthRegion:
    """Uma região anotada contra a qual as predições são pontuadas."""

    mask: Mask
    label: str | None = None


# Agrega as métricas de qualidade de uma execução (cobertura de região, IoU
# médio das masks casadas, taxa de acerto de label, e taxa de hallucination),
# todas normalizadas em [0, 1]. Produzida por score_observation e agregada
# por run_ablation.
@dataclass(frozen=True)
class QualityMetrics:
    region_coverage: float
    mean_matched_mask_iou: float
    label_match_rate: float
    hallucination_rate: float

    # Valida que cada métrica está no intervalo [0, 1] esperado, falhando
    # cedo se score_observation ou run_ablation produzirem um valor fora da
    # faixa.
    def __post_init__(self) -> None:
        for name in ("region_coverage", "mean_matched_mask_iou", "label_match_rate", "hallucination_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")


# Resultado completo de uma execução de ablation: qual dataset, qual
# configuração de ablation, as métricas agregadas, o tempo de execução, e
# quantas chamadas de modelo foram feitas. Retornado por run_ablation.
@dataclass(frozen=True)
class BenchmarkReport:
    dataset: DatasetReference
    ablation: AblationConfig
    metrics: QualityMetrics
    runtime_s: float
    model_calls: int = field(default=0)


_MATCH_IOU_THRESHOLD = 0.5


# Compara as regiões previstas pelo pipeline canônico com as regiões de
# ground-truth anotadas, casando cada predição com o melhor ground-truth
# ainda não casado (por IoU) para calcular cobertura, IoU médio, taxa de
# acerto de label e taxa de hallucination. Chamada por run_ablation para
# cada amostra do benchmark.
def score_observation(
    observation: VisualObservation, ground_truth: tuple[GroundTruthRegion, ...]
) -> QualityMetrics:
    """Pontua uma observação canônica contra suas regiões de ground-truth."""
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


# Roda o pipeline canônico (run_canonical_pipeline) para cada amostra do
# conjunto fixo sob a configuração de uma ablation, pontua cada resultado via
# score_observation, agrega as métricas e mede o tempo total de execução. É
# o ponto de entrada do harness de benchmark/ablation (#175) usado por
# benchmarks/test_harness.py e por scripts de benchmark externos.
def run_ablation(
    dataset: DatasetReference,
    ablation: AblationConfig,
    samples: tuple[tuple[ImageObservation, ImagePayload, tuple[GroundTruthRegion, ...]], ...],
    ports: PerceptionPorts,
) -> BenchmarkReport:
    """Executa o pipeline canônico sob uma ablation sobre um conjunto fixo de amostras."""
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
