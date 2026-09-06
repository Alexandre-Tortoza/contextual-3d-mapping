"""Avaliação de um conjunto de predições contra o conjunto de referência.

Issue: #198.

Este módulo junta as primitivas de ``metrics.py`` em um relatório completo:
métricas de região, de borda, de semântica de vocabulário aberto, de
atributo/condição/material/hazard, de relação e de calibração, todas
quebradas por estrato e acompanhadas de cobertura, contagem de anotações
ignoradas, taxa de falha e intervalos de confiança.

Casos-limite têm tratamento determinístico e documentado:

- **amostra que falhou**: entra na taxa de falha e não contribui com nenhuma
  métrica de qualidade — descartá-la em silêncio inflaria o resultado;
- **anotação ignorada**: absorve a predição que a cobre, sem contar como
  acerto nem como falso positivo;
- **anotação ``unknown``**: conta para detecção (a região existe) e é
  excluída da semântica (não há label de referência para comparar);
- **claim abstida ou sem score**: sai das métricas de confiabilidade e entra
  no relatório de cobertura e abstenção;
- **tarefa sem anotação**: a métrica fica *sem suporte*, e não igual a zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from contextual_mapping_datasets.annotation_manifest import (
    AnnotationCertainty,
    ReferenceManifest,
    SampleAnnotation,
    Split,
)

from .masks import decode_mask
from .metrics import (
    DEFAULT_MATCH_IOU,
    MetricValue,
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
from .prediction import PredictedRegion, PredictionSet, SamplePrediction, ScoredValue, SupportState

#: Tarefas semânticas de conjunto avaliadas por região, na ordem em que
#: aparecem no report. O nome à esquerda é o da métrica; à direita, de onde
#: vêm os valores de referência e de predição.
_SET_TASKS = ("attribute", "condition", "material", "hazard")


# Reúne os parâmetros que definem *como* a avaliação mede, para que um
# report possa ser reproduzido exatamente. Existe porque limiar de IoU,
# tolerância de borda e semente de bootstrap mudam os números e portanto
# fazem parte do resultado.
@dataclass(frozen=True)
class EvaluationConfig:
    """Os parâmetros que definem como a avaliação mede."""

    iou_threshold: float = DEFAULT_MATCH_IOU
    boundary_tolerance: int = 2
    small_region_max_area_px: int = 1024
    calibration_bins: int = 10
    bootstrap_seed: int = 0
    bootstrap_resamples: int = 1000

    # Valida limiares e contagens, para que uma configuração impossível
    # falhe antes de produzir números.
    def __post_init__(self) -> None:
        """Rejeita limiares fora de faixa e parâmetros de bootstrap inválidos."""
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1].")
        if self.boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must not be negative.")
        if self.small_region_max_area_px <= 0:
            raise ValueError("small_region_max_area_px must be positive.")
        if self.calibration_bins <= 0:
            raise ValueError("calibration_bins must be positive.")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive.")


# Registra o resultado de uma região anotada individual. Existe para que a
# agregação por estrato use exatamente as mesmas observações da agregação
# global, e para que um caso de falha possa ser rastreado até a região.
@dataclass(frozen=True)
class RegionOutcome:
    """O que aconteceu com uma região anotada específica."""

    sample_id: str
    annotation_id: str
    area: int
    strata: tuple[str, ...]
    matched: bool
    iou: float | None = None
    boundary_f1: float | None = None
    label_top1: float | None = None
    label_reciprocal_rank: float | None = None
    predicted_region_id: str | None = None


# Registra uma decisão semântica pontuável, com os dois scores possíveis.
# Existe porque as métricas de calibração precisam da mesma decisão vista
# duas vezes: como o produtor a pontuou e como a calibração a pontuou.
@dataclass(frozen=True)
class SemanticDecision:
    """Uma decisão de label pontuável, com score bruto, calibrado e acerto."""

    sample_id: str
    annotation_id: str
    claim_kind: str
    correct: bool
    raw_confidence: float | None
    calibrated_confidence: float | None
    support_state: SupportState


# Registra uma claim prevista que nenhuma anotação sustenta. Existe para que
# a taxa de alucinação seja auditável caso a caso, e não apenas um número
# agregado (critério de aceitação da #199).
@dataclass(frozen=True)
class UnsupportedClaim:
    """Uma claim prevista sobre uma região que nenhuma anotação sustenta."""

    sample_id: str
    predicted_region_id: str
    value: str
    reason: str


# Agrupa tudo que uma única amostra contribuiu para o report. Existe para
# que a agregação por split e por estrato trabalhe sobre observações já
# calculadas, sem reprocessar máscaras.
@dataclass(frozen=True)
class SampleOutcome:
    """A contribuição de uma amostra para o relatório de avaliação."""

    sample_id: str
    split: Split
    strata: tuple[str, ...]
    failed: bool
    failure_reason: str | None
    ignored_annotations: int
    matching: RegionMatching | None
    regions: tuple[RegionOutcome, ...]
    decisions: tuple[SemanticDecision, ...]
    unsupported_claims: tuple[UnsupportedClaim, ...]
    set_metrics: dict[str, MetricValue]
    relation_metrics: dict[str, MetricValue]
    latency_s: float | None = None
    peak_vram_bytes: int | None = None
    model_calls: int | None = None


# Agrupa o relatório completo de uma configuração sobre um split. Existe
# como o objeto único que o report em Markdown (#200) e a análise de
# calibração (#199) consomem.
@dataclass(frozen=True)
class EvaluationReport:
    """O resultado completo de avaliar uma configuração sobre um split."""

    reference_id: str
    run_id: str
    configuration: str
    config_fingerprint: str
    code_revision: str
    split: Split
    sample_count: int
    scored_sample_count: int
    failed_sample_count: int
    ignored_annotation_count: int
    metrics: dict[str, MetricValue]
    intervals: dict[str, tuple[float, float]]
    strata: dict[str, dict[str, MetricValue]]
    calibration: dict[str, dict[str, MetricValue]]
    reliability: dict[str, tuple[dict[str, float | int], ...]]
    abstention: dict[str, MetricValue]
    cost: dict[str, MetricValue]
    unsupported_claims: tuple[UnsupportedClaim, ...] = field(default_factory=tuple)
    outcomes: tuple[SampleOutcome, ...] = field(default_factory=tuple)

    # Filtra as métricas que de fato foram medidas. Existe para que um
    # report nunca apresente uma métrica sem suporte como evidência de
    # superioridade (critério de aceitação da #198).
    def measured_metrics(self) -> dict[str, MetricValue]:
        """Retorna apenas as métricas com suporte, prontas para comparação."""
        return {name: metric for name, metric in self.metrics.items() if metric.measured}


# Ponto de entrada público da avaliação: pontua um conjunto de predições
# contra um split do conjunto de referência. Chamada pelos experimentos de
# calibração (#199) e de ablation (#200).
def evaluate_split(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Avalia ``predictions`` contra as amostras de ``split`` do conjunto de referência.

    Argumentos:
        manifest: o conjunto de referência anotado (#197).
        predictions: as predições versionadas de uma configuração.
        split: o split a avaliar; nenhuma outra amostra é considerada.
        config: parâmetros de medição; o default é reproduzível.
    Retorna:
        o :class:`EvaluationReport` completo.
    Levanta:
        ValueError: se o split não tiver amostras no manifest.
    """
    settings = config or EvaluationConfig()
    samples = manifest.samples_in(split)
    if not samples:
        raise ValueError(f"the reference manifest has no samples in split {split.value!r}.")
    predicted_by_id = predictions.by_sample_id()

    outcomes = tuple(
        _score_sample(sample, predicted_by_id.get(sample.sample_id), settings) for sample in samples
    )
    return _aggregate(manifest, predictions, split, outcomes, settings)


# Pontua uma única amostra, isolando os casos de falha e de ausência de
# predição. Helper de evaluate_split, separado para que cada amostra seja
# testável isoladamente.
def _score_sample(
    sample: SampleAnnotation, prediction: SamplePrediction | None, config: EvaluationConfig
) -> SampleOutcome:
    """Pontua uma amostra anotada contra a sua predição, quando existe."""
    ignored_count = len(sample.regions) - len(sample.scored_regions)
    if prediction is None or prediction.failed:
        reason = (
            "no prediction was produced for this sample"
            if prediction is None
            else prediction.failure_reason
        )
        return SampleOutcome(
            sample_id=sample.sample_id,
            split=sample.split,
            strata=sample.strata,
            failed=True,
            failure_reason=reason,
            ignored_annotations=ignored_count,
            matching=None,
            regions=(),
            decisions=(),
            unsupported_claims=(),
            set_metrics={},
            relation_metrics={},
        )

    reference_masks = [
        decode_mask(region.mask.width, region.mask.height, region.mask.runs) for region in sample.regions
    ]
    prediction_masks = [region.mask() for region in prediction.regions]
    matching = match_regions(
        reference_masks,
        prediction_masks,
        ignored=[region.ignored for region in sample.regions],
        iou_threshold=config.iou_threshold,
    )

    regions: list[RegionOutcome] = []
    decisions: list[SemanticDecision] = []
    matched_reference = {match.reference_index: match for match in matching.matches}
    for index, annotation in enumerate(sample.regions):
        if annotation.ignored:
            continue
        match = matched_reference.get(index)
        strata = _region_strata(annotation, reference_masks[index], config)
        if match is None:
            regions.append(
                RegionOutcome(
                    sample_id=sample.sample_id,
                    annotation_id=annotation.annotation_id,
                    area=int(reference_masks[index].sum()),
                    strata=strata,
                    matched=False,
                )
            )
            continue
        predicted = prediction.regions[match.prediction_index]
        top1, reciprocal = _label_scores(annotation, predicted)
        regions.append(
            RegionOutcome(
                sample_id=sample.sample_id,
                annotation_id=annotation.annotation_id,
                area=int(reference_masks[index].sum()),
                strata=strata,
                matched=True,
                iou=match.iou,
                boundary_f1=float(
                    boundary_metrics(
                        reference_masks,
                        prediction_masks,
                        RegionMatching((match,), (), ()),
                        tolerance=config.boundary_tolerance,
                    ).value
                    or 0.0
                ),
                label_top1=top1,
                label_reciprocal_rank=reciprocal,
                predicted_region_id=predicted.region_id,
            )
        )
        decisions.extend(_label_decisions(sample, annotation, predicted, correct=bool(top1)))

    return SampleOutcome(
        sample_id=sample.sample_id,
        split=sample.split,
        strata=sample.strata,
        failed=False,
        failure_reason=None,
        ignored_annotations=ignored_count,
        matching=matching,
        regions=tuple(regions),
        decisions=tuple(decisions),
        unsupported_claims=_unsupported_claims(sample, prediction, matching),
        set_metrics=_sample_set_metrics(sample, prediction, matching),
        relation_metrics=_relation_metrics(sample, prediction, matching),
        latency_s=prediction.latency_s,
        peak_vram_bytes=prediction.peak_vram_bytes,
        model_calls=prediction.model_calls,
    )


# Deriva os estratos de uma região anotada: os declarados pela anotação mais
# o tamanho medido. Existe para que o report separe região pequena de região
# comum sem que o anotador precise classificá-las manualmente.
def _region_strata(annotation, mask: np.ndarray, config: EvaluationConfig) -> tuple[str, ...]:
    """Retorna os estratos aplicáveis a uma região anotada."""
    area = int(mask.sum())
    size = "small_region" if area <= config.small_region_max_area_px else "ordinary_region"
    strata = [size]
    if annotation.visibility:
        strata.append(f"visibility:{annotation.visibility}")
    if annotation.certainty is AnnotationCertainty.AMBIGUOUS:
        strata.append("ambiguous_annotation")
    return tuple(strata)


# Pontua a semântica de uma região casada sob vocabulário aberto, tratando
# anotação desconhecida como não pontuável. Helper de _score_sample.
def _label_scores(annotation, predicted: PredictedRegion) -> tuple[float | None, float | None]:
    """Retorna ``(top1, reciprocal_rank)``, ou ``(None, None)`` sem label de referência."""
    if annotation.certainty is AnnotationCertainty.UNKNOWN or not annotation.labels:
        return None, None
    ranked = [value.value for value in _rank(predicted.labels)]
    top1, reciprocal = open_vocabulary_scores(annotation.labels, ranked)
    return top1, reciprocal


# Converte os labels previstos de uma região casada em decisões pontuáveis.
# Existe porque a métrica de calibração precisa do par (score, acerto) de
# cada decisão, com bruto e calibrado lado a lado.
def _label_decisions(
    sample: SampleAnnotation, annotation, predicted: PredictedRegion, *, correct: bool
) -> list[SemanticDecision]:
    """Retorna a decisão de label de topo desta região, quando ela é pontuável."""
    if annotation.certainty is AnnotationCertainty.UNKNOWN or not annotation.labels:
        return []
    ranked = _rank(predicted.labels)
    if not ranked:
        return []
    best = ranked[0]
    return [
        SemanticDecision(
            sample_id=sample.sample_id,
            annotation_id=annotation.annotation_id,
            claim_kind="label",
            correct=correct,
            raw_confidence=best.raw_confidence,
            calibrated_confidence=best.calibrated_confidence,
            support_state=best.support_state,
        )
    ]


# Ordena valores previstos por confiança, preferindo o score calibrado e
# caindo para o bruto. Um valor sem score nenhum vai para o fim, sem que a
# ausência seja convertida em número.
def _rank(values: Sequence[ScoredValue]) -> tuple[ScoredValue, ...]:
    """Ordena valores previstos por confiança decrescente, sem inventar score."""
    def key(item: tuple[int, ScoredValue]) -> tuple[int, float, int]:
        index, value = item
        score = value.calibrated_confidence
        if score is None:
            score = value.raw_confidence
        return (0, -score, index) if score is not None else (1, 0.0, index)

    return tuple(value for _, value in sorted(enumerate(values), key=key))


# Identifica claims previstas que nenhuma anotação sustenta. Ambiguidade não
# entra: uma predição sobre uma região anotada como ambígua ou desconhecida
# tem anotação, apenas incerta. Só uma região prevista que não casou com
# nada (nem com uma região ignorada) e ainda assim afirma um label conta.
def _unsupported_claims(
    sample: SampleAnnotation, prediction: SamplePrediction, matching: RegionMatching
) -> tuple[UnsupportedClaim, ...]:
    """Retorna as claims previstas sem nenhuma anotação que as sustente."""
    claims: list[UnsupportedClaim] = []
    for index in matching.unmatched_prediction:
        region = prediction.regions[index]
        for value in region.labels:
            claims.append(
                UnsupportedClaim(
                    sample_id=sample.sample_id,
                    predicted_region_id=region.region_id,
                    value=value.value,
                    reason="predicted region matched no annotated region above the IoU threshold",
                )
            )
    return tuple(claims)


# Agrega as métricas de conjunto (atributo, condição, material, hazard) da
# amostra, unindo os valores de todas as regiões casadas. Helper de
# _score_sample.
def _sample_set_metrics(
    sample: SampleAnnotation, prediction: SamplePrediction, matching: RegionMatching
) -> dict[str, MetricValue]:
    """Retorna as métricas de conjunto da amostra, por tarefa semântica."""
    reference: dict[str, list[str]] = {task: [] for task in _SET_TASKS}
    predicted: dict[str, list[str]] = {task: [] for task in _SET_TASKS}
    for match in matching.matches:
        annotation = sample.regions[match.reference_index]
        region = prediction.regions[match.prediction_index]
        reference["attribute"].extend(annotation.attributes)
        reference["hazard"].extend(annotation.hazards)
        if annotation.condition:
            reference["condition"].append(annotation.condition)
        if annotation.material:
            reference["material"].append(annotation.material)
        predicted["attribute"].extend(value.value for value in region.attributes)
        predicted["hazard"].extend(value.value for value in region.hazards)
        predicted["condition"].extend(value.value for value in region.conditions)
        predicted["material"].extend(value.value for value in region.materials)

    metrics: dict[str, MetricValue] = {}
    for task in _SET_TASKS:
        metrics.update(set_metrics(task, reference[task], predicted[task]))
    return metrics


# Pontua as relações previstas contra as anotadas, traduzindo os ids de
# região prevista para os ids de anotação através do casamento de regiões.
# Existe porque uma relação só é comparável depois que os dois lados falam
# do mesmo par de regiões.
def _relation_metrics(
    sample: SampleAnnotation, prediction: SamplePrediction, matching: RegionMatching
) -> dict[str, MetricValue]:
    """Retorna precisão, recall e F1 das relações previstas."""
    annotation_by_prediction = {
        prediction.regions[match.prediction_index].region_id: sample.regions[
            match.reference_index
        ].annotation_id
        for match in matching.matches
    }
    reference_triples = [
        f"{relation.subject_annotation_id}|{relation.predicate}|{relation.object_annotation_id}"
        for relation in sample.relations
    ]
    predicted_triples = []
    for relation in prediction.relations:
        subject = annotation_by_prediction.get(relation.subject_region_id)
        target = annotation_by_prediction.get(relation.object_region_id)
        if subject is None or target is None:
            continue
        predicted_triples.append(f"{subject}|{relation.predicate}|{target}")
    return set_metrics("relation", reference_triples, predicted_triples)


# Agrega as contribuições de todas as amostras em um relatório único, com
# estratos, calibração, abstenção, custo e intervalos de confiança. Helper
# final de evaluate_split.
def _aggregate(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    outcomes: tuple[SampleOutcome, ...],
    config: EvaluationConfig,
) -> EvaluationReport:
    """Combina as amostras pontuadas no relatório final da configuração."""
    scored = tuple(outcome for outcome in outcomes if not outcome.failed)
    all_regions = tuple(region for outcome in scored for region in outcome.regions)
    decisions = tuple(decision for outcome in scored for decision in outcome.decisions)

    combined = RegionMatching(
        matches=tuple(
            match for outcome in scored if outcome.matching for match in outcome.matching.matches
        ),
        unmatched_reference=tuple(
            index
            for outcome in scored
            if outcome.matching
            for index in outcome.matching.unmatched_reference
        ),
        unmatched_prediction=tuple(
            index
            for outcome in scored
            if outcome.matching
            for index in outcome.matching.unmatched_prediction
        ),
    )
    metrics = detection_metrics(combined)
    metrics["boundary_f1"] = _mean_metric(
        "boundary_f1", [region.boundary_f1 for region in all_regions if region.matched]
    )
    metrics["label_top1"] = _mean_metric(
        "label_top1", [region.label_top1 for region in all_regions]
    )
    metrics["label_mean_reciprocal_rank"] = _mean_metric(
        "label_mean_reciprocal_rank", [region.label_reciprocal_rank for region in all_regions]
    )
    metrics["sample_failure_rate"] = MetricValue(
        "sample_failure_rate",
        (len(outcomes) - len(scored)) / len(outcomes),
        len(outcomes),
    )
    for task in (*_SET_TASKS, "relation"):
        for suffix in ("precision", "recall", "f1"):
            name = f"{task}_{suffix}"
            source = "relation_metrics" if task == "relation" else "set_metrics"
            metrics[name] = _mean_metric(
                name,
                [
                    getattr(outcome, source)[name].value
                    for outcome in scored
                    if name in getattr(outcome, source) and getattr(outcome, source)[name].measured
                ],
            )

    unsupported = tuple(claim for outcome in scored for claim in outcome.unsupported_claims)
    predicted_region_total = len(combined.matches) + len(combined.unmatched_prediction)
    metrics["unsupported_claim_rate"] = MetricValue(
        "unsupported_claim_rate",
        len(combined.unmatched_prediction) / predicted_region_total if predicted_region_total else None,
        predicted_region_total,
        detail="predicted regions with no annotated counterpart above the IoU threshold",
    )

    return EvaluationReport(
        reference_id=manifest.reference_id,
        run_id=predictions.run_id,
        configuration=predictions.configuration,
        config_fingerprint=predictions.config_fingerprint,
        code_revision=predictions.code_revision,
        split=split,
        sample_count=len(outcomes),
        scored_sample_count=len(scored),
        failed_sample_count=len(outcomes) - len(scored),
        ignored_annotation_count=sum(outcome.ignored_annotations for outcome in outcomes),
        metrics=metrics,
        intervals=_intervals(all_regions, config),
        strata=_strata_metrics(outcomes, all_regions),
        calibration=_calibration_metrics(decisions, config),
        reliability=_reliability(decisions, config),
        abstention=_abstention(decisions),
        cost=_cost(scored),
        unsupported_claims=unsupported,
        outcomes=outcomes,
    )


# Calcula intervalos de confiança por bootstrap para as métricas por região.
# Existe porque a média de um conjunto de referência pequeno é instável, e a
# #198 exige que a incerteza acompanhe o número.
def _intervals(
    regions: tuple[RegionOutcome, ...], config: EvaluationConfig
) -> dict[str, tuple[float, float]]:
    """Retorna os intervalos de bootstrap das métricas por região."""
    sources = {
        "matched_mask_iou": [region.iou for region in regions if region.matched],
        "boundary_f1": [region.boundary_f1 for region in regions if region.matched],
        "label_top1": [region.label_top1 for region in regions],
    }
    intervals: dict[str, tuple[float, float]] = {}
    for name, values in sources.items():
        observed = [value for value in values if value is not None]
        interval = bootstrap_interval(
            observed, seed=config.bootstrap_seed, resamples=config.bootstrap_resamples
        )
        if interval is not None:
            intervals[name] = interval
    return intervals


# Quebra as métricas por estrato, combinando os estratos declarados pela
# amostra com os derivados por região. Existe porque a #198 exige um recorte
# por estrato para toda métrica agregada.
def _strata_metrics(
    outcomes: tuple[SampleOutcome, ...], regions: tuple[RegionOutcome, ...]
) -> dict[str, dict[str, MetricValue]]:
    """Retorna as métricas de região agrupadas por estrato."""
    sample_strata = {
        outcome.sample_id: outcome.strata for outcome in outcomes if not outcome.failed
    }
    grouped: dict[str, list[RegionOutcome]] = {}
    for region in regions:
        for stratum in (*region.strata, *sample_strata.get(region.sample_id, ())):
            grouped.setdefault(stratum, []).append(region)

    return {
        stratum: {
            "region_recall": MetricValue(
                "region_recall",
                sum(1 for region in members if region.matched) / len(members),
                len(members),
            ),
            "matched_mask_iou": _mean_metric(
                "matched_mask_iou", [region.iou for region in members if region.matched]
            ),
            "boundary_f1": _mean_metric(
                "boundary_f1", [region.boundary_f1 for region in members if region.matched]
            ),
            "label_top1": _mean_metric("label_top1", [region.label_top1 for region in members]),
        }
        for stratum, members in sorted(grouped.items())
    }


# Calcula ECE e Brier separadamente para o score bruto e para o calibrado.
# Existe porque a #199 exige que confiabilidade bruta e calibrada sejam
# reportadas lado a lado, nunca fundidas em um número só.
def _calibration_metrics(
    decisions: tuple[SemanticDecision, ...], config: EvaluationConfig
) -> dict[str, dict[str, MetricValue]]:
    """Retorna ECE e Brier para os modos ``raw`` e ``calibrated``."""
    report: dict[str, dict[str, MetricValue]] = {}
    for mode in ("raw", "calibrated"):
        scores, correct = _scored_pairs(decisions, calibrated=mode == "calibrated")
        report[mode] = {
            "expected_calibration_error": expected_calibration_error(
                scores, correct, bins=config.calibration_bins
            ),
            "brier_score": brier_score(scores, correct),
            "scored_decisions": MetricValue(
                "scored_decisions", float(len(scores)) if decisions else None, len(decisions)
            ),
        }
    return report


# Constrói as tabelas de confiabilidade dos dois modos, usadas pelos
# diagramas do report.
def _reliability(
    decisions: tuple[SemanticDecision, ...], config: EvaluationConfig
) -> dict[str, tuple[dict[str, float | int], ...]]:
    """Retorna as tabelas de confiabilidade dos modos ``raw`` e ``calibrated``."""
    return {
        mode: reliability_table(
            *_scored_pairs(decisions, calibrated=mode == "calibrated"), bins=config.calibration_bins
        )
        for mode in ("raw", "calibrated")
    }


# Resume cobertura, abstenção e risco seletivo das decisões calibradas.
# Existe para que claims sem score continuem visíveis no report, em vez de
# desaparecerem do denominador.
def _abstention(decisions: tuple[SemanticDecision, ...]) -> dict[str, MetricValue]:
    """Retorna cobertura, abstenção, taxa sem score e risco seletivo."""
    correct = [decision.correct for decision in decisions]
    scored = [decision.calibrated_confidence is not None for decision in decisions]
    abstained = [decision.support_state is SupportState.ABSTAINED for decision in decisions]
    return abstention_report(correct, scored, abstained=abstained)


# Agrega o custo medido por amostra. Existe porque a #200 compara qualidade
# *e* custo, e um ganho de qualidade só é decidível junto do que ele custou.
def _cost(outcomes: tuple[SampleOutcome, ...]) -> dict[str, MetricValue]:
    """Retorna latência média, VRAM de pico e chamadas de modelo por amostra."""
    latencies = [outcome.latency_s for outcome in outcomes if outcome.latency_s is not None]
    vram = [outcome.peak_vram_bytes for outcome in outcomes if outcome.peak_vram_bytes is not None]
    calls = [outcome.model_calls for outcome in outcomes if outcome.model_calls is not None]
    return {
        "mean_latency_s": _mean_metric("mean_latency_s", latencies),
        "peak_vram_bytes": MetricValue(
            "peak_vram_bytes", float(max(vram)) if vram else None, len(vram)
        ),
        "mean_model_calls": _mean_metric("mean_model_calls", calls),
    }


# Seleciona os pares (score, acerto) das decisões que têm score no modo
# pedido. Existe para que ausência de score seja excluída do cálculo de
# confiabilidade, e não convertida em número.
def _scored_pairs(
    decisions: tuple[SemanticDecision, ...], *, calibrated: bool
) -> tuple[list[float], list[bool]]:
    """Retorna os pares pontuados do modo pedido, descartando os sem score."""
    scores: list[float] = []
    correct: list[bool] = []
    for decision in decisions:
        score = decision.calibrated_confidence if calibrated else decision.raw_confidence
        if score is None:
            continue
        scores.append(float(score))
        correct.append(decision.correct)
    return scores, correct


# Constrói a média de uma lista de observações opcionais como MetricValue,
# tratando lista vazia como ausência de suporte. Helper compartilhado por
# toda a agregação deste módulo.
def _mean_metric(name: str, values: Sequence[float | int | None]) -> MetricValue:
    """Retorna a média das observações informadas, ou uma métrica sem suporte."""
    observed = [float(value) for value in values if value is not None]
    if not observed:
        return MetricValue(name, None, 0)
    return MetricValue(name, float(np.mean(observed)), len(observed))
