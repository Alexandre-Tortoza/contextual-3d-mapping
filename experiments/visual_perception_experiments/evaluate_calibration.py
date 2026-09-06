"""Avaliação de calibração, abstenção e claims sem suporte.

Issue: #199.

O experimento responde três perguntas separadas, e mantém as respostas
separadas:

1. **Confiabilidade**: um score de 0,8 acerta 80% das vezes? Medida em
   ``raw`` e em ``calibrated``, lado a lado e nunca fundidas — se a
   calibração melhora algo, é aqui que aparece.
2. **Abstenção**: quanto o sistema se recusa a pontuar, e o que ele erra
   entre o que pontuou (risco seletivo). Um sistema que se abstém de tudo
   tem risco zero e cobertura zero; os dois números só significam algo
   juntos.
3. **Claim sem suporte**: quanto o sistema afirma sobre regiões que nenhuma
   anotação sustenta.

## Critério de claim sem suporte

Uma claim é contada como *sem suporte* quando a região prevista não casa com
nenhuma região anotada acima do limiar de IoU, e não foi absorvida por uma
região anotada como ignorada.

Ambiguidade **não** conta como erro. Uma predição sobre uma região anotada
como ``ambiguous`` acerta se bater com qualquer um dos labels aceitáveis;
uma sobre uma região ``unknown`` não é pontuada semanticamente, porque não
há referência para comparar. Confundir "o anotador não soube dizer" com "o
modelo alucinou" produziria uma taxa de alucinação inflada e sem sentido.

## Quando não há anotação

Sem conjunto de referência revisado, nada de (1) e (3) é calculável.
:func:`operational_summary` mede o que independe de anotação — cobertura de
evidência por slot, distribuição de estado de suporte, taxa de falha e custo
— para que uma execução real ainda produza evidência, declarada como tal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from contextual_mapping_datasets import (
    ReferenceManifest,
    Split,
    load_reference_manifest,
)
from contextual_mapping_datasets.annotation_manifest import AnnotationCertainty
from visual_perception_evaluation.masks import decode_mask
from visual_perception_evaluation.metrics import (
    MetricValue,
    brier_score,
    expected_calibration_error,
    match_regions,
)
from visual_perception_evaluation.prediction import (
    PredictionSet,
    SamplePrediction,
    load_predictions,
)
from visual_perception_evaluation.report import format_metric
from visual_perception_evaluation.suite import (
    EvaluationConfig,
    EvaluationReport,
    evaluate_split,
)

from .paths import REFERENCE_MANIFEST, RESULTS_ROOT


# Registra um caso de falha auditável, ligando-o à imagem, à região, à
# predição, à evidência e à proveniência de calibração. Existe porque a #199
# exige que cada falha possa ser inspecionada, e não apenas contada.
@dataclass(frozen=True)
class FailureCase:
    """Uma falha individual, com tudo que é preciso para auditá-la."""

    kind: str
    sample_id: str
    image_uri: str
    annotation_id: str | None = None
    predicted_region_id: str | None = None
    predicted_value: str | None = None
    reference_labels: tuple[str, ...] = ()
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    support_state: str | None = None
    support_reason: str | None = None
    evidence_slots: tuple[str, ...] = ()
    evidence_artifacts: tuple[str, ...] = ()
    source: str | None = None
    calibration_version: str | None = None
    calibration_artifact: str | None = None
    strata: tuple[str, ...] = ()


# Agrupa o estudo completo de confiabilidade de uma configuração. Existe
# como o objeto que o relatório renderiza e que o experimento de ablation
# (#200) consome.
@dataclass(frozen=True)
class CalibrationStudy:
    """O resultado do estudo de calibração, abstenção e claims sem suporte."""

    configuration: str
    split: Split
    report: EvaluationReport
    by_claim_kind: dict[str, dict[str, MetricValue]]
    by_source: dict[str, dict[str, MetricValue]]
    by_stratum: dict[str, dict[str, MetricValue]]
    by_evidence_slot: dict[str, dict[str, MetricValue]]
    failures: tuple[FailureCase, ...] = ()
    operational: dict[str, Any] = field(default_factory=dict)


# Mede o que não depende de anotação nenhuma. Existe para que uma execução
# real sobre um conjunto ainda em revisão produza evidência legítima, em vez
# de nenhum número.
def operational_summary(predictions: PredictionSet) -> dict[str, Any]:
    """Resume cobertura de evidência, estado de suporte e custo, sem anotação.

    Argumentos:
        predictions: as predições de uma configuração.
    Retorna:
        um dict com contagens e medianas de custo, todas derivadas apenas da
        própria predição.
    """
    scored = [sample for sample in predictions.samples if not sample.failed]
    regions = [region for sample in scored for region in sample.regions]
    values = [value for region in regions for value in region.labels]
    slots = Counter(slot for region in regions for slot in region.evidence_slots)
    states = Counter(value.support_state.value for value in values)
    raw_scores = [
        value.raw_confidence for value in values if value.raw_confidence is not None
    ]
    calibrated_scores = [
        value.calibrated_confidence
        for value in values
        if value.calibrated_confidence is not None
    ]
    latencies = [sample.latency_s for sample in scored if sample.latency_s is not None]
    vram = [
        sample.peak_vram_bytes
        for sample in scored
        if sample.peak_vram_bytes is not None
    ]
    calls = [sample.model_calls for sample in scored if sample.model_calls is not None]
    return {
        "configuration": predictions.configuration,
        "config_fingerprint": predictions.config_fingerprint,
        "code_revision": predictions.code_revision,
        "hardware": predictions.hardware,
        "samples": len(predictions.samples),
        "failed_samples": len(predictions.samples) - len(scored),
        "regions": len(regions),
        "regions_per_sample": len(regions) / len(scored) if scored else 0.0,
        "label_claims": len(values),
        "support_states": dict(sorted(states.items())),
        "evidence_slot_coverage": {
            name: count / len(regions) for name, count in sorted(slots.items())
        }
        if regions
        else {},
        "raw_scored_fraction": len(raw_scores) / len(values) if values else 0.0,
        "calibrated_scored_fraction": len(calibrated_scores) / len(values)
        if values
        else 0.0,
        "mean_raw_confidence": float(np.mean(raw_scores)) if raw_scores else None,
        "mean_calibrated_confidence": float(np.mean(calibrated_scores))
        if calibrated_scores
        else None,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "peak_vram_bytes": max(vram) if vram else None,
        "mean_model_calls": float(np.mean(calls)) if calls else None,
    }


# Executa o estudo completo sobre um split. Ponto de entrada do experimento,
# reutilizado pela matriz de ablation (#200).
def study_calibration(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    config: EvaluationConfig | None = None,
) -> CalibrationStudy:
    """Avalia confiabilidade, abstenção e claims sem suporte de uma configuração.

    Argumentos:
        manifest: o conjunto de referência anotado.
        predictions: as predições da configuração.
        split: o split a avaliar.
        config: parâmetros de medição.
    Retorna:
        o :class:`CalibrationStudy` completo.
    """
    settings = config or EvaluationConfig()
    report = evaluate_split(manifest, predictions, split, settings)
    decisions = tuple(
        decision
        for outcome in report.outcomes
        if not outcome.failed
        for decision in outcome.decisions
    )
    samples_by_id = {sample.sample_id: sample for sample in manifest.samples_in(split)}
    strata_by_annotation = {}
    for outcome in report.outcomes:
        sample = samples_by_id[outcome.sample_id]
        for region in outcome.regions:
            strata_by_annotation[(region.sample_id, region.annotation_id)] = (
                *region.strata,
                *sample.strata,
                *sample.capture_conditions,
                f"scene:{manifest.dataset_id}",
            )
    return CalibrationStudy(
        configuration=predictions.configuration,
        split=split,
        report=report,
        by_claim_kind=_grouped_reliability(
            decisions,
            key=lambda decision: decision.claim_kind,
            bins=settings.calibration_bins,
        ),
        by_source=_source_reliability(
            manifest, predictions, split, settings, decisions
        ),
        by_stratum=_grouped_reliability(
            decisions,
            key=lambda decision: strata_by_annotation.get(
                (decision.sample_id, decision.annotation_id), ("unknown_stratum",)
            ),
            bins=settings.calibration_bins,
        ),
        by_evidence_slot=_evidence_slot_reliability(
            manifest, predictions, split, settings, decisions
        ),
        failures=collect_failures(manifest, predictions, split, settings),
        operational=operational_summary(predictions),
    )


# Reúne os casos de falha auditáveis do split. Existe para que o relatório
# possa mostrar exemplos concretos, e não apenas taxas.
def collect_failures(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    config: EvaluationConfig,
    *,
    limit: int = 200,
) -> tuple[FailureCase, ...]:
    """Coleta os casos de falha, ligando cada um à imagem, região e evidência."""
    by_sample = predictions.by_sample_id()
    failures: list[FailureCase] = []

    for sample in manifest.samples_in(split):
        prediction = by_sample.get(sample.sample_id)
        if prediction is None or prediction.failed:
            failures.append(
                FailureCase(
                    kind="sample_failed",
                    sample_id=sample.sample_id,
                    image_uri=sample.artifact.uri,
                    support_reason=(
                        "no prediction was produced"
                        if prediction is None
                        else prediction.failure_reason
                    ),
                )
            )
            continue
        failures.extend(
            _sample_failures(sample, prediction, config, manifest.dataset_id)
        )
        if len(failures) >= limit:
            break
    return tuple(failures[:limit])


# Reúne as falhas de uma única amostra: label errado, região não detectada e
# claim sem suporte. Helper de collect_failures.
def _sample_failures(
    sample: Any,
    prediction: SamplePrediction,
    config: EvaluationConfig,
    dataset_id: str,
) -> list[FailureCase]:
    """Coleta as falhas de uma amostra específica."""
    reference_masks = [
        decode_mask(region.mask.width, region.mask.height, region.mask.runs)
        for region in sample.regions
    ]
    matching = match_regions(
        reference_masks,
        [region.mask() for region in prediction.regions],
        ignored=[region.ignored for region in sample.regions],
        iou_threshold=config.iou_threshold,
    )
    failures: list[FailureCase] = []

    for match in matching.matches:
        annotation = sample.regions[match.reference_index]
        predicted = prediction.regions[match.prediction_index]
        if annotation.certainty is AnnotationCertainty.UNKNOWN or not annotation.labels:
            continue
        scored = sorted(predicted.labels, key=lambda value: -_decision_score(value))
        if not scored:
            continue
        best = scored[0]
        acceptable = {label.strip().lower() for label in annotation.labels}
        if best.value.strip().lower() in acceptable:
            continue
        failures.append(
            FailureCase(
                kind="wrong_label",
                sample_id=sample.sample_id,
                image_uri=sample.artifact.uri,
                annotation_id=annotation.annotation_id,
                predicted_region_id=predicted.region_id,
                predicted_value=best.value,
                reference_labels=annotation.labels,
                raw_confidence=best.raw_confidence,
                calibrated_confidence=best.calibrated_confidence,
                support_state=best.support_state.value,
                support_reason=best.reason,
                evidence_slots=predicted.evidence_slots,
                evidence_artifacts=best.evidence_artifacts,
                source=best.source,
                calibration_version=best.calibration_version,
                calibration_artifact=best.calibration_artifact,
                strata=_failure_strata(sample, annotation, config, dataset_id),
            )
        )

    for index in matching.unmatched_reference:
        annotation = sample.regions[index]
        failures.append(
            FailureCase(
                kind="missed_region",
                sample_id=sample.sample_id,
                image_uri=sample.artifact.uri,
                annotation_id=annotation.annotation_id,
                reference_labels=annotation.labels,
                strata=_failure_strata(sample, annotation, config, dataset_id),
            )
        )

    for index in matching.unmatched_prediction:
        predicted = prediction.regions[index]
        for value in predicted.labels:
            failures.append(
                FailureCase(
                    kind="unsupported_claim",
                    sample_id=sample.sample_id,
                    image_uri=sample.artifact.uri,
                    predicted_region_id=predicted.region_id,
                    predicted_value=value.value,
                    raw_confidence=value.raw_confidence,
                    calibrated_confidence=value.calibrated_confidence,
                    support_state=value.support_state.value,
                    support_reason=value.reason,
                    evidence_slots=predicted.evidence_slots,
                    evidence_artifacts=value.evidence_artifacts,
                    source=value.source,
                    calibration_version=value.calibration_version,
                    calibration_artifact=value.calibration_artifact,
                    strata=(
                        *sample.strata,
                        *sample.capture_conditions,
                        f"scene:{dataset_id}",
                    ),
                )
            )
    return failures


# Seleciona o score efetivamente usado para ordenar uma hipótese. Existe para
# preservar 0,0 como score válido em vez de tratá-lo como ausência por truthiness.
def _decision_score(value: Any) -> float:
    """Retorna score calibrado, bruto ou -1 quando ambos estão ausentes."""
    if value.calibrated_confidence is not None:
        return float(value.calibrated_confidence)
    if value.raw_confidence is not None:
        return float(value.raw_confidence)
    return -1.0


# Deriva os recortes auditáveis de uma falha anotada. Existe para que os
# exemplos e os agregados usem o mesmo vocabulário de tamanho, visibilidade,
# condição de captura e tipo de cena.
def _failure_strata(
    sample: Any, annotation: Any, config: EvaluationConfig, dataset_id: str
) -> tuple[str, ...]:
    """Retorna os estratos aplicáveis a uma falha sobre uma região anotada."""
    size = (
        "small_region"
        if annotation.mask.area <= config.small_region_max_area_px
        else "ordinary_region"
    )
    visibility = (
        (f"visibility:{annotation.visibility}",) if annotation.visibility else ()
    )
    return (
        size,
        *visibility,
        *sample.strata,
        *sample.capture_conditions,
        f"scene:{dataset_id}",
    )


# Agrupa as decisões por uma chave e calcula ECE e Brier em cada grupo, nos
# dois modos. Existe para que os recortes por tipo de claim e por estrato
# compartilhem exatamente a mesma matemática.
def _grouped_reliability(
    decisions: tuple[Any, ...], *, key: Any, bins: int
) -> dict[str, dict[str, MetricValue]]:
    """Calcula confiabilidade bruta e calibrada por grupo de decisões."""
    grouped: dict[str, list[Any]] = {}
    for decision in decisions:
        value = key(decision)
        for name in (value,) if isinstance(value, str) else value:
            grouped.setdefault(name, []).append(decision)

    result: dict[str, dict[str, MetricValue]] = {}
    for name, members in sorted(grouped.items()):
        entry: dict[str, MetricValue] = {}
        for mode in ("raw", "calibrated"):
            scores, correct = [], []
            for decision in members:
                score = (
                    decision.calibrated_confidence
                    if mode == "calibrated"
                    else decision.raw_confidence
                )
                if score is None:
                    continue
                scores.append(float(score))
                correct.append(decision.correct)
            entry[f"{mode}_expected_calibration_error"] = expected_calibration_error(
                scores, correct, bins=bins
            )
            entry[f"{mode}_brier_score"] = brier_score(scores, correct)
        covered = sum(
            1 for decision in members if decision.calibrated_confidence is not None
        )
        entry["decisions"] = MetricValue("decisions", float(len(members)), len(members))
        entry["calibrated_coverage"] = MetricValue(
            "calibrated_coverage",
            covered / len(members) if members else None,
            len(members),
        )
        result[name] = entry
    return result


# Calcula confiabilidade por fonte original da claim. Existe porque scores de
# produtores distintos não são comparáveis sem mostrar qual fonte os emitiu.
def _source_reliability(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    config: EvaluationConfig,
    decisions: tuple[Any, ...],
) -> dict[str, dict[str, MetricValue]]:
    """Calcula confiabilidade agrupada pela fonte do label selecionado."""
    values = _selected_values_by_annotation(manifest, predictions, split, config)

    # Resolve a chave em uma função nomeada para ler o mapa uma única vez e
    # manter a política de fallback explícita.
    def source_of(decision: Any) -> str:
        """Retorna a fonte da decisão ou o grupo explícito de fonte ausente."""
        value = values.get((decision.sample_id, decision.annotation_id))
        return value.source if value is not None and value.source else "unknown_source"

    return _grouped_reliability(
        decisions,
        key=source_of,
        bins=config.calibration_bins,
    )


# Liga cada anotação casada ao label escolhido naquela região prevista.
# Existe como fonte única para breakdowns que dependem da metadata da claim.
def _selected_values_by_annotation(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    config: EvaluationConfig,
) -> dict[tuple[str, str], Any]:
    """Retorna o melhor valor previsto por identidade de anotação casada."""
    by_sample = predictions.by_sample_id()
    selected: dict[tuple[str, str], Any] = {}
    for sample in manifest.samples_in(split):
        prediction = by_sample.get(sample.sample_id)
        if prediction is None or prediction.failed:
            continue
        reference_masks = [
            decode_mask(region.mask.width, region.mask.height, region.mask.runs)
            for region in sample.regions
        ]
        matching = match_regions(
            reference_masks,
            [region.mask() for region in prediction.regions],
            ignored=[region.ignored for region in sample.regions],
            iou_threshold=config.iou_threshold,
        )
        for match in matching.matches:
            annotation = sample.regions[match.reference_index]
            labels = prediction.regions[match.prediction_index].labels
            if labels:
                selected[(sample.sample_id, annotation.annotation_id)] = max(
                    labels, key=_decision_score
                )
    return selected


# Calcula a confiabilidade recortada por slot de evidência disponível na
# região. Existe porque a #199 pede a análise de falha por slot: se as
# regiões com crop contextual erram menos, isso precisa aparecer.
def _evidence_slot_reliability(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split,
    config: EvaluationConfig,
    decisions: tuple[Any, ...],
) -> dict[str, dict[str, MetricValue]]:
    """Calcula confiabilidade agrupada pelos slots de evidência das regiões."""
    by_sample = predictions.by_sample_id()
    slots_by_annotation: dict[tuple[str, str], tuple[str, ...]] = {}
    for sample in manifest.samples_in(split):
        prediction = by_sample.get(sample.sample_id)
        if prediction is None or prediction.failed:
            continue
        reference_masks = [
            decode_mask(region.mask.width, region.mask.height, region.mask.runs)
            for region in sample.regions
        ]
        matching = match_regions(
            reference_masks,
            [region.mask() for region in prediction.regions],
            ignored=[region.ignored for region in sample.regions],
            iou_threshold=config.iou_threshold,
        )
        for match in matching.matches:
            annotation = sample.regions[match.reference_index]
            predicted = prediction.regions[match.prediction_index]
            slots_by_annotation[(sample.sample_id, annotation.annotation_id)] = (
                predicted.evidence_slots or ("no_evidence_slot",)
            )
    return _grouped_reliability(
        decisions,
        key=lambda decision: slots_by_annotation.get(
            (decision.sample_id, decision.annotation_id), ("no_evidence_slot",)
        ),
        bins=config.calibration_bins,
    )


# Renderiza o estudo em Markdown. Existe para que o resultado seja legível
# sem abrir o JSON, mantendo exatamente os mesmos números.
def render_study(study: CalibrationStudy) -> str:
    """Renderiza o estudo de calibração em Markdown."""
    report = study.report
    lines = [
        f"# Calibração, abstenção e claims sem suporte — `{study.configuration}` (#199)",
        "",
        f"- conjunto de referência: `{report.reference_id}` / split `{study.split.value}`",
        f"- run: `{report.run_id}` — revisão `{report.code_revision}`",
        f"- fingerprint de configuração: `{report.config_fingerprint}`",
        f"- hardware: {study.operational.get('hardware', 'não registrado')}",
        (
            f"- amostras avaliadas: {report.scored_sample_count}/{report.sample_count} "
            f"(falhas: {report.failed_sample_count})"
        ),
        "",
        "## Confiabilidade: bruto versus calibrado",
        "",
        "| modo | ECE | Brier | decisões pontuadas |",
        "| --- | --- | --- | --- |",
    ]
    for mode in ("raw", "calibrated"):
        metrics = report.calibration[mode]
        lines.append(
            f"| `{mode}` | {format_metric(metrics['expected_calibration_error'])} "
            f"| {format_metric(metrics['brier_score'])} "
            f"| {format_metric(metrics['scored_decisions'])} |"
        )

    lines += ["", "## Abstenção e risco seletivo", ""]
    lines += [
        f"- `{name}`: {format_metric(metric)}"
        for name, metric in sorted(report.abstention.items())
    ]

    lines += [
        "",
        "## Claims sem suporte",
        "",
        f"- taxa: {format_metric(report.metrics['unsupported_claim_rate'])}",
        (
            "- critério: região prevista sem correspondência anotada acima do limiar de IoU, "
            "não absorvida por uma região ignorada."
        ),
        (
            "- ambiguidade não conta como erro: uma predição sobre região `ambiguous` acerta "
            "com qualquer label aceitável, e regiões `unknown` não são pontuadas semanticamente."
        ),
    ]

    for title, grouped in (
        ("Por tipo de claim", study.by_claim_kind),
        ("Por fonte", study.by_source),
        ("Por estrato", study.by_stratum),
        ("Por slot de evidência", study.by_evidence_slot),
    ):
        if not grouped:
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| grupo | decisões | cobertura calibrada | ECE bruto | ECE calibrado |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, metrics in grouped.items():
            lines.append(
                f"| `{name}` | {format_metric(metrics['decisions'])} "
                f"| {format_metric(metrics['calibrated_coverage'])} "
                f"| {format_metric(metrics['raw_expected_calibration_error'])} "
                f"| {format_metric(metrics['calibrated_expected_calibration_error'])} |"
            )

    if study.failures:
        counts = Counter(failure.kind for failure in study.failures)
        lines += ["", "## Casos de falha auditáveis", "", "Contagem por tipo:", ""]
        lines += [f"- `{kind}`: {count}" for kind, count in sorted(counts.items())]
        lines += ["", "Exemplos:", ""]
        for failure in study.failures[:10]:
            lines.append(
                f"- `{failure.kind}` em `{failure.sample_id}` (`{failure.image_uri}`): "
                f"previsto {failure.predicted_value!r}, referência {list(failure.reference_labels)}, "
                f"bruto={failure.raw_confidence}, calibrado={failure.calibrated_confidence}, "
                f"suporte={failure.support_state} ({failure.support_reason}), "
                f"slots={list(failure.evidence_slots)}, "
                f"artifacts={list(failure.evidence_artifacts)}, "
                f"calibração={failure.calibration_version}/{failure.calibration_artifact}, "
                f"estratos={list(failure.strata)}"
            )

    lines += ["", "## Medidas independentes de anotação", ""]
    for name, value in sorted(study.operational.items()):
        lines.append(f"- `{name}`: {value}")
    return "\n".join(lines) + "\n"


# Serializa o estudo para JSON, para que o relatório de ablation (#200)
# possa recompor as tabelas sem reexecutar a avaliação.
def study_to_mapping(study: CalibrationStudy) -> dict[str, Any]:
    """Converte o estudo em um documento JSON serializável."""
    return {
        "configuration": study.configuration,
        "split": study.split.value,
        "reference_id": study.report.reference_id,
        "run_id": study.report.run_id,
        "code_revision": study.report.code_revision,
        "config_fingerprint": study.report.config_fingerprint,
        "sample_count": study.report.sample_count,
        "failed_sample_count": study.report.failed_sample_count,
        "ignored_annotation_count": study.report.ignored_annotation_count,
        "metrics": {
            name: asdict(metric) for name, metric in study.report.metrics.items()
        },
        "calibration": {
            mode: {name: asdict(metric) for name, metric in metrics.items()}
            for mode, metrics in study.report.calibration.items()
        },
        "reliability": study.report.reliability,
        "abstention": {
            name: asdict(metric) for name, metric in study.report.abstention.items()
        },
        "cost": {name: asdict(metric) for name, metric in study.report.cost.items()},
        "strata": {
            stratum: {name: asdict(metric) for name, metric in metrics.items()}
            for stratum, metrics in study.report.strata.items()
        },
        "by_claim_kind": {
            name: {key: asdict(metric) for key, metric in metrics.items()}
            for name, metrics in study.by_claim_kind.items()
        },
        "by_source": {
            name: {key: asdict(metric) for key, metric in metrics.items()}
            for name, metrics in study.by_source.items()
        },
        "by_stratum": {
            name: {key: asdict(metric) for key, metric in metrics.items()}
            for name, metrics in study.by_stratum.items()
        },
        "by_evidence_slot": {
            name: {key: asdict(metric) for key, metric in metrics.items()}
            for name, metrics in study.by_evidence_slot.items()
        },
        "failures": [asdict(failure) for failure in study.failures],
        "operational": study.operational,
    }


# Interface de linha de comando do experimento.
def main() -> None:
    """Executa o estudo de calibração sobre um conjunto de predições."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", type=str, default=Split.TEST.value)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--operational-only", action="store_true")
    arguments = parser.parse_args()

    predictions = load_predictions(arguments.predictions)
    destination = (
        arguments.output_dir or RESULTS_ROOT / "calibration" / predictions.run_id
    )
    destination.mkdir(parents=True, exist_ok=True)

    if arguments.operational_only:
        summary = operational_summary(predictions)
        path = destination / f"{predictions.configuration}-operational.json"
        path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"\nWrote {path}")
        return

    manifest = load_reference_manifest(arguments.manifest)
    study = study_calibration(manifest, predictions, Split(arguments.split))
    json_path = destination / f"{predictions.configuration}-{arguments.split}.json"
    json_path.write_text(
        json.dumps(study_to_mapping(study), indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    summary_path = json_path.with_suffix(".md")
    summary_path.write_text(render_study(study), encoding="utf-8")
    print(render_study(study))
    print(f"Wrote {json_path}\nWrote {summary_path}")


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
