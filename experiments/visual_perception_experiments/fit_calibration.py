"""Ajuste de uma tabela de calibração medida a partir do split de calibração.

Issues: #196 (a fronteira que consome o artifact), #199 (o experimento que
compara bruto e calibrado).

A calibração deste módulo é orientada a dados: ela converte um score bruto no
acerto empírico observado para aquela faixa de score. Este ajuste é o que
produz esse dado. Ele roda **apenas sobre o split de calibração**, que existe
exatamente para isto: usar o split de teste aqui contaminaria o resultado que
o teste deveria medir.

Bins com poucas observações são descartados em vez de ajustados. Um bin com
duas amostras produziria uma "acurácia empírica" de 0,0 ou 1,0 que não
descreve nada; descartá-lo faz o calibrador se abster para aquela faixa, que
é a resposta correta quando não há dado.

Sem anotação revisada não há acerto a observar, e a ferramenta recusa
produzir um artifact — um calibrador ajustado contra nada seria pior que
nenhum calibrador.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from contextual_mapping_datasets import (
    ReferenceManifest,
    Split,
    load_reference_manifest,
    validate_reference,
)
from contextual_mapping_datasets.annotation_manifest import AnnotationCertainty
from contextual_mapping_datasets.annotation_manifest_io import reference_manifest_to_mapping
from visual_perception_evaluation.masks import decode_mask
from visual_perception_evaluation.metrics import match_regions
from visual_perception_evaluation.prediction import (
    PredictionSet,
    load_predictions,
    prediction_set_to_mapping,
)
from visual_perception_evaluation.suite import EvaluationConfig

#: Versão de contract do artifact produzido. Precisa estar entre as versões
#: que ``domain/semantic_support.py`` sabe interpretar.
CALIBRATION_VERSION = "calibration/1"

#: Mínimo de observações para um bin entrar no artifact. Abaixo disso a
#: acurácia empírica é ruído, e o calibrador deve se abster naquela faixa.
MIN_BIN_SAMPLES = 5

# Gate conservador para ativação: um artifact pode ser produzido para
# diagnóstico com menos dados, mas não deve virar configuração de referência
# sem volume total e cobertura de faixas minimamente informativos.
MIN_ACTIVATION_OBSERVATIONS = 30
MIN_ACTIVATION_BINS = 3


# Registra um par (score bruto, acerto) observado no split de calibração.
# Existe para que o ajuste trabalhe sobre observações explícitas, e não
# sobre estruturas de predição.
@dataclass(frozen=True)
class Observation:
    """Um score bruto e se a decisão correspondente estava correta."""

    claim_kind: str
    raw_confidence: float
    correct: bool


# Percorre o split de calibração casando predição e anotação, e devolve as
# observações pontuadas. Chamada por fit_reliability_table.
def collect_observations(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    split: Split = Split.CALIBRATION,
    config: EvaluationConfig | None = None,
) -> tuple[Observation, ...]:
    """Coleta os pares (score bruto, acerto) do split de calibração.

    Argumentos:
        manifest: o conjunto de referência anotado.
        predictions: as predições da configuração a calibrar.
        split: o split usado para o ajuste; nunca deve ser o de teste.
        config: parâmetros de casamento (limiar de IoU).
    Retorna:
        as observações pontuáveis encontradas.
    """
    settings = config or EvaluationConfig()
    by_sample = predictions.by_sample_id()
    observations: list[Observation] = []

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
            iou_threshold=settings.iou_threshold,
        )
        for match in matching.matches:
            annotation = sample.regions[match.reference_index]
            if (
                annotation.certainty is AnnotationCertainty.UNKNOWN
                or not annotation.labels
            ):
                continue
            predicted = prediction.regions[match.prediction_index]
            scored = [
                value for value in predicted.labels if value.raw_confidence is not None
            ]
            if not scored:
                continue
            best = max(scored, key=lambda value: value.raw_confidence or 0.0)
            acceptable = {label.strip().lower() for label in annotation.labels}
            observations.append(
                Observation(
                    claim_kind="label",
                    raw_confidence=float(best.raw_confidence or 0.0),
                    correct=best.value.strip().lower() in acceptable,
                )
            )
    return tuple(observations)


# Ajusta os bins de confiabilidade a partir das observações. Existe separada
# da coleta para que os testes verifiquem a matemática do ajuste contra
# exemplos calculados à mão.
def fit_bins(
    observations: tuple[Observation, ...],
    *,
    bins: int = 5,
    min_samples: int = MIN_BIN_SAMPLES,
) -> dict[str, list[list[float]]]:
    """Ajusta a acurácia empírica por faixa de score, descartando bins sem dado.

    Argumentos:
        observations: os pares (score bruto, acerto) do split de calibração.
        bins: número de faixas de largura igual.
        min_samples: mínimo de observações para uma faixa ser publicada.
    Retorna:
        ``{claim_kind: [[lower, upper, calibrated], ...]}``, sem faixas vazias.
    """
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.claim_kind, []).append(observation)

    table: dict[str, list[list[float]]] = {}
    for claim_kind, items in sorted(grouped.items()):
        scores = np.asarray([item.raw_confidence for item in items], dtype=np.float64)
        correct = np.asarray(
            [1.0 if item.correct else 0.0 for item in items], dtype=np.float64
        )
        edges = np.linspace(0.0, 1.0, bins + 1)
        entries: list[list[float]] = []
        for index in range(bins):
            lower, upper = float(edges[index]), float(edges[index + 1])
            in_bin = (
                (scores > lower) & (scores <= upper)
                if index
                else (scores >= lower) & (scores <= upper)
            )
            if int(in_bin.sum()) < min_samples:
                continue
            entries.append([lower, upper, float(correct[in_bin].mean())])
        if entries:
            table[claim_kind] = entries
    return table


# Monta o documento do artifact de calibração, com a proveniência que o
# torna rastreável. Existe separada da escrita para que os testes validem o
# documento sem tocar em disco.
def build_artifact(
    table: dict[str, list[list[float]]],
    counts: dict[str, int],
    *,
    domain: str,
    source: str,
    reference_id: str,
    run_id: str,
    reference_digest: str,
    predictions_digest: str,
    config_fingerprint: str,
    code_revision: str,
    min_activation_observations: int = MIN_ACTIVATION_OBSERVATIONS,
    min_activation_bins: int = MIN_ACTIVATION_BINS,
) -> dict[str, Any]:
    """Monta o documento JSON do artifact de calibração."""
    total_observations = sum(counts.values())
    populated_bins = sum(len(entries) for entries in table.values())
    activation_enabled = (
        total_observations >= min_activation_observations
        and populated_bins >= min_activation_bins
    )
    document: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "domain": domain,
        "fitted_from": {
            "reference_id": reference_id,
            "split": Split.CALIBRATION.value,
            "prediction_run_id": run_id,
            "reference_digest": reference_digest,
            "predictions_digest": predictions_digest,
            "config_fingerprint": config_fingerprint,
            "code_revision": code_revision,
        },
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            source: {
                claim_kind: {"bins": entries, "samples": counts.get(claim_kind, 0)}
                for claim_kind, entries in table.items()
            }
        },
        "activation": {
            "enabled": activation_enabled,
            "observations": total_observations,
            "populated_bins": populated_bins,
            "minimum_observations": min_activation_observations,
            "minimum_populated_bins": min_activation_bins,
            "reason": (
                None
                if activation_enabled
                else "cobertura medida insuficiente; manter calibração desabilitada"
            ),
        },
    }
    document["payload_digest"] = _payload_digest(document)
    return document


# Calcula um digest canônico de um documento, excluindo o próprio campo de
# digest. Usado para detectar edição/truncamento do artifact após o ajuste.
def _payload_digest(document: dict[str, Any]) -> str:
    """Retorna o SHA-256 do payload JSON canônico sem ``payload_digest``."""
    payload = {key: value for key, value in document.items() if key != "payload_digest"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Garante que o arquivo entregue ao ajuste contém exclusivamente amostras do
# split calibration. Mesmo que a coleta filtre, recusar test no input evita
# provenance ambígua e vazamento acidental em futuras transformações.
def validate_calibration_predictions(
    manifest: ReferenceManifest, predictions: PredictionSet
) -> None:
    """Rejeita predições ausentes ou pertencentes a development/test."""
    calibration_ids = {sample.sample_id for sample in manifest.samples_in(Split.CALIBRATION)}
    predicted_ids = {sample.sample_id for sample in predictions.samples}
    foreign = sorted(predicted_ids - calibration_ids)
    if foreign:
        raise ValueError(
            f"Calibration predictions contain samples outside the calibration split: {foreign}."
        )
    missing = sorted(calibration_ids - predicted_ids)
    if missing:
        raise ValueError(f"Calibration predictions are missing samples: {missing}.")


# Ponto de entrada do ajuste: coleta, ajusta e grava o artifact. Recusa
# produzir qualquer coisa quando não há observação suficiente.
def fit_reliability_table(
    manifest: ReferenceManifest,
    predictions: PredictionSet,
    destination: Path,
    *,
    source: str,
    domain: str,
    bins: int = 5,
    min_samples: int = MIN_BIN_SAMPLES,
) -> dict[str, Any]:
    """Ajusta e grava a tabela de calibração do split de calibração.

    Argumentos:
        manifest: o conjunto de referência anotado.
        predictions: as predições da configuração a calibrar.
        destination: onde gravar o artifact.
        source: o produtor cujos scores estão sendo calibrados.
        domain: o domínio em que a tabela vale.
        bins: número de faixas de largura igual.
        min_samples: mínimo de observações por faixa.
    Retorna:
        o documento gravado.
    Levanta:
        SystemExit: se não houver observação pontuável suficiente.
    """
    validate_reference(manifest, require_reviewed=True)
    validate_calibration_predictions(manifest, predictions)
    observations = collect_observations(manifest, predictions)
    if not observations:
        raise SystemExit(
            "No scored observations were found in the calibration split. This happens when the "
            "reference set has no reviewed annotations yet: fitting a calibrator against nothing "
            "would fabricate the confidence it claims to calibrate."
        )
    table = fit_bins(observations, bins=bins, min_samples=min_samples)
    if not table:
        raise SystemExit(
            f"No confidence bin reached {min_samples} observations ({len(observations)} scored "
            "decisions in total). Collect more calibration samples before fitting."
        )
    counts: dict[str, int] = {}
    for observation in observations:
        counts[observation.claim_kind] = counts.get(observation.claim_kind, 0) + 1

    document = build_artifact(
        table,
        counts,
        domain=domain,
        source=source,
        reference_id=manifest.reference_id,
        run_id=predictions.run_id,
        reference_digest=_payload_digest(reference_manifest_to_mapping(manifest)),
        predictions_digest=_payload_digest(prediction_set_to_mapping(predictions)),
        config_fingerprint=predictions.config_fingerprint,
        code_revision=predictions.code_revision,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


# Interface de linha de comando do ajuste.
def main() -> None:
    """Ajusta a tabela de calibração a partir de um conjunto de predições."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=MIN_BIN_SAMPLES)
    arguments = parser.parse_args()

    document = fit_reliability_table(
        load_reference_manifest(arguments.manifest),
        load_predictions(arguments.predictions),
        arguments.out,
        source=arguments.source,
        domain=arguments.domain,
        bins=arguments.bins,
        min_samples=arguments.min_samples,
    )
    entries = document["sources"][arguments.source]
    print(f"Wrote {arguments.out}")
    for claim_kind, entry in entries.items():
        print(
            f"  {claim_kind}: {len(entry['bins'])} bins from {entry['samples']} observations"
        )


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
