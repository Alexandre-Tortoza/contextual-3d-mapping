"""Contract versionado de predição consumido pela avaliação.

Issue: #198.

As métricas consomem *apenas* dois contracts versionados: o manifest de
referência (`contextual_mapping_datasets.annotation_manifest`) e este. Nada
aqui importa ``visual_perception``: a avaliação precisa continuar válida
quando a representação interna do módulo mudar, e precisa poder pontuar a
saída de qualquer produtor que emita este formato.

O contract preserva as três distinções que a fase de calibração introduziu
(#195/#196) e sem as quais as métricas de confiabilidade seriam impossíveis:
``raw_confidence`` (o que o produtor informou), ``calibrated_confidence`` (o
que a calibração verificou) e ``support_state`` (por que não há score, quando
não há). Um valor ausente permanece ausente: nada aqui converte ``None`` em
número.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .masks import decode_mask

#: Versão do contract de predição. Incrementar junto com uma migração
#: explícita nas ferramentas que o produzem e o consomem.
PREDICTION_SCHEMA_VERSION = "visual-prediction/1"


# Espelha os estados de suporte semântico do produtor (#195). Existe aqui, e
# não importado do módulo, porque este contract precisa ser legível sem a
# stack do produtor instalada.
class SupportState(StrEnum):
    """O estado de calibração informado pelo produtor para um valor previsto."""

    UNKNOWN = "unknown"
    RAW = "raw"
    CALIBRATED = "calibrated"
    ABSTAINED = "abstained"
    FAILED = "failed"


# Representa um valor semântico previsto com seus dois scores possíveis.
# Existe para que bruto e calibrado cheguem à métrica separados: agregá-los
# em um único campo tornaria a comparação raw-versus-calibrated (#199)
# impossível de calcular.
@dataclass(frozen=True)
class ScoredValue:
    """Um valor semântico previsto, com score bruto e calibrado separados."""

    value: str
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    support_state: SupportState = SupportState.UNKNOWN
    reason: str | None = None

    # Valida o valor e a faixa dos dois scores, para que uma métrica nunca
    # receba uma probabilidade fora de [0, 1].
    def __post_init__(self) -> None:
        """Rejeita valores vazios e scores fora de ``[0, 1]``."""
        if not self.value:
            raise ValueError("ScoredValue.value must not be empty.")
        for name in ("raw_confidence", "calibrated_confidence"):
            score = getattr(self, name)
            if score is None:
                continue
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise ValueError(f"ScoredValue.{name} must be a real number or null.")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"ScoredValue.{name} must be in [0, 1], got {score}.")

    # Retorna o score do modo pedido, mantendo a ausência como ausência.
    # Existe para que os cálculos de calibração escolham o modo sem espalhar
    # condicionais por todas as métricas.
    def score(self, *, calibrated: bool) -> float | None:
        """Retorna o score calibrado ou bruto, ou ``None`` quando não informado."""
        return self.calibrated_confidence if calibrated else self.raw_confidence


# Representa uma região prevista com sua máscara e seus valores semânticos.
# Existe como a unidade que as métricas de região e de semântica comparam
# com uma :class:`RegionAnnotation` da referência.
@dataclass(frozen=True)
class PredictedRegion:
    """Uma região prevista: geometria mais valores semânticos ordenados."""

    region_id: str
    width: int
    height: int
    runs: tuple[int, ...]
    labels: tuple[ScoredValue, ...] = field(default_factory=tuple)
    attributes: tuple[ScoredValue, ...] = field(default_factory=tuple)
    conditions: tuple[ScoredValue, ...] = field(default_factory=tuple)
    materials: tuple[ScoredValue, ...] = field(default_factory=tuple)
    hazards: tuple[ScoredValue, ...] = field(default_factory=tuple)
    evidence_slots: tuple[str, ...] = field(default_factory=tuple)

    # Valida identidade e resolução; a validade do RLE é imposta na
    # decodificação, para não pagar o custo duas vezes.
    def __post_init__(self) -> None:
        """Rejeita regiões sem identidade ou com resolução inválida."""
        if not self.region_id:
            raise ValueError("PredictedRegion.region_id must not be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"PredictedRegion({self.region_id!r}) has a non-positive resolution.")

    # Decodifica a máscara sob demanda. Existe como método (e não campo) para
    # que carregar um conjunto grande de predições não materialize todas as
    # máscaras de uma vez.
    def mask(self) -> np.ndarray:
        """Decodifica a máscara desta região em um array booleano."""
        return decode_mask(self.width, self.height, self.runs)


# Representa uma relação prevista entre duas regiões previstas.
@dataclass(frozen=True)
class PredictedRelation:
    """Uma relação prevista entre duas regiões da mesma amostra."""

    relation_id: str
    subject_region_id: str
    predicate: str
    object_region_id: str
    confidence: float | None = None

    # Valida identidade e a faixa da confiança informada.
    def __post_init__(self) -> None:
        """Rejeita relações sem identidade ou com confiança fora de ``[0, 1]``."""
        for name in ("relation_id", "subject_region_id", "predicate", "object_region_id"):
            if not getattr(self, name):
                raise ValueError(f"PredictedRelation.{name} must not be empty.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("PredictedRelation.confidence must be in [0, 1] or null.")


# Agrupa a predição completa de uma amostra, incluindo custo medido e o
# estado de falha. Existe para que a taxa de falha entre na avaliação como
# dado, e não como amostra silenciosamente ausente.
@dataclass(frozen=True)
class SamplePrediction:
    """A predição de uma amostra, com custo medido e estado de falha explícito."""

    sample_id: str
    width: int
    height: int
    regions: tuple[PredictedRegion, ...] = field(default_factory=tuple)
    relations: tuple[PredictedRelation, ...] = field(default_factory=tuple)
    scene_claims: tuple[ScoredValue, ...] = field(default_factory=tuple)
    failed: bool = False
    failure_reason: str | None = None
    latency_s: float | None = None
    peak_vram_bytes: int | None = None
    model_calls: int | None = None

    # Impõe que uma amostra que falhou explique a falha e não traga regiões,
    # e que a resolução das regiões case com a da amostra.
    def __post_init__(self) -> None:
        """Valida coerência entre falha, regiões e resolução declarada."""
        if not self.sample_id:
            raise ValueError("SamplePrediction.sample_id must not be empty.")
        if self.failed and not self.failure_reason:
            raise ValueError(f"SamplePrediction({self.sample_id!r}) failed without a reason.")
        if self.failed and self.regions:
            raise ValueError(
                f"SamplePrediction({self.sample_id!r}) is marked failed but carries regions."
            )
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError(f"SamplePrediction({self.sample_id!r}) has duplicate region ids.")
        for region in self.regions:
            if (region.width, region.height) != (self.width, self.height):
                raise ValueError(
                    f"PredictedRegion({region.region_id!r}) resolution does not match "
                    f"sample {self.sample_id!r}."
                )
        known = set(region_ids)
        for relation in self.relations:
            for region_id in (relation.subject_region_id, relation.object_region_id):
                if region_id not in known:
                    raise ValueError(
                        f"PredictedRelation({relation.relation_id!r}) references unknown region "
                        f"{region_id!r} in sample {self.sample_id!r}."
                    )


# Agrupa as predições de uma configuração inteira, junto da proveniência que
# torna o resultado reproduzível. Existe porque um número de avaliação sem a
# revisão de código e o fingerprint de config que o geraram não é evidência.
@dataclass(frozen=True)
class PredictionSet:
    """As predições de uma configuração, com a proveniência que as reproduz."""

    run_id: str
    configuration: str
    config_fingerprint: str
    code_revision: str
    samples: tuple[SamplePrediction, ...]
    schema_version: str = PREDICTION_SCHEMA_VERSION
    hardware: str | None = None
    created_at: str | None = None

    # Valida a versão do contract, a proveniência obrigatória e a unicidade
    # das amostras.
    def __post_init__(self) -> None:
        """Rejeita versões desconhecidas, proveniência faltante e amostras duplicadas."""
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported prediction schema_version {self.schema_version!r}; "
                f"expected {PREDICTION_SCHEMA_VERSION!r}."
            )
        for name in ("run_id", "configuration", "config_fingerprint", "code_revision"):
            if not getattr(self, name):
                raise ValueError(f"PredictionSet.{name} must not be empty.")
        if not self.samples:
            raise ValueError("a prediction set must contain at least one sample.")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("a prediction set must not contain duplicate sample ids.")

    # Indexa as amostras por id. Usada pela avaliação, que percorre a
    # referência e busca a predição correspondente.
    def by_sample_id(self) -> dict[str, SamplePrediction]:
        """Retorna as predições indexadas pelo id da amostra."""
        return {sample.sample_id: sample for sample in self.samples}


# Carrega e valida um conjunto de predições gravado em disco. Existe para que
# a avaliação nunca receba um dict solto, e para que uma incompatibilidade de
# versão falhe na leitura em vez de virar uma métrica errada.
def load_predictions(path: Path) -> PredictionSet:
    """Carrega um :class:`PredictionSet` a partir de um arquivo JSON.

    Argumentos:
        path: arquivo JSON produzido por um experimento.
    Retorna:
        o conjunto de predições validado.
    Levanta:
        ValueError: se o JSON ou o contract não forem válidos.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON prediction set {path}: {error.msg}.") from error
    return prediction_set_from_mapping(document, source=str(path))


# Grava um conjunto de predições de forma determinística, para que dois runs
# idênticos produzam arquivos idênticos e a reprodutibilidade seja
# verificável por digest.
def save_predictions(predictions: PredictionSet, path: Path) -> None:
    """Grava ``predictions`` como JSON determinístico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prediction_set_to_mapping(predictions), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# Converte o documento desserializado no conjunto tipado. É pública para que
# quem produz predições valide o payload antes de gravá-lo.
def prediction_set_from_mapping(document: object, *, source: str = "predictions") -> PredictionSet:
    """Converte um mapeamento JSON em um :class:`PredictionSet` validado."""
    root = _mapping(document, source)
    return PredictionSet(
        run_id=_string(root, "run_id", source),
        configuration=_string(root, "configuration", source),
        config_fingerprint=_string(root, "config_fingerprint", source),
        code_revision=_string(root, "code_revision", source),
        samples=tuple(
            _sample(item, f"{source}.samples[{index}]")
            for index, item in enumerate(_sequence(root, "samples", source))
        ),
        schema_version=_string(root, "schema_version", source, PREDICTION_SCHEMA_VERSION),
        hardware=root.get("hardware"),
        created_at=root.get("created_at"),
    )


# Converte o conjunto tipado de volta em documento JSON — inverso exato de
# prediction_set_from_mapping.
def prediction_set_to_mapping(predictions: PredictionSet) -> dict[str, Any]:
    """Converte um :class:`PredictionSet` em um documento JSON serializável."""
    return {
        "schema_version": predictions.schema_version,
        "run_id": predictions.run_id,
        "configuration": predictions.configuration,
        "config_fingerprint": predictions.config_fingerprint,
        "code_revision": predictions.code_revision,
        "hardware": predictions.hardware,
        "created_at": predictions.created_at,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "width": sample.width,
                "height": sample.height,
                "failed": sample.failed,
                "failure_reason": sample.failure_reason,
                "latency_s": sample.latency_s,
                "peak_vram_bytes": sample.peak_vram_bytes,
                "model_calls": sample.model_calls,
                "scene_claims": [_scored_to_mapping(value) for value in sample.scene_claims],
                "regions": [
                    {
                        "region_id": region.region_id,
                        "mask": {
                            "width": region.width,
                            "height": region.height,
                            "runs": list(region.runs),
                        },
                        "labels": [_scored_to_mapping(value) for value in region.labels],
                        "attributes": [_scored_to_mapping(value) for value in region.attributes],
                        "conditions": [_scored_to_mapping(value) for value in region.conditions],
                        "materials": [_scored_to_mapping(value) for value in region.materials],
                        "hazards": [_scored_to_mapping(value) for value in region.hazards],
                        "evidence_slots": list(region.evidence_slots),
                    }
                    for region in sample.regions
                ],
                "relations": [
                    {
                        "relation_id": relation.relation_id,
                        "subject_region_id": relation.subject_region_id,
                        "predicate": relation.predicate,
                        "object_region_id": relation.object_region_id,
                        "confidence": relation.confidence,
                    }
                    for relation in sample.relations
                ],
            }
            for sample in predictions.samples
        ],
    }


# Constrói a predição de uma amostra a partir do documento.
def _sample(value: object, source: str) -> SamplePrediction:
    """Converte um item ``samples[i]`` em :class:`SamplePrediction`."""
    document = _mapping(value, source)
    return SamplePrediction(
        sample_id=_string(document, "sample_id", source),
        width=_positive_int(document, "width", source),
        height=_positive_int(document, "height", source),
        regions=tuple(
            _region(item, f"{source}.regions[{index}]")
            for index, item in enumerate(_sequence(document, "regions", source, ()))
        ),
        relations=tuple(
            _relation(item, f"{source}.relations[{index}]")
            for index, item in enumerate(_sequence(document, "relations", source, ()))
        ),
        scene_claims=_scored_sequence(document, "scene_claims", source),
        failed=bool(document.get("failed", False)),
        failure_reason=document.get("failure_reason"),
        latency_s=document.get("latency_s"),
        peak_vram_bytes=document.get("peak_vram_bytes"),
        model_calls=document.get("model_calls"),
    )


# Constrói uma região prevista, incluindo a sua máscara RLE.
def _region(value: object, source: str) -> PredictedRegion:
    """Converte um item ``regions[i]`` em :class:`PredictedRegion`."""
    document = _mapping(value, source)
    mask = _mapping(document.get("mask"), f"{source}.mask")
    runs = _sequence(mask, "runs", f"{source}.mask")
    return PredictedRegion(
        region_id=_string(document, "region_id", source),
        width=_positive_int(mask, "width", f"{source}.mask"),
        height=_positive_int(mask, "height", f"{source}.mask"),
        runs=tuple(int(run) for run in runs),
        labels=_scored_sequence(document, "labels", source),
        attributes=_scored_sequence(document, "attributes", source),
        conditions=_scored_sequence(document, "conditions", source),
        materials=_scored_sequence(document, "materials", source),
        hazards=_scored_sequence(document, "hazards", source),
        evidence_slots=tuple(str(item) for item in _sequence(document, "evidence_slots", source, ())),
    )


# Constrói uma relação prevista.
def _relation(value: object, source: str) -> PredictedRelation:
    """Converte um item ``relations[i]`` em :class:`PredictedRelation`."""
    document = _mapping(value, source)
    return PredictedRelation(
        relation_id=_string(document, "relation_id", source),
        subject_region_id=_string(document, "subject_region_id", source),
        predicate=_string(document, "predicate", source),
        object_region_id=_string(document, "object_region_id", source),
        confidence=document.get("confidence"),
    )


# Converte uma lista de valores pontuados, compartilhada por labels,
# atributos, condições, materiais, hazards e claims de cena.
def _scored_sequence(document: Mapping[str, Any], field_name: str, source: str) -> tuple[ScoredValue, ...]:
    """Converte uma lista JSON de valores pontuados em tupla de :class:`ScoredValue`."""
    items = _sequence(document, field_name, source, ())
    return tuple(
        _scored(item, f"{source}.{field_name}[{index}]") for index, item in enumerate(items)
    )


# Constrói um valor pontuado, preservando a ausência de score.
def _scored(value: object, source: str) -> ScoredValue:
    """Converte um valor pontuado do documento em :class:`ScoredValue`."""
    document = _mapping(value, source)
    return ScoredValue(
        value=_string(document, "value", source),
        raw_confidence=document.get("raw_confidence"),
        calibrated_confidence=document.get("calibrated_confidence"),
        support_state=SupportState(document.get("support_state", SupportState.UNKNOWN.value)),
        reason=document.get("reason"),
    )


# Converte um valor pontuado de volta em documento JSON.
def _scored_to_mapping(value: ScoredValue) -> dict[str, Any]:
    """Converte um :class:`ScoredValue` em um dict serializável."""
    return {
        "value": value.value,
        "raw_confidence": value.raw_confidence,
        "calibrated_confidence": value.calibrated_confidence,
        "support_state": value.support_state.value,
        "reason": value.reason,
    }


# Obtém um objeto JSON, recusando listas e escalares.
def _mapping(value: object, source: str) -> Mapping[str, Any]:
    """Retorna ``value`` como objeto JSON, ou falha nomeando o campo."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be a JSON object.")
    return value


# Obtém uma lista JSON, recusando strings.
def _sequence(
    document: Mapping[str, Any], field_name: str, source: str, default: Sequence[object] | None = None
) -> Sequence[object]:
    """Retorna ``document[field_name]`` como lista JSON."""
    value = document.get(field_name, default)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{source}.{field_name} must be a JSON array.")
    return value


# Obtém uma string obrigatória, sem coerção de outros tipos.
def _string(
    document: Mapping[str, Any], field_name: str, source: str, default: str | None = None
) -> str:
    """Retorna um campo string não vazio."""
    value = document.get(field_name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}.{field_name} must be a non-empty string.")
    return value


# Obtém um inteiro estritamente positivo, recusando booleanos e floats.
def _positive_int(document: Mapping[str, Any], field_name: str, source: str) -> int:
    """Retorna um campo inteiro estritamente positivo."""
    value = document.get(field_name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{source}.{field_name} must be a positive integer.")
    return value
