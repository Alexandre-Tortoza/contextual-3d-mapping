"""Primitivas de métrica da avaliação de percepção visual.

Issue: #198.

Cada função aqui é pura, determinística e verificável à mão: elas recebem
listas e arrays, e não objetos de contract. É o que permite que os testes
comprovem os valores contra exemplos calculados no papel, em vez de contra a
própria implementação.

Duas convenções valem para todas as métricas deste módulo:

- **suporte explícito**: toda métrica devolve um :class:`MetricValue`, que
  carrega quantas observações a sustentam. Sem observações, ``value`` é
  ``None`` — "não medido" nunca vira zero, e um report não pode apresentar
  uma métrica sem suporte como evidência (ver :func:`is_comparable`);
- **ausência preservada**: um score ausente (claim não pontuada, abstenção)
  é excluído do cálculo de confiabilidade e contabilizado no relatório de
  cobertura, nunca substituído por 0.0 ou 0.5.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .masks import boundary_f1, intersection_over_union

#: IoU mínima para que uma região prevista seja considerada a mesma região
#: anotada. 0.5 é a convenção de detecção usada como padrão pelo harness de
#: benchmark do módulo (#175); os reports declaram o valor usado.
DEFAULT_MATCH_IOU = 0.5


# Representa o valor de uma métrica junto do suporte que a sustenta. Existe
# porque a #198 proíbe apresentar uma métrica sem suporte como evidência: o
# número e a quantidade de observações precisam viajar juntos.
@dataclass(frozen=True)
class MetricValue:
    """O valor de uma métrica e quantas observações o sustentam."""

    name: str
    value: float | None
    support: int
    detail: str | None = None

    # Impõe que "não medido" e "medido como zero" sejam estados distintos:
    # sem suporte não pode haver valor, e com suporte o valor é obrigatório.
    def __post_init__(self) -> None:
        """Rejeita a confusão entre métrica não medida e métrica igual a zero."""
        if not self.name:
            raise ValueError("MetricValue.name must not be empty.")
        if self.support < 0:
            raise ValueError(f"MetricValue({self.name!r}).support must not be negative.")
        if self.support == 0 and self.value is not None:
            raise ValueError(
                f"MetricValue({self.name!r}) has no support and therefore cannot carry a value."
            )
        if self.support > 0 and self.value is None:
            raise ValueError(f"MetricValue({self.name!r}) has support but no value.")

    # Responde se esta métrica pode sustentar uma comparação entre
    # configurações. Usada pelo report de ablation (#200) para recusar
    # ranquear configurações por uma métrica que ninguém mediu.
    @property
    def measured(self) -> bool:
        """Indica se a métrica tem suporte suficiente para ser lida como resultado."""
        return self.support > 0 and self.value is not None


# Agrupa um par referência/predição casado e a qualidade do casamento.
# Existe para que as métricas de semântica saibam exatamente qual região
# prevista responde por qual região anotada.
@dataclass(frozen=True)
class RegionMatch:
    """Um par referência/predição casado, com a IoU que os uniu."""

    reference_index: int
    prediction_index: int
    iou: float


# Agrupa o resultado completo da associação entre referência e predição,
# incluindo o que ficou de fora de cada lado. Existe porque cobertura e
# alucinação se calculam justamente a partir do que *não* casou.
@dataclass(frozen=True)
class RegionMatching:
    """O resultado da associação entre regiões anotadas e previstas."""

    matches: tuple[RegionMatch, ...]
    unmatched_reference: tuple[int, ...]
    unmatched_prediction: tuple[int, ...]
    ignored_prediction: tuple[int, ...] = field(default_factory=tuple)


# Associa cada região prevista à melhor região anotada ainda livre, de forma
# gulosa e determinística. Regiões anotadas como ignoradas absorvem as
# predições que as cobrem, sem contar como acerto nem como erro. Chamada por
# toda métrica que precisa de correspondência região a região.
def match_regions(
    reference_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    *,
    ignored: Sequence[bool] | None = None,
    iou_threshold: float = DEFAULT_MATCH_IOU,
) -> RegionMatching:
    """Casa predições com anotações por IoU, isolando as anotações ignoradas.

    O casamento é guloso sobre os pares ordenados por IoU decrescente, com
    desempate determinístico pelos índices, para que a mesma entrada produza
    sempre a mesma associação.

    Argumentos:
        reference_masks: máscaras anotadas, na ordem do manifest.
        prediction_masks: máscaras previstas, na ordem da predição.
        ignored: para cada anotação, se ela é uma região ignorada.
        iou_threshold: IoU mínima para aceitar um par.
    Retorna:
        o :class:`RegionMatching` completo.
    """
    flags = tuple(ignored or (False,) * len(reference_masks))
    if len(flags) != len(reference_masks):
        raise ValueError("The ignored flags must have one entry per reference mask.")

    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference_mask in enumerate(reference_masks):
        for prediction_index, prediction_mask in enumerate(prediction_masks):
            iou = intersection_over_union(reference_mask, prediction_mask)
            if iou >= iou_threshold:
                candidates.append((iou, reference_index, prediction_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matches: list[RegionMatch] = []
    ignored_predictions: list[int] = []
    used_reference: set[int] = set()
    used_prediction: set[int] = set()
    for iou, reference_index, prediction_index in candidates:
        if reference_index in used_reference or prediction_index in used_prediction:
            continue
        used_reference.add(reference_index)
        used_prediction.add(prediction_index)
        if flags[reference_index]:
            ignored_predictions.append(prediction_index)
            continue
        matches.append(RegionMatch(reference_index, prediction_index, iou))

    unmatched_reference = tuple(
        index
        for index in range(len(reference_masks))
        if index not in used_reference and not flags[index]
    )
    unmatched_prediction = tuple(
        index for index in range(len(prediction_masks)) if index not in used_prediction
    )
    return RegionMatching(
        matches=tuple(matches),
        unmatched_reference=unmatched_reference,
        unmatched_prediction=unmatched_prediction,
        ignored_prediction=tuple(sorted(ignored_predictions)),
    )


# Calcula precisão, recall, F1 e IoU média a partir de uma associação já
# feita. Existe separada de match_regions para que a mesma associação sirva
# a várias métricas sem ser recalculada.
def detection_metrics(matching: RegionMatching) -> dict[str, MetricValue]:
    """Retorna precisão, recall, F1 e IoU média das regiões casadas.

    Argumentos:
        matching: o resultado de :func:`match_regions`.
    Retorna:
        um dict de :class:`MetricValue` por nome de métrica.
    """
    true_positive = len(matching.matches)
    predicted = true_positive + len(matching.unmatched_prediction)
    annotated = true_positive + len(matching.unmatched_reference)
    precision = _ratio("region_precision", true_positive, predicted)
    recall = _ratio("region_recall", true_positive, annotated)
    ious = [match.iou for match in matching.matches]
    return {
        "region_precision": precision,
        "region_recall": recall,
        "region_f1": _harmonic("region_f1", precision, recall, support=predicted + annotated),
        "matched_mask_iou": MetricValue(
            "matched_mask_iou",
            float(np.mean(ious)) if ious else None,
            len(ious),
        ),
    }


# Calcula o boundary F1 médio das regiões casadas. Existe porque a IoU de
# área quase não penaliza uma borda ruim em regiões grandes, e a borda é
# justamente o que a evidência densa de alta resolução (#192) deve melhorar.
def boundary_metrics(
    reference_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    matching: RegionMatching,
    *,
    tolerance: int = 2,
) -> MetricValue:
    """Retorna o boundary F1 médio das regiões casadas, com a tolerância usada."""
    scores = [
        boundary_f1(
            reference_masks[match.reference_index],
            prediction_masks[match.prediction_index],
            tolerance=tolerance,
        )
        for match in matching.matches
    ]
    return MetricValue(
        "boundary_f1",
        float(np.mean(scores)) if scores else None,
        len(scores),
        detail=f"tolerance={tolerance}px",
    )


# Pontua a semântica de vocabulário aberto de uma região casada: acerto de
# topo-1 e reciprocal rank. Existe porque a referência admite vários labels
# aceitáveis por região (ambiguidade explícita, #197), então acerto é
# pertencer ao conjunto, e não coincidir com uma classe única.
def open_vocabulary_scores(
    acceptable: Sequence[str], ranked_predictions: Sequence[str]
) -> tuple[float, float]:
    """Retorna ``(top1, reciprocal_rank)`` de uma região sob vocabulário aberto.

    Argumentos:
        acceptable: labels que a referência aceita para a região.
        ranked_predictions: labels previstos, em ordem decrescente de confiança.
    Retorna:
        ``(1.0/0.0, 1/rank)``; ``(0.0, 0.0)`` quando nenhum previsto é aceitável.
    """
    allowed = {value.strip().lower() for value in acceptable}
    for rank, value in enumerate(ranked_predictions, start=1):
        if value.strip().lower() in allowed:
            return (1.0 if rank == 1 else 0.0), 1.0 / rank
    return 0.0, 0.0


# Calcula precisão, recall e F1 entre dois conjuntos de valores textuais.
# Existe para atributos, condições, materiais e hazards, que compartilham
# exatamente a mesma semântica de conjunto.
def set_metrics(
    name: str, reference_values: Sequence[str], predicted_values: Sequence[str]
) -> dict[str, MetricValue]:
    """Retorna precisão, recall e F1 entre dois conjuntos de valores textuais."""
    reference = {value.strip().lower() for value in reference_values}
    predicted = {value.strip().lower() for value in predicted_values}
    if not reference:
        # Sem anotação para esta tarefa não há o que medir: a métrica fica
        # sem suporte em vez de virar zero (que leria como "errou tudo").
        return {
            f"{name}_precision": MetricValue(f"{name}_precision", None, 0),
            f"{name}_recall": MetricValue(f"{name}_recall", None, 0),
            f"{name}_f1": MetricValue(f"{name}_f1", None, 0),
        }
    hits = len(reference & predicted)
    precision = _ratio(f"{name}_precision", hits, len(predicted))
    recall = _ratio(f"{name}_recall", hits, len(reference))
    return {
        f"{name}_precision": precision,
        f"{name}_recall": recall,
        f"{name}_f1": _harmonic(f"{name}_f1", precision, recall, support=len(reference)),
    }


# Calcula o erro de calibração esperado (ECE) por binning de largura igual.
# Existe como a métrica central de confiabilidade: ela responde se um score
# de 0,8 acerta 80% das vezes, que é o que a fase de calibração (#196)
# pretende melhorar.
def expected_calibration_error(
    scores: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> MetricValue:
    """Retorna o ECE de largura igual entre ``scores`` e ``correct``.

    Argumentos:
        scores: probabilidades previstas, todas em ``[0, 1]``.
        correct: se cada predição correspondente estava certa.
        bins: número de faixas de largura igual.
    Retorna:
        o :class:`MetricValue` do ECE, sem suporte quando não há score.
    """
    values, hits = _paired(scores, correct)
    if values.size == 0:
        return MetricValue("expected_calibration_error", None, 0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (values > lower) & (values <= upper) if index else (values >= lower) & (values <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        total += (count / values.size) * abs(float(hits[in_bin].mean()) - float(values[in_bin].mean()))
    return MetricValue("expected_calibration_error", total, int(values.size), detail=f"bins={bins}")


# Calcula o Brier score (erro quadrático médio da probabilidade). Existe ao
# lado do ECE porque ele penaliza simultaneamente calibração e discriminação,
# enquanto o ECE ignora um modelo que sempre prevê a taxa base.
def brier_score(scores: Sequence[float], correct: Sequence[bool]) -> MetricValue:
    """Retorna o Brier score entre ``scores`` e ``correct``."""
    values, hits = _paired(scores, correct)
    if values.size == 0:
        return MetricValue("brier_score", None, 0)
    return MetricValue("brier_score", float(np.mean((values - hits) ** 2)), int(values.size))


# Constrói a tabela de confiabilidade usada nos diagramas do report. Existe
# para que o report possa mostrar *onde* a calibração erra, e não apenas
# quanto ela erra no agregado.
def reliability_table(
    scores: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> tuple[dict[str, float | int], ...]:
    """Retorna, por faixa de confiança, a confiança média, a acurácia e a contagem."""
    values, hits = _paired(scores, correct)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table: list[dict[str, float | int]] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if values.size == 0:
            table.append({"lower": lower, "upper": upper, "count": 0})
            continue
        in_bin = (values > lower) & (values <= upper) if index else (values >= lower) & (values <= upper)
        count = int(in_bin.sum())
        entry: dict[str, float | int] = {"lower": lower, "upper": upper, "count": count}
        if count:
            entry["mean_confidence"] = float(values[in_bin].mean())
            entry["accuracy"] = float(hits[in_bin].mean())
        table.append(entry)
    return tuple(table)


# Resume cobertura, abstenção e risco seletivo de um conjunto de decisões.
# Existe porque a #198 exige que claims sem score permaneçam visíveis: um
# sistema que se abstém de tudo teria risco zero e cobertura zero, e os dois
# números precisam ser lidos juntos.
def abstention_report(
    correct: Sequence[bool], scored: Sequence[bool], *, abstained: Sequence[bool] | None = None
) -> dict[str, MetricValue]:
    """Retorna cobertura, taxa de abstenção, taxa sem score e risco seletivo.

    Argumentos:
        correct: se cada decisão estava correta.
        scored: se cada decisão recebeu um score utilizável.
        abstained: se cada decisão foi explicitamente uma abstenção.
    Retorna:
        um dict de :class:`MetricValue` por nome de métrica.
    """
    if len(correct) != len(scored):
        raise ValueError("The correct and scored sequences must have the same length.")
    total = len(correct)
    flags = tuple(abstained or (False,) * total)
    if len(flags) != total:
        raise ValueError("The abstained flags must have one entry per decision.")
    covered = [index for index, is_scored in enumerate(scored) if is_scored]
    errors = sum(1 for index in covered if not correct[index])
    return {
        "coverage": _ratio("coverage", len(covered), total),
        "abstention_rate": _ratio("abstention_rate", sum(1 for flag in flags if flag), total),
        "unscored_rate": _ratio("unscored_rate", total - len(covered), total),
        "selective_risk": _ratio("selective_risk", errors, len(covered)),
    }


# Calcula um intervalo de confiança por bootstrap percentil sobre amostras.
# Existe porque um conjunto de referência pequeno produz médias instáveis, e
# publicar a média sem intervalo daria a impressão de precisão inexistente.
def bootstrap_interval(
    values: Sequence[float], *, seed: int = 0, resamples: int = 1000, level: float = 0.95
) -> tuple[float, float] | None:
    """Retorna o intervalo percentil de bootstrap da média de ``values``.

    Argumentos:
        values: as observações por amostra.
        seed: semente do gerador, fixada para tornar o intervalo reproduzível.
        resamples: número de reamostragens.
        level: nível de confiança, ex. ``0.95``.
    Retorna:
        ``(lower, upper)``, ou ``None`` com menos de duas observações.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


# Constrói a razão entre duas contagens como MetricValue, tratando
# denominador zero como ausência de suporte. Helper compartilhado por todas
# as métricas de contagem deste módulo.
def _ratio(name: str, numerator: int, denominator: int) -> MetricValue:
    """Retorna ``numerator / denominator`` como métrica, ou sem suporte se vazio."""
    if denominator <= 0:
        return MetricValue(name, None, 0)
    return MetricValue(name, numerator / denominator, denominator)


# Combina precisão e recall em F1, propagando a ausência de suporte de
# qualquer um dos lados. Helper de detection_metrics e set_metrics.
def _harmonic(name: str, precision: MetricValue, recall: MetricValue, *, support: int) -> MetricValue:
    """Retorna o F1 de ``precision`` e ``recall``, ou sem suporte se algum faltar."""
    if not precision.measured or not recall.measured or support <= 0:
        return MetricValue(name, None, 0)
    total = (precision.value or 0.0) + (recall.value or 0.0)
    if math.isclose(total, 0.0):
        return MetricValue(name, 0.0, support)
    return MetricValue(name, 2.0 * (precision.value or 0.0) * (recall.value or 0.0) / total, support)


# Converte scores e acertos em arrays validados, descartando nada e
# recusando entradas inconsistentes. Helper das métricas de calibração.
def _paired(scores: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Converte scores e acertos em arrays float64 alinhados."""
    if len(scores) != len(correct):
        raise ValueError("The scores and correct sequences must have the same length.")
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError("Calibration scores must all lie in [0, 1].")
    return values, np.asarray([1.0 if hit else 0.0 for hit in correct], dtype=np.float64)
