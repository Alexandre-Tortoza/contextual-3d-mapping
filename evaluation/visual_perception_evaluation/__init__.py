"""Métricas reutilizáveis de avaliação de percepção visual.

Issue: #198.

Este pacote consome apenas contracts versionados — o manifest de referência
anotado (#197) e o contract de predição definido em ``prediction.py``. Ele
não importa ``visual_perception``: as métricas precisam permanecer válidas
quando a representação interna do módulo avaliado mudar, e precisam poder
pontuar a saída de qualquer produtor que emita o formato de predição.
"""

from .masks import boundary, boundary_f1, decode_mask, dilate, intersection_over_union
from .metrics import (
    DEFAULT_MATCH_IOU,
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
from .prediction import (
    PREDICTION_SCHEMA_VERSION,
    PredictedRegion,
    PredictedRelation,
    PredictionSet,
    SamplePrediction,
    ScoredValue,
    SupportState,
    load_predictions,
    prediction_set_from_mapping,
    prediction_set_to_mapping,
    save_predictions,
)
from .report import compare, format_metric, render_comparison, render_report
from .suite import (
    EvaluationConfig,
    EvaluationReport,
    RegionOutcome,
    SampleOutcome,
    SemanticDecision,
    UnsupportedClaim,
    evaluate_split,
)

__all__ = [
    "DEFAULT_MATCH_IOU",
    "PREDICTION_SCHEMA_VERSION",
    "EvaluationConfig",
    "EvaluationReport",
    "MetricValue",
    "PredictedRegion",
    "PredictedRelation",
    "PredictionSet",
    "RegionMatch",
    "RegionMatching",
    "RegionOutcome",
    "SampleOutcome",
    "SamplePrediction",
    "ScoredValue",
    "SemanticDecision",
    "SupportState",
    "UnsupportedClaim",
    "abstention_report",
    "bootstrap_interval",
    "boundary",
    "boundary_f1",
    "boundary_metrics",
    "brier_score",
    "compare",
    "decode_mask",
    "detection_metrics",
    "dilate",
    "evaluate_split",
    "expected_calibration_error",
    "format_metric",
    "intersection_over_union",
    "load_predictions",
    "match_regions",
    "open_vocabulary_scores",
    "prediction_set_from_mapping",
    "prediction_set_to_mapping",
    "reliability_table",
    "render_comparison",
    "render_report",
    "save_predictions",
    "set_metrics",
]
