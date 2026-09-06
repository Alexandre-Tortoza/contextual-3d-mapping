"""Testes do contract versionado de predição usado pela avaliação (#198/#199)."""

from __future__ import annotations

from visual_perception_evaluation.prediction import (
    PredictionSet,
    SamplePrediction,
    ScoredValue,
    SupportState,
    prediction_set_from_mapping,
    prediction_set_to_mapping,
)


# Protege os campos que tornam uma falha calibrada auditável. Existe porque
# a #199 precisa reconstruir evidência e proveniência sem importar o produtor.
def test_scored_value_provenance_round_trips_through_prediction_contract() -> None:
    """Preserva fonte, versão, artifact de calibração e artifacts de evidência."""
    predictions = PredictionSet(
        run_id="run",
        configuration="calibrated",
        config_fingerprint="config",
        code_revision="revision",
        samples=(
            SamplePrediction(
                "sample",
                8,
                8,
                scene_claims=(
                    ScoredValue(
                        "corridor",
                        raw_confidence=0.9,
                        calibrated_confidence=0.7,
                        support_state=SupportState.CALIBRATED,
                        source="vlm",
                        calibration_version="calibration/1",
                        calibration_artifact="sha256:table",
                        evidence_artifacts=("artifact://scene", "artifact://crop"),
                    ),
                ),
            ),
        ),
    )

    restored = prediction_set_from_mapping(prediction_set_to_mapping(predictions))

    assert restored == predictions
