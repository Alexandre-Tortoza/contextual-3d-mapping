"""Testes do experimento de calibração, abstenção e claims sem suporte (#199)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from contextual_mapping_datasets import Split
from visual_perception.application.semantic_calibration import load_calibration_artifact
from visual_perception_evaluation.prediction import (
    PredictedRegion,
    PredictionSet,
    SamplePrediction,
    ScoredValue,
    SupportState,
)
from visual_perception_evaluation.suite import EvaluationConfig
from visual_perception_experiments.evaluate_calibration import (
    collect_failures,
    operational_summary,
    render_study,
    study_calibration,
    study_to_mapping,
)
from visual_perception_experiments.fit_calibration import (
    Observation,
    fit_bins,
    fit_reliability_table,
)
from visual_perception_experiments.fixture_reference import (
    FIXTURE_SIZE,
    build_fixture_manifest,
    encode_runs,
)


# Constrói uma máscara pequena fora das regiões anotadas. Existe para criar
# uma claim comprovadamente sem suporte na fixture sintética.
def _unsupported_runs() -> tuple[int, ...]:
    """Retorna o RLE de uma região sem interseção com a referência de teste."""
    mask = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=np.bool_)
    mask[1:4, 58:61] = True
    return encode_runs(mask)


# Monta uma execução com erro, abstenção e alucinação observáveis. Existe
# para que os testes da #199 compartilhem exatamente o mesmo cenário.
def _test_predictions() -> PredictionSet:
    """Retorna predições sintéticas que cobrem os casos centrais da #199."""
    sample = build_fixture_manifest().samples_in(Split.TEST)[0]
    floor, ignored = sample.regions
    return PredictionSet(
        run_id="run-199",
        configuration="calibrated",
        config_fingerprint="config-199",
        code_revision="deadbeef",
        hardware="fixture-cpu",
        samples=(
            SamplePrediction(
                sample_id=sample.sample_id,
                width=sample.width,
                height=sample.height,
                latency_s=0.5,
                peak_vram_bytes=1024,
                model_calls=2,
                regions=(
                    PredictedRegion(
                        "pred-floor",
                        sample.width,
                        sample.height,
                        floor.mask.runs,
                        labels=(
                            ScoredValue(
                                "parede",
                                raw_confidence=0.9,
                                calibrated_confidence=0.6,
                                support_state=SupportState.CALIBRATED,
                                source="vlm",
                                calibration_version="calibration/1",
                                calibration_artifact="sha256:calibration",
                                evidence_artifacts=("artifact://tight-floor",),
                            ),
                        ),
                        evidence_slots=("tight_crop",),
                    ),
                    PredictedRegion(
                        "pred-ignored",
                        sample.width,
                        sample.height,
                        ignored.mask.runs,
                        labels=(
                            ScoredValue(
                                "incerto",
                                support_state=SupportState.ABSTAINED,
                                reason="suporte visual insuficiente",
                            ),
                        ),
                    ),
                    PredictedRegion(
                        "pred-unsupported",
                        sample.width,
                        sample.height,
                        _unsupported_runs(),
                        labels=(ScoredValue("cone", raw_confidence=0.8, source="vlm"),),
                    ),
                ),
            ),
        ),
    )


# Confere a visão operacional que não depende de anotação. Protege a regra
# de que claims sem score permanecem visíveis na cobertura.
def test_operational_summary_keeps_unscored_claims_in_coverage() -> None:
    """Conta claims abstidas sem inventar score para elas."""
    summary = operational_summary(_test_predictions())

    assert summary["label_claims"] == 3
    assert summary["support_states"] == {"abstained": 1, "calibrated": 1, "unknown": 1}
    assert summary["raw_scored_fraction"] == pytest.approx(2 / 3)
    assert summary["calibrated_scored_fraction"] == pytest.approx(1 / 3)


# Exercita o report completo da #199. Existe para garantir que bruto e
# calibrado nunca sejam agregados no mesmo campo e que os recortes sobrevivam.
def test_calibration_study_separates_reliability_and_serializes_strata() -> None:
    """Produz confiabilidade separada e mantém os recortes no artifact JSON."""
    study = study_calibration(
        build_fixture_manifest(),
        _test_predictions(),
        Split.TEST,
        EvaluationConfig(bootstrap_resamples=20),
    )

    assert set(study.report.calibration) == {"raw", "calibrated"}
    assert "label" in study.by_claim_kind
    assert "vlm" in study.by_source
    assert "small_region" in study.by_stratum
    assert "visibility:clear" in study.by_stratum
    assert "normal_light" in study.by_stratum
    assert "scene:corridor-02" in study.by_stratum
    document = study_to_mapping(study)
    assert document["by_stratum"]
    assert document["by_source"]
    assert document["calibration"]["raw"] != document["calibration"]["calibrated"]
    markdown = render_study(study)
    assert "Confiabilidade: bruto versus calibrado" in markdown
    assert (
        "Ambiguidade" not in markdown
    )  # o critério usa a grafia em minúsculas no texto
    assert "ambiguidade não conta como erro" in markdown


# Verifica a fronteira entre erro, região ignorada e alucinação. Existe para
# impedir que ambiguidade ou conteúdo ignorado inflem a taxa da #199.
def test_failure_cases_link_evidence_and_do_not_count_ignored_regions() -> None:
    """Mantém falhas auditáveis e absorve a predição sobre região ignorada."""
    failures = collect_failures(
        build_fixture_manifest(),
        _test_predictions(),
        Split.TEST,
        EvaluationConfig(bootstrap_resamples=20),
    )

    assert [failure.kind for failure in failures] == [
        "wrong_label",
        "unsupported_claim",
    ]
    wrong = failures[0]
    assert wrong.predicted_region_id == "pred-floor"
    assert wrong.evidence_artifacts == ("artifact://tight-floor",)
    assert wrong.calibration_version == "calibration/1"
    assert wrong.calibration_artifact == "sha256:calibration"
    assert "visibility:clear" in wrong.strata
    assert "scene:corridor-02" in wrong.strata
    assert all(failure.predicted_region_id != "pred-ignored" for failure in failures)


# Compara bins contra um exemplo calculável à mão. Existe para assegurar que
# o ajuste mede acurácia empírica, não apenas remapeia o score de entrada.
def test_reliability_bins_match_hand_computed_accuracy() -> None:
    """Calcula 50% de acerto para duas observações no mesmo bin."""
    table = fit_bins(
        (
            Observation("label", 0.8, True),
            Observation("label", 0.9, False),
        ),
        bins=2,
        min_samples=2,
    )

    assert table == {"label": [[0.5, 1.0, 0.5]]}


# Exercita a produção real do artifact consumido pelo calibrador. Existe para
# ligar manifest, predição, split e proveniência em uma execução reproduzível.
def test_fitted_artifact_is_versioned_and_loadable(tmp_path: Path) -> None:
    """Ajusta a fixture de calibração e carrega o artifact pelo módulo produtor."""
    manifest = build_fixture_manifest()
    sample = manifest.samples_in(Split.CALIBRATION)[0]
    annotation = sample.regions[0]
    predictions = PredictionSet(
        run_id="fit-run",
        configuration="raw",
        config_fingerprint="fit-config",
        code_revision="deadbeef",
        samples=(
            SamplePrediction(
                sample.sample_id,
                sample.width,
                sample.height,
                regions=(
                    PredictedRegion(
                        "pred-door",
                        sample.width,
                        sample.height,
                        annotation.mask.runs,
                        labels=(
                            ScoredValue("porta", raw_confidence=0.9, source="vlm"),
                        ),
                    ),
                ),
            ),
        ),
    )
    destination = tmp_path / "calibration.json"

    document = fit_reliability_table(
        manifest,
        predictions,
        destination,
        source="vlm",
        domain="synthetic",
        bins=2,
        min_samples=1,
    )
    artifact = load_calibration_artifact(destination)

    assert document["calibration_version"] == "calibration/1"
    assert document["fitted_from"]["split"] == "calibration"
    assert artifact.version == "calibration/1"
    assert artifact.domain == "synthetic"
