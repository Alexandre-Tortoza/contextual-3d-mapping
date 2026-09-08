"""Testes da pontuação calibrada de claims e da abstenção explícita (#196)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.application.quality_audit import audit_observation
from visual_perception.application.semantic_calibration import (
    ReliabilityTableCalibrator,
    UncalibratedSupport,
    build_calibrator,
    calibrate_observation_claims,
    derive_support_inputs,
    load_calibration_artifact,
)
from visual_perception.config import CalibrationConfig
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace
from visual_perception.domain.geometry import CoordinateTransform, Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import (
    EvidenceSlot,
    EvidenceState,
    RegionEvidenceSlot,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantic_support import SupportPolarity, SupportState
from visual_perception.domain.semantics import (
    IDENTITY_CLAIM_KINDS,
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="vlm", config_fingerprint="abc123")
_EVIDENCE = (Evidence(description="raw region response"),)

_ARTIFACT = {
    "calibration_version": "calibration/1",
    "domain": "indoor_corridor",
    "sources": {
        "vlm": {
            "label": {
                "bins": [[0.0, 0.5, 0.10], [0.5, 0.8, 0.42], [0.8, 1.0, 0.71]],
                "samples": 250,
            }
        }
    },
}


# Grava um artifact de calibração no diretório temporário do teste e devolve
# a configuração que o seleciona, para que cada teste declare seus dados.
def _artifact_config(
    tmp_path: Path, document: dict[str, object] | None = None, **overrides: object
) -> CalibrationConfig:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document or _ARTIFACT), encoding="utf-8")
    defaults: dict[str, object] = {
        "enabled": True,
        "artifact_path": str(path),
        "domain": "indoor_corridor",
        "min_visual_support": 0.0,
    }
    defaults.update(overrides)
    return CalibrationConfig(**defaults)  # type: ignore[arg-type]


# Constrói uma região com um slot de foreground que declara a cobertura de
# suporte pedida, para exercitar a regra de suporte insuficiente.
def _region(support_ratio: float | None = 0.9, claims: tuple[SemanticClaim, ...] = ()) -> ObservedRegion:
    data = np.zeros((32, 32), dtype=np.bool_)
    data[4:12, 4:12] = True
    mask = Mask(data, 32, 32)
    evidence: tuple[RegionEvidenceSlot, ...] = ()
    if support_ratio is not None:
        evidence = (
            RegionEvidenceSlot(
                slot=EvidenceSlot.FOREGROUND_DENSE,
                region_id="region-a",
                state=EvidenceState.AVAILABLE,
                crop_box=mask.bounding_box(),
                transform=CoordinateTransform.identity(),
                preprocessing="mask_aware_pooling:pixel_nearest_highres",
                artifact_ref="visual-region-a",
                space=EmbeddingSpace(
                    "dinov2", "facebook/dinov2-base", 768, EmbeddingModality.VISUAL_DENSE
                ),
                support_ratio=support_ratio,
            ),
        )
    return ObservedRegion(
        region_id="region-a",
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=0.8,
        contributing_proposal_ids=("proposal-1",),
        claims=claims,
        evidence=evidence,
    )


# Constrói uma claim de label pontuada com o valor bruto pedido.
def _claim(value: str, raw: float | None = 0.9, kind: ClaimKind = ClaimKind.LABEL) -> SemanticClaim:
    confidence = None if raw is None else ConfidenceScore(raw, source="vlm")
    role = HypothesisRole.PRIMARY if kind in IDENTITY_CLAIM_KINDS else None
    return SemanticClaim(kind, value, confidence, _EVIDENCE, _PROVENANCE, role=role)


def test_a_calibrated_score_is_emitted_only_when_the_rule_declares_it_valid(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    region = _region(claims=(_claim("door", 0.9),))

    regions, _, failures = calibrate_observation_claims(
        (region,), SceneContext(), build_calibrator(config), config
    )

    support = regions[0].claims[0].support
    assert support.state is SupportState.CALIBRATED
    assert support.calibrated_confidence.value == pytest.approx(0.71)
    assert regions[0].claims[0].confidence.value == 0.9  # o bruto continua intacto
    assert failures == ()


def test_calibration_is_traceable_through_versioned_provenance(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    artifact = load_calibration_artifact(Path(config.artifact_path))

    regions, _, _ = calibrate_observation_claims(
        (_region(claims=(_claim("door"),)),), SceneContext(), build_calibrator(config), config
    )

    support = regions[0].claims[0].support
    assert support.calibration_version == "calibration/1"
    assert support.calibration_artifact == artifact.digest
    assert support.domain == "indoor_corridor"


def test_an_unscored_claim_abstains_instead_of_becoming_false_certainty(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)

    regions, _, _ = calibrate_observation_claims(
        (_region(claims=(_claim("door", raw=None),)),), SceneContext(), build_calibrator(config), config
    )

    support = regions[0].claims[0].support
    assert support.state is SupportState.ABSTAINED
    assert "unscored source" in support.reason
    assert support.calibrated_confidence is None


def test_an_unsupported_claim_type_abstains_with_an_explicit_reason(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    scene = SceneContext(claims=(_claim("corridor", 0.9, ClaimKind.SCENE_TYPE),))

    _, calibrated_scene, _ = calibrate_observation_claims(
        (), scene, build_calibrator(config), config
    )

    support = calibrated_scene.claims[0].support
    assert support.state is SupportState.ABSTAINED
    assert "unsupported claim type" in support.reason


def test_out_of_distribution_evidence_abstains_instead_of_extrapolating(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path, domain="outdoor_rubble")

    regions, _, _ = calibrate_observation_claims(
        (_region(claims=(_claim("door"),)),), SceneContext(), build_calibrator(config), config
    )

    support = regions[0].claims[0].support
    assert support.state is SupportState.ABSTAINED
    assert "out-of-distribution evidence" in support.reason


def test_insufficient_visual_support_abstains_and_reports_the_measurement(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path, min_visual_support=0.5)

    weak, _, _ = calibrate_observation_claims(
        (_region(support_ratio=0.2, claims=(_claim("door"),)),),
        SceneContext(),
        build_calibrator(config),
        config,
    )
    unmeasured, _, _ = calibrate_observation_claims(
        (_region(support_ratio=None, claims=(_claim("door"),)),),
        SceneContext(),
        build_calibrator(config),
        config,
    )

    assert "insufficient support" in weak[0].claims[0].support.reason
    assert "0.200" in weak[0].claims[0].support.reason
    assert "not measured" in unmeasured[0].claims[0].support.reason


def test_pre_calibration_support_is_preserved_for_audit(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    region = _region(claims=(_claim("door", 0.9), _claim("window", 0.6)))

    regions, _, _ = calibrate_observation_claims(
        (region,), SceneContext(), build_calibrator(config), config
    )

    support = regions[0].claims[0].support
    assert support.visual_support == pytest.approx(0.9)
    assert support.region_quality == pytest.approx(0.8)
    assert support.contradiction_support == pytest.approx(1.0)  # a única irmã discorda
    assert support.source_reliability == pytest.approx(1.0)  # 250 amostras saturam em 1.0


def test_conflicting_hypotheses_stay_side_by_side_after_calibration(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    region = _region(claims=(_claim("door", 0.9), _claim("window", 0.6)))

    regions, _, _ = calibrate_observation_claims(
        (region,), SceneContext(), build_calibrator(config), config
    )

    values = {claim.value: claim.support.calibrated_confidence.value for claim in regions[0].claims}
    assert values == {"door": pytest.approx(0.71), "window": pytest.approx(0.42)}


def test_disabled_calibration_keeps_the_raw_score_without_promoting_it() -> None:
    config = CalibrationConfig()
    calibrator = build_calibrator(config)

    regions, _, failures = calibrate_observation_claims(
        (_region(claims=(_claim("door"),)),), SceneContext(), calibrator, config
    )

    assert isinstance(calibrator, UncalibratedSupport)
    support = regions[0].claims[0].support
    assert support.state is SupportState.RAW
    assert support.calibrated_confidence is None
    assert failures == ()


def test_a_broken_calibration_artifact_becomes_an_auditable_failure(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{ not json", encoding="utf-8")
    config = CalibrationConfig(enabled=True, artifact_path=str(path), domain="indoor_corridor")

    regions, _, failures = calibrate_observation_claims(
        (_region(claims=(_claim("door"),)),), SceneContext(), build_calibrator(config), config
    )

    support = regions[0].claims[0].support
    assert support.state is SupportState.FAILED
    assert "could not be loaded" in support.reason
    assert [failure.value for failure in failures] == ["door"]


def test_the_quality_auditor_separates_calibration_failure_from_semantic_disagreement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{ not json", encoding="utf-8")
    config = CalibrationConfig(enabled=True, artifact_path=str(path), domain="indoor_corridor")
    region = _region(claims=(_claim("door", 0.9), _claim("window", 0.6)))
    regions, _, _ = calibrate_observation_claims(
        (region,), SceneContext(), build_calibrator(config), config
    )
    observation = VisualObservation(
        source=image_observation().source,
        image_width=32,
        image_height=32,
        scene_context=SceneContext(),
        regions=regions,
        relations=(),
    )

    codes = {issue.code for issue in audit_observation(observation).warnings}

    assert "calibration_failed" in codes
    assert "contradictory_claims" in codes


def test_an_artifact_with_the_wrong_version_is_refused_instead_of_applied(tmp_path: Path) -> None:
    document = {**_ARTIFACT, "calibration_version": "calibration/99"}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported calibration_version"):
        load_calibration_artifact(path)


def test_malformed_calibration_bins_are_rejected(tmp_path: Path) -> None:
    for bins, message in (
        ([[0.5, 0.2, 0.4]], "lower < upper"),
        ([[0.0, 0.5, 1.4]], r"must map into \[0, 1\]"),
        ([[0.0, 0.6, 0.2], [0.4, 0.9, 0.5]], "sorted and non-overlapping"),
    ):
        document = {
            **_ARTIFACT,
            "sources": {"vlm": {"label": {"bins": bins, "samples": 10}}},
        }
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_calibration_artifact(path)


def test_a_raw_score_outside_the_measured_bins_abstains(tmp_path: Path) -> None:
    document = {
        **_ARTIFACT,
        "sources": {"vlm": {"label": {"bins": [[0.0, 0.5, 0.1]], "samples": 10}}},
    }
    config = _artifact_config(tmp_path, document)

    regions, _, _ = calibrate_observation_claims(
        (_region(claims=(_claim("door", 0.95),)),), SceneContext(), build_calibrator(config), config
    )

    assert "outside the measured bins" in regions[0].claims[0].support.reason


def test_support_inputs_are_derived_from_evidence_that_was_actually_measured() -> None:
    region = _region(support_ratio=0.42)
    claim = _claim("door")

    inputs = derive_support_inputs(claim, region=region, siblings=(), domain="indoor_corridor")

    assert inputs.visual_support == pytest.approx(0.42)
    assert inputs.region_quality == pytest.approx(0.8)
    assert [item.artifact_uri for item in inputs.evidence] == ["visual-region-a"]


def test_the_canonical_pipeline_attaches_support_to_every_claim() -> None:
    config = default_config()
    payload = payload_with_blobs()

    result = run_canonical_pipeline(image_observation(), payload, config, default_ports())

    assert result.calibration_failures == ()
    for claim in result.observation.scene_context.claims:
        assert claim.support is not None
    for region in result.observation.regions:
        for claim in region.claims:
            assert claim.support is not None


def test_a_reliability_table_calibrator_reports_its_own_provenance(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    calibrator = build_calibrator(config)

    assert isinstance(calibrator, ReliabilityTableCalibrator)
    assert calibrator.version == "calibration/1"
    assert calibrator.artifact_id is not None


# Regressão da #202 no sinal que a calibração consome: até então
# ``derive_support_inputs`` contava como contradição toda irmã de mesmo kind com
# ``value`` diferente. Medido em ``corridor-02-002``, isso deixava
# contradiction_support = 1.0 nos atributos de todas as 60 regiões.
def test_coexisting_attributes_do_not_raise_contradiction_support() -> None:
    """Atributos que coexistem não são medidos como contradição."""
    attributes = (
        _claim("smooth", raw=None, kind=ClaimKind.ATTRIBUTE),
        _claim("white", raw=None, kind=ClaimKind.ATTRIBUTE),
        _claim("damaged", raw=None, kind=ClaimKind.ATTRIBUTE),
    )
    inputs = derive_support_inputs(
        attributes[0],
        region=_region(claims=attributes),
        siblings=attributes[1:],
        domain="unspecified",
    )

    assert inputs.contradiction_support == pytest.approx(0.0)
    assert not [
        item for item in inputs.evidence if item.polarity is SupportPolarity.CONTRADICTORY
    ]


# A contrapartida: hipóteses de identidade concorrentes continuam elevando o
# sinal, e a evidência contraditória continua nomeando quem discordou.
def test_a_competing_identity_hypothesis_still_raises_contradiction_support() -> None:
    """Uma alternativa de identidade continua contando como contradição."""
    primary = _claim("wall", 0.9)
    alternative = _claim("door", 0.2)
    inputs = derive_support_inputs(
        primary,
        region=_region(claims=(primary, alternative)),
        siblings=(alternative,),
        domain="unspecified",
    )

    assert inputs.contradiction_support == pytest.approx(1.0)
    assert [
        item.artifact_uri
        for item in inputs.evidence
        if item.polarity is SupportPolarity.CONTRADICTORY
    ] == ["claim:label:door"]
