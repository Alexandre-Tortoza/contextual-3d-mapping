"""Testes de round-trip de serialização (#172)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.config import CalibrationConfig, MultiContextConfig
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.semantics import ClaimKind, Evidence, SemanticClaim
from visual_perception.infrastructure.serialization import (
    SUPPORTED_SCHEMA_VERSION,
    UnsupportedSchemaVersionError,
    _claim_from_dict,
    _claim_to_dict,
    deserialize_observation,
    serialize_observation,
)


# Roda o pipeline canônico completo e verifica que serialize/deserialize é um
# round-trip sem perda: regiões, claims e relações devem sobreviver intactos.
def test_round_trip_preserves_regions_claims_and_relations() -> None:
    payload = payload_with_blobs(
        blobs=((2, 2, 8, 8, (200, 30, 30)), (20, 20, 28, 28, (30, 200, 30)))
    )
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    round_tripped = deserialize_observation(serialize_observation(result.observation))

    assert round_tripped == result.observation


# Um payload serializado com um schema_version desconhecido deve falhar de forma
# previsível (UnsupportedSchemaVersionError), nunca ser interpretado silenciosamente
# como se fosse a versão atual.
def test_unsupported_schema_version_fails_predictably() -> None:
    payload = serialize_observation(
        run_canonical_pipeline(
            image_observation(), payload_with_blobs(), default_config(), default_ports()
        ).observation
    )
    payload["schema_version"] = 999
    with pytest.raises(UnsupportedSchemaVersionError):
        deserialize_observation(payload)


# Constrói um claim de label sem score, o caso que o contrato novo do VLM torna
# possível quando o modelo não pontua a hipótese.
def _unscored_claim() -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        "floor",
        None,
        (Evidence("e"),),
        ModelProvenance(stage="region_semantics", producer="qwen_vl", config_fingerprint="abc"),
    )


# Caso central do passo na fronteira de persistência: a ausência de score precisa
# sobreviver ao round-trip, senão ela volta do disco como um número inventado.
def test_unscored_claim_survives_round_trip() -> None:
    assert _claim_from_dict(_claim_to_dict(_unscored_claim())).confidence is None


# Protege contra o conserto errado: gravar 0.0 ou 1.0 no lugar da ausência
# reintroduziria exatamente a regressão que este passo existe para matar.
def test_unscored_claim_does_not_serialize_as_zero_or_one() -> None:
    assert _claim_to_dict(_unscored_claim())["confidence"] is None


# Uma observação com evidência multi-contexto (#193) e suporte calibrado
# (#195) precisa sobreviver ao round-trip inteira: perder um slot ou um
# estado de calibração no disco reintroduziria a ambiguidade que os dois
# contracts existem para eliminar.
def test_round_trip_preserves_multi_context_evidence_and_calibrated_support(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "calibration_version": "calibration/1",
                "domain": "indoor_corridor",
                "sources": {"fake": {"label": {"bins": [[0.0, 1.0, 0.6]], "samples": 40}}},
            }
        ),
        encoding="utf-8",
    )
    config = dataclasses.replace(
        default_config(),
        multi_context=MultiContextConfig(contextual_crop_enabled=True, scene_conditioned_enabled=True),
        calibration=CalibrationConfig(
            enabled=True, artifact_path=str(artifact), domain="indoor_corridor"
        ),
    )
    result = run_canonical_pipeline(image_observation(), payload_with_blobs(), config, default_ports())

    round_tripped = deserialize_observation(serialize_observation(result.observation))

    assert round_tripped == result.observation
    region = round_tripped.regions[0]
    assert {slot.slot for slot in region.evidence} == set(EvidenceSlot)
    assert all(claim.support is not None for claim in region.claims)


# Um payload gravado no schema v1 (sem evidência nem suporte) precisa ser
# migrado explicitamente, e não rejeitado nem reinterpretado como v2.
def test_a_v1_payload_is_migrated_explicitly_instead_of_being_rejected() -> None:
    payload = serialize_observation(
        run_canonical_pipeline(
            image_observation(), payload_with_blobs(), default_config(), default_ports()
        ).observation
    )
    payload["schema_version"] = 1
    for region in payload["regions"]:
        region.pop("evidence")
        for claim in region["claims"]:
            claim.pop("support")
    for relation in payload["relations"]:
        relation["confidence"] = {"value": 1.0, "source": "geometric_2d"}

    migrated = deserialize_observation(payload)

    assert migrated.schema_version == SUPPORTED_SCHEMA_VERSION
    assert migrated.regions[0].evidence == ()
    assert all(claim.support is None for claim in migrated.regions[0].claims)


# Um RLE corrompido precisa falhar na fronteira de leitura, e não produzir
# uma máscara silenciosamente truncada ou com pixels a mais.
def test_a_corrupt_mask_encoding_is_rejected_at_the_boundary() -> None:
    payload = serialize_observation(
        run_canonical_pipeline(
            image_observation(), payload_with_blobs(), default_config(), default_ports()
        ).observation
    )
    payload["regions"][0]["mask"]["rle"] = [3, 4]

    with pytest.raises(ValueError, match="covering the complete image"):
        deserialize_observation(payload)
