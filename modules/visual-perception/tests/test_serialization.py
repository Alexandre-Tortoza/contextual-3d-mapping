"""Testes de round-trip de serialização (#172)."""

from __future__ import annotations

import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import ClaimKind, Evidence, SemanticClaim
from visual_perception.infrastructure.serialization import (
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
