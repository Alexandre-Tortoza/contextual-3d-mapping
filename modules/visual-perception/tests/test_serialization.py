"""Testes de round-trip de serialização (#172)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.config import CalibrationConfig, MultiContextConfig
from visual_perception.domain.contextual_entities import (
    ContextualEntityHypothesis,
    EntityHypothesisKind,
    EntityHypothesisStatus,
)
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantic_support import HypothesisSupportSignal, SupportSignalStatus
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation
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
        role=HypothesisRole.PRIMARY,
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


# O contract da #202 só vale se atravessar a serialização: papel, category e
# RegionKind precisam voltar iguais, e não apenas existir em memória.
def test_identity_fields_survive_the_round_trip() -> None:
    """``role``, ``category`` e ``region_kind`` sobrevivem à serialização."""
    provenance = ModelProvenance(
        stage="region_semantics", producer="qwen_vl", config_fingerprint="abc"
    )
    primary = SemanticClaim(
        ClaimKind.LABEL,
        "plain wall",
        ConfidenceScore(0.9, source="qwen_vl"),
        (Evidence("e"),),
        provenance,
        role=HypothesisRole.PRIMARY,
        category="wall",
        region_kind=RegionKind.STUFF,
    )
    alternative = SemanticClaim(
        ClaimKind.LABEL, "panel", None, (Evidence("e"),), provenance, role=HypothesisRole.ALTERNATIVE
    )
    mask = Mask(np.ones((4, 4), dtype=np.bool_), 4, 4)
    region = ObservedRegion(
        "region-a", mask, mask.bounding_box(), 0.9, ("p-1",), (primary, alternative)
    )
    observation = VisualObservation(
        source=image_observation(width=4, height=4).source,
        image_width=4,
        image_height=4,
        scene_context=SceneContext(),
        regions=(region,),
        relations=(),
    )

    restored = deserialize_observation(serialize_observation(observation))

    assert restored == observation
    restored_primary, restored_alternative = restored.regions[0].claims
    assert restored_primary.role is HypothesisRole.PRIMARY
    assert restored_primary.category == "wall"
    assert restored_primary.region_kind is RegionKind.STUFF
    assert restored_alternative.role is HypothesisRole.ALTERNATIVE
    assert restored_alternative.category is None


# Um payload gravado antes da #202 não tem papel registrado. Rejeitá-lo seria
# perder observações válidas; assumir "primary" é a única leitura que aquele
# formato admitia, já que ele não distinguia hipóteses.
def test_a_payload_without_roles_is_read_as_primary_instead_of_being_rejected() -> None:
    """Um claim de label sem papel gravado é migrado para ``PRIMARY``."""
    provenance = ModelProvenance(
        stage="region_semantics", producer="qwen_vl", config_fingerprint="abc"
    )
    claim = SemanticClaim(
        ClaimKind.LABEL, "wall", None, (Evidence("e"),), provenance, role=HypothesisRole.PRIMARY
    )
    mask = Mask(np.ones((4, 4), dtype=np.bool_), 4, 4)
    observation = VisualObservation(
        source=image_observation(width=4, height=4).source,
        image_width=4,
        image_height=4,
        scene_context=SceneContext(),
        regions=(ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p-1",), (claim,)),),
        relations=(),
    )
    payload = serialize_observation(observation)
    for serialized in payload["regions"][0]["claims"]:
        del serialized["role"]

    restored = deserialize_observation(payload)

    assert restored.regions[0].claims[0].role is HypothesisRole.PRIMARY


# Os dois contracts que a #214/#205 acrescentaram precisam sobreviver ao disco
# inteiros: um sinal que perdesse o espaço em que foi medido, ou um grupo que
# perdesse os membros, voltariam como afirmações que ninguém pode auditar.
def test_signals_and_entity_hypotheses_survive_the_round_trip() -> None:
    """Sinais de suporte e hipóteses de entidade fazem round-trip sem perda."""
    space = EmbeddingSpace(
        "clip", "openai/clip-vit-large-patch14", 768, EmbeddingModality.LANGUAGE_ALIGNED
    )
    signals = (
        HypothesisSupportSignal(
            source="clip_alignment",
            hypothesis="wall",
            slot=EvidenceSlot.TIGHT_CROP,
            status=SupportSignalStatus.SUPPORTS,
            score=0.21,
            margin=0.03,
            space=space,
        ),
        HypothesisSupportSignal(
            source="clip_alignment",
            hypothesis="wall",
            slot=EvidenceSlot.CONTEXTUAL_CROP,
            status=SupportSignalStatus.UNAVAILABLE,
            reason="no language-aligned embedding available",
        ),
    )
    data = np.zeros((16, 16), dtype=np.bool_)
    data[2:8, 2:8] = True
    mask = Mask(data, 16, 16)
    claim = SemanticClaim(
        ClaimKind.LABEL,
        "wall",
        ConfidenceScore(0.9, source="qwen_vl"),
        (Evidence("raw region response"),),
        ModelProvenance(stage="region_semantics", producer="qwen_vl", config_fingerprint="fp"),
        role=HypothesisRole.PRIMARY,
        region_kind=RegionKind.STUFF,
        signals=signals,
    )
    regions = (
        ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p1",), claims=(claim,)),
        ObservedRegion("region-b", mask, mask.bounding_box(), 0.8, ("p2",), claims=(claim,)),
    )
    entity = ContextualEntityHypothesis(
        entity_id="entity-0123456789abcdef",
        kind=EntityHypothesisKind.SAME_SURFACE,
        canonical_concept="wall",
        region_kind=RegionKind.STUFF,
        member_region_ids=("region-a", "region-b"),
        status=EntityHypothesisStatus.SUPPORTED,
        evidence=(Evidence("2 touching regions share the reconciled concept 'wall'"),),
        provenance=ModelProvenance(
            stage="intra_frame_reconciliation",
            producer="intra_frame_reconciliation",
            config_fingerprint="fp",
        ),
        feature_coherence=0.81,
        space=EmbeddingSpace("dinov2", "facebook/dinov2-base", 768, EmbeddingModality.VISUAL_DENSE),
        member_raw_labels=("wall", "plain wall"),
    )
    observation = VisualObservation(
        source=image_observation().source,
        image_width=16,
        image_height=16,
        scene_context=SceneContext(),
        regions=regions,
        relations=(),
        entity_hypotheses=(entity,),
    )

    round_tripped = deserialize_observation(serialize_observation(observation))

    assert round_tripped == observation
    assert round_tripped.entity_hypotheses[0].feature_coherence == 0.81
    assert round_tripped.regions[0].claims[0].signals[1].reason


# Um payload da v2 continua legível: ele simplesmente não tem sinais nem
# grupos, e desserializa com os dois campos vazios em vez de ser recusado.
def test_a_v2_payload_is_read_without_signals_or_entities() -> None:
    """Um payload anterior à #214 desserializa com os campos novos vazios."""
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())
    document = serialize_observation(result.observation)
    document["schema_version"] = 2
    document.pop("entity_hypotheses")
    for region in document["regions"]:
        for claim in region["claims"]:
            claim.pop("signals")

    migrated = deserialize_observation(document)

    assert migrated.schema_version == SUPPORTED_SCHEMA_VERSION
    assert migrated.entity_hypotheses == ()
    assert all(claim.signals == () for region in migrated.regions for claim in region.claims)
