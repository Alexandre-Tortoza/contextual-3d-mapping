"""Fronteira de serialização da observação visual canônica.

Issue: #172.

Masks são embutidas como booleanos compactados via run-length encoding
(pequeno, exato e autocontido) em vez de irem para um artifact store externo,
já que a própria geometria de uma região é necessária para sequer interpretar
a observação. Embeddings visuais/de linguagem e evidência bruta de modelo são
referenciados só por id (``visual_embedding_ref`` / ``language_embedding_ref``
/ ``Evidence.artifact``), conforme a regra "referencie payloads grandes" de
#154.
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import CandidateRelation, RelationSource
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext, VisualObservation

#: A única versão de schema que este módulo consegue ler. Incremente junto
#: com uma migração ou um erro explícito de incompatibilidade, nunca em silêncio.
SUPPORTED_SCHEMA_VERSION = 1


# Sinaliza que um payload foi serializado com uma versão de schema que este
# módulo não sabe interpretar, em vez de deixar a desserialização falhar de
# forma obscura mais adiante.
class UnsupportedSchemaVersionError(ValueError):
    """Levantada ao desserializar um payload de uma versão de schema incompatível."""


# Converte uma VisualObservation canônica em um dict simples, pronto para
# JSON — o ponto de entrada da serialização, usado por persist_observation em
# persistence_integration.py para gravar a observação no evidence store.
def serialize_observation(observation: VisualObservation) -> dict[str, Any]:
    """Serializa uma VisualObservation canônica em um dict simples, pronto para JSON."""
    if observation.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Cannot serialize schema_version={observation.schema_version}; "
            f"this module writes schema_version={SUPPORTED_SCHEMA_VERSION}."
        )
    return {
        "schema_version": observation.schema_version,
        "coordinate_convention": observation.coordinate_convention,
        "source": _source_to_dict(observation.source),
        "image_width": observation.image_width,
        "image_height": observation.image_height,
        "scene_context": {"claims": [_claim_to_dict(claim) for claim in observation.scene_context.claims]},
        "regions": [_region_to_dict(region) for region in observation.regions],
        "relations": [_relation_to_dict(relation) for relation in observation.relations],
    }


# Reconstrói uma VisualObservation a partir do dict produzido por
# serialize_observation — o lado inverso da serialização, usado por
# reload_observation em persistence_integration.py.
def deserialize_observation(payload: dict[str, Any]) -> VisualObservation:
    """Reconstrói uma VisualObservation canônica, fazendo o round-trip sem perda de informação."""
    schema_version = payload.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Cannot deserialize schema_version={schema_version!r}; "
            f"this module reads schema_version={SUPPORTED_SCHEMA_VERSION}."
        )
    return VisualObservation(
        source=_source_from_dict(payload["source"]),
        image_width=payload["image_width"],
        image_height=payload["image_height"],
        scene_context=SceneContext(
            claims=tuple(_claim_from_dict(claim) for claim in payload["scene_context"]["claims"])
        ),
        regions=tuple(_region_from_dict(region) for region in payload["regions"]),
        relations=tuple(_relation_from_dict(relation) for relation in payload["relations"]),
        schema_version=schema_version,
        coordinate_convention=payload["coordinate_convention"],
    )


# Converte a ObservationReference (identidade de origem, timestamp, frame,
# calibração) em um dict serializável — helper interno usado por
# serialize_observation.
def _source_to_dict(source: ObservationReference) -> dict[str, Any]:
    return {
        "observation_id": source.observation_id,
        "dataset_id": source.dataset_id,
        "sequence_id": source.sequence_id,
        "sensor_id": source.sensor_id,
        "sequence_index": source.sequence_index,
        "timestamp": {"nanoseconds": source.timestamp.nanoseconds, "clock_id": source.timestamp.clock_id},
        "frame_id": source.frame_id.value,
        "calibration_id": source.calibration_id,
    }


# Reconstrói a ObservationReference a partir do dict, incluindo os tipos
# fortes Timestamp/FrameId — lado inverso de _source_to_dict, usado por
# deserialize_observation.
def _source_from_dict(payload: dict[str, Any]) -> ObservationReference:
    payload = dict(payload)
    payload["timestamp"] = Timestamp(**payload["timestamp"])
    payload["frame_id"] = FrameId(payload["frame_id"])
    return ObservationReference(**payload)


# Compacta a mask booleana via run-length encoding para manter o payload
# serializado pequeno e exato, sem depender de um artifact store externo —
# ver a justificativa de design na docstring do módulo.
def _mask_to_dict(mask: Mask) -> dict[str, Any]:
    flat = mask.data.reshape(-1)
    runs = [len(list(group)) for _, group in groupby(flat.tolist())]
    if flat.size > 0 and bool(flat[0]):
        runs = [0, *runs]  # O RLE sempre começa com uma contagem de run False, mesmo que zero.
    return {"width": mask.image_width, "height": mask.image_height, "rle": runs}


# Reconstrói a mask booleana a partir da codificação RLE — lado inverso de
# _mask_to_dict.
def _mask_from_dict(payload: dict[str, Any]) -> Mask:
    width, height, runs = payload["width"], payload["height"], payload["rle"]
    flat = np.zeros(width * height, dtype=np.bool_)
    cursor = 0
    value = False
    for run_length in runs:
        if value:
            flat[cursor : cursor + run_length] = True
        cursor += run_length
        value = not value
    return Mask(flat.reshape(height, width), width, height)


# Converte a BoundingBox em um dict simples de coordenadas — helper interno
# usado por _region_to_dict.
def _box_to_dict(box: BoundingBox) -> dict[str, float]:
    return {"x_min": box.x_min, "y_min": box.y_min, "x_max": box.x_max, "y_max": box.y_max}


# Reconstrói a BoundingBox a partir do dict — lado inverso de _box_to_dict.
def _box_from_dict(payload: dict[str, float]) -> BoundingBox:
    return BoundingBox(**payload)


# Converte a proveniência do modelo (stage, producer, checkpoint, etc.) em um
# dict serializável — helper interno reutilizado por claims e relations.
def _provenance_to_dict(provenance: ModelProvenance) -> dict[str, Any]:
    return {
        "stage": provenance.stage,
        "producer": provenance.producer,
        "config_fingerprint": provenance.config_fingerprint,
        "model_id": provenance.model_id,
        "checkpoint": provenance.checkpoint,
        "prompt_version": provenance.prompt_version,
    }


# Reconstrói a ModelProvenance a partir do dict — lado inverso de
# _provenance_to_dict.
def _provenance_from_dict(payload: dict[str, Any]) -> ModelProvenance:
    return ModelProvenance(**payload)


# Converte a Evidence (descrição + referência opcional de artifact) em um
# dict serializável, achatando o SourceArtifactReference quando presente.
def _evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    artifact = None if evidence.artifact is None else vars(evidence.artifact)
    return {"description": evidence.description, "artifact": artifact}


# Reconstrói a Evidence a partir do dict — lado inverso de _evidence_to_dict.
def _evidence_from_dict(payload: dict[str, Any]) -> Evidence:
    artifact = None if payload.get("artifact") is None else SourceArtifactReference(**payload["artifact"])
    return Evidence(description=payload["description"], artifact=artifact)


# Converte uma SemanticClaim completa (kind, value, confidence, evidence,
# provenance) em um dict serializável, delegando os sub-tipos aos helpers
# correspondentes.
def _claim_to_dict(claim: SemanticClaim) -> dict[str, Any]:
    return {
        "kind": claim.kind.value,
        "value": claim.value,
        "confidence": {"value": claim.confidence.value, "source": claim.confidence.source},
        "evidence": [_evidence_to_dict(item) for item in claim.evidence],
        "provenance": _provenance_to_dict(claim.provenance),
    }


# Reconstrói a SemanticClaim a partir do dict — lado inverso de
# _claim_to_dict.
def _claim_from_dict(payload: dict[str, Any]) -> SemanticClaim:
    return SemanticClaim(
        kind=ClaimKind(payload["kind"]),
        value=payload["value"],
        confidence=ConfidenceScore(**payload["confidence"]),
        evidence=tuple(_evidence_from_dict(item) for item in payload["evidence"]),
        provenance=_provenance_from_dict(payload["provenance"]),
    )


# Converte uma ObservedRegion completa (mask, box, claims, embedding refs
# etc.) em um dict serializável, delegando geometria e claims aos helpers
# correspondentes.
def _region_to_dict(region: ObservedRegion) -> dict[str, Any]:
    return {
        "region_id": region.region_id,
        "mask": _mask_to_dict(region.mask),
        "box": _box_to_dict(region.box),
        "geometric_confidence": region.geometric_confidence,
        "contributing_proposal_ids": list(region.contributing_proposal_ids),
        "claims": [_claim_to_dict(claim) for claim in region.claims],
        "visual_embedding_ref": region.visual_embedding_ref,
        "language_embedding_ref": region.language_embedding_ref,
    }


# Reconstrói a ObservedRegion a partir do dict, validando o region_id antes
# de montar o objeto — lado inverso de _region_to_dict.
def _region_from_dict(payload: dict[str, Any]) -> ObservedRegion:
    validate_identifier(payload["region_id"], field="region_id")
    return ObservedRegion(
        region_id=payload["region_id"],
        mask=_mask_from_dict(payload["mask"]),
        box=_box_from_dict(payload["box"]),
        geometric_confidence=payload["geometric_confidence"],
        contributing_proposal_ids=tuple(payload["contributing_proposal_ids"]),
        claims=tuple(_claim_from_dict(claim) for claim in payload["claims"]),
        visual_embedding_ref=payload["visual_embedding_ref"],
        language_embedding_ref=payload["language_embedding_ref"],
    )


# Converte uma CandidateRelation (par sujeito/objeto, predicado, confiança,
# proveniência) em um dict serializável.
def _relation_to_dict(relation: CandidateRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "subject_region_id": relation.subject_region_id,
        "predicate": relation.predicate,
        "object_region_id": relation.object_region_id,
        "confidence": {"value": relation.confidence.value, "source": relation.confidence.source},
        "source": relation.source.value,
        "evidence": [_evidence_to_dict(item) for item in relation.evidence],
        "provenance": _provenance_to_dict(relation.provenance),
    }


# Reconstrói a CandidateRelation a partir do dict — lado inverso de
# _relation_to_dict.
def _relation_from_dict(payload: dict[str, Any]) -> CandidateRelation:
    return CandidateRelation(
        relation_id=payload["relation_id"],
        subject_region_id=payload["subject_region_id"],
        predicate=payload["predicate"],
        object_region_id=payload["object_region_id"],
        confidence=ConfidenceScore(**payload["confidence"]),
        source=RelationSource(payload["source"]),
        evidence=tuple(_evidence_from_dict(item) for item in payload["evidence"]),
        provenance=_provenance_from_dict(payload["provenance"]),
    )
