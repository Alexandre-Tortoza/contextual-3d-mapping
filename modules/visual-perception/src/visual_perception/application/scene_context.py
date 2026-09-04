"""Etapa de análise contextual em nível de cena.

Issue: #164.

Analisa a imagem completa para scene type, description, atributos globais
e hazards. Nunca toca na geometria de region: a enumeração de regions
permanece de posse de region discovery/merge.
"""

from __future__ import annotations

from typing import Any

from visual_perception.application.support import fingerprint_of
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner

_REQUIRED_FIELDS = ("scene_type", "description")


# Ponto de entrada público: analisa a cena inteira via multimodal reasoner
# e converte a resposta bruta validada em um SceneContext com claims e
# proveniência. Chamada pelo pipeline principal antes da interpretação por
# region, para fornecer contexto de cena a etapas downstream.
def analyze_scene(
    image: ImagePayload,
    reasoner: MultimodalReasoner,
    config: MultimodalReasoningConfig,
) -> SceneContext:
    """Produz um :class:`SceneContext` validado a partir de uma resposta multimodal bruta."""
    response = reasoner.analyze_scene(image, config)
    _validate_scene_response(response)

    provenance = ModelProvenance(
        stage="scene_context",
        producer=config.backend,
        config_fingerprint=fingerprint_of(config),
        checkpoint=config.checkpoint,
        prompt_version=config.prompt_version,
    )
    confidence = ConfidenceScore(float(response.get("confidence", 1.0)), source=config.backend)
    evidence = (Evidence(description="raw multimodal scene response"),)

    claims = [
        SemanticClaim(ClaimKind.SCENE_TYPE, str(response["scene_type"]), confidence, evidence, provenance),
        SemanticClaim(
            ClaimKind.SCENE_DESCRIPTION, str(response["description"]), confidence, evidence, provenance
        ),
    ]
    for attribute in response.get("attributes", []):
        claims.append(SemanticClaim(ClaimKind.ATTRIBUTE, str(attribute), confidence, evidence, provenance))
    for hazard in response.get("hazards", []):
        claims.append(SemanticClaim(ClaimKind.HAZARD, str(hazard), confidence, evidence, provenance))

    return SceneContext(claims=tuple(claims))


# Valida a forma mínima da resposta bruta de cena (campos obrigatórios
# scene_type/description como strings não vazias, listas opcionais bem
# tipadas) antes de convertê-la em claims. Chamada por analyze_scene.
def _validate_scene_response(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise ValueError(f"Malformed scene response: expected an object, got {type(response)!r}.")
    for required in _REQUIRED_FIELDS:
        value = response.get(required)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Malformed scene response: field {required!r} must be a non-empty string."
            )
    for list_field in ("attributes", "hazards"):
        if list_field in response and not isinstance(response[list_field], list):
            raise ValueError(f"Malformed scene response: field {list_field!r} must be a list.")
