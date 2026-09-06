"""Etapa de análise contextual em nível de cena.

Issues: #164 (contexto de cena), #195 (proibição de confiança inventada).

Analisa a imagem completa para scene type, description, atributos globais
e hazards. Nunca toca na geometria de region: a enumeração de regions
permanece de posse de region discovery/merge.

Esta etapa atribuía ``ConfidenceScore(1.0)`` a *todo* claim de cena sempre
que o modelo omitia o score — inclusive a descrições livres e hazards, que
são exatamente os claims que a #195 proíbe de carregar confiança bruta.
Agora só ``scene_type`` recebe o score que o modelo de fato informou, e a
resposta bruta é preservada em ``Evidence.raw_response_json`` para que a
calibração (#196) possa pontuar os demais a partir de evidência.
"""

from __future__ import annotations

import json
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
    scene_type_confidence = _parse_scene_confidence(response.get("confidence"), config.backend)
    evidence = (
        Evidence(
            description="raw multimodal scene response",
            raw_response_json=json.dumps(response, sort_keys=True, default=str),
        ),
    )

    claims = [
        SemanticClaim(
            ClaimKind.SCENE_TYPE, str(response["scene_type"]), scene_type_confidence, evidence, provenance
        ),
        SemanticClaim(ClaimKind.SCENE_DESCRIPTION, str(response["description"]), None, evidence, provenance),
    ]
    for attribute in response.get("attributes", []):
        claims.append(SemanticClaim(ClaimKind.ATTRIBUTE, str(attribute), None, evidence, provenance))
    for hazard in response.get("hazards", []):
        claims.append(SemanticClaim(ClaimKind.HAZARD, str(hazard), None, evidence, provenance))

    return SceneContext(claims=tuple(claims))


# Converte o score de cena bruto em ConfidenceScore, mantendo ausência como
# ausência. Existe para que a regra que substituiu o antigo fallback ``1.0``
# fique em um único lugar testável; chamada por analyze_scene.
def _parse_scene_confidence(raw: Any, source: str) -> ConfidenceScore | None:
    """Converte o score de cena informado pelo modelo, ou ``None`` quando ausente.

    Argumentos:
        raw: valor bruto do campo ``confidence`` da resposta.
        source: identidade do backend que produziu a resposta.
    Retorna:
        o :class:`ConfidenceScore` informado, ou ``None``.
    Levanta:
        ValueError: se o campo existir mas não for um número em ``[0, 1]``.
    """
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"Malformed scene response: 'confidence' must be a number or absent, got {raw!r}.")
    return ConfidenceScore(float(raw), source=source)


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
