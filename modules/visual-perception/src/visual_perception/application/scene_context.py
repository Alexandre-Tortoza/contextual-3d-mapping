"""Scene-level contextual analysis stage.

Issue: #164.

Analyzes the full image for scene type, description, global attributes, and
hazards. Never touches region geometry: region enumeration stays owned by
region discovery/merge.
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


def analyze_scene(
    image: ImagePayload,
    reasoner: MultimodalReasoner,
    config: MultimodalReasoningConfig,
) -> SceneContext:
    """Produce a validated :class:`SceneContext` from a raw multimodal response."""
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
