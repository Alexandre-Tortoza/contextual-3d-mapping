"""Region-level semantic interpretation stage.

Issue: #165.

Interprets each region independently. A region's geometry (id, mask, box,
geometric confidence, contributing proposals) is never modified here: only
``claims`` is populated. A failure interpreting one region is isolated and
reported, and never invalidates the other regions.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from visual_perception.application.support import fingerprint_of
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner


def interpret_regions(
    regions: tuple[ObservedRegion, ...],
    image: ImagePayload,
    scene_context: SceneContext | None,
    reasoner: MultimodalReasoner,
    config: MultimodalReasoningConfig,
) -> tuple[tuple[ObservedRegion, ...], tuple[RegionInterpretationFailure, ...]]:
    """Interpret every region, isolating per-region failures."""
    scene_summary = _summarize_scene(scene_context)
    updated: list[ObservedRegion] = []
    failures: list[RegionInterpretationFailure] = []

    for region in regions:
        try:
            claims = _interpret_one_region(region, image, scene_summary, reasoner, config)
        except (KeyError, ValueError, TypeError) as error:
            failures.append(RegionInterpretationFailure(region.region_id, str(error)))
            updated.append(region)
            continue
        updated.append(dataclasses.replace(region, claims=region.claims + claims))

    return tuple(updated), tuple(failures)


def _interpret_one_region(
    region: ObservedRegion,
    image: ImagePayload,
    scene_summary: str | None,
    reasoner: MultimodalReasoner,
    config: MultimodalReasoningConfig,
) -> tuple[SemanticClaim, ...]:
    box = region.box
    crop = image.crop(int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max))
    response = reasoner.analyze_region(image, crop, scene_summary, config)
    _validate_region_response(response)

    provenance = ModelProvenance(
        stage="region_semantics",
        producer=config.backend,
        config_fingerprint=fingerprint_of(config),
        checkpoint=config.checkpoint,
        prompt_version=config.prompt_version,
    )
    evidence = (Evidence(description=f"raw multimodal region response for {region.region_id}"),)

    claims: list[SemanticClaim] = []
    for label in response["labels"]:
        value, confidence_value = _label_value_and_confidence(label)
        claims.append(
            SemanticClaim(
                ClaimKind.LABEL,
                value,
                ConfidenceScore(confidence_value, source=config.backend),
                evidence,
                provenance,
            )
        )
    default_confidence = ConfidenceScore(1.0, source=config.backend)
    if response.get("description"):
        claims.append(
            SemanticClaim(
                ClaimKind.ATTRIBUTE, str(response["description"]), default_confidence, evidence, provenance
            )
        )
    for attribute in response.get("attributes", []):
        claims.append(
            SemanticClaim(ClaimKind.ATTRIBUTE, str(attribute), default_confidence, evidence, provenance)
        )
    if response.get("condition"):
        claims.append(
            SemanticClaim(
                ClaimKind.CONDITION, str(response["condition"]), default_confidence, evidence, provenance
            )
        )
    if response.get("material"):
        claims.append(
            SemanticClaim(
                ClaimKind.MATERIAL, str(response["material"]), default_confidence, evidence, provenance
            )
        )
    return tuple(claims)


def _label_value_and_confidence(label: Any) -> tuple[str, float]:
    if isinstance(label, dict):
        if "value" not in label:
            raise ValueError("Malformed label hypothesis: missing 'value'.")
        return str(label["value"]), float(label.get("confidence", 1.0))
    if isinstance(label, str) and label:
        return label, 1.0
    raise ValueError(f"Malformed label hypothesis: {label!r}.")


def _validate_region_response(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise ValueError(f"Malformed region response: expected an object, got {type(response)!r}.")
    labels = response.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Malformed region response: 'labels' must be a non-empty list.")


def _summarize_scene(scene_context: SceneContext | None) -> str | None:
    if scene_context is None:
        return None
    descriptions = [claim.value for claim in scene_context.claims if claim.kind.value == "scene_description"]
    return descriptions[0] if descriptions else None
