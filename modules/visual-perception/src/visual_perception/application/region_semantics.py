"""Etapa de interpretação semântica em nível de region.

Issue: #165.

Interpreta cada region de forma independente. A geometria de uma region
(id, mask, box, geometric confidence, proposals contribuintes) nunca é
modificada aqui: apenas ``claims`` é populado. Uma falha ao interpretar
uma region é isolada e reportada, e nunca invalida as demais regions.
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


# Interpreta todas as regions de uma observation, isolando falhas por
# region (uma region que falha não derruba as demais). Existe como o ponto
# de entrada público desta etapa; usada pelo pipeline principal e por
# refine_observation (refinement.py) para reprocessar regions específicas.
def interpret_regions(
    regions: tuple[ObservedRegion, ...],
    image: ImagePayload,
    scene_context: SceneContext | None,
    reasoner: MultimodalReasoner,
    config: MultimodalReasoningConfig,
) -> tuple[tuple[ObservedRegion, ...], tuple[RegionInterpretationFailure, ...]]:
    """Interpreta todas as regions, isolando falhas por region."""
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


# Interpreta uma única region: recorta a imagem pelo box, consulta o
# multimodal reasoner, valida a resposta e converte os campos retornados
# (labels, description, attributes, condition, material) em SemanticClaim
# com proveniência (ModelProvenance). Chamada por interpret_regions para
# cada region, dentro do try/except que isola falhas.
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


# Normaliza uma hipótese de label bruta (dict com 'value'/'confidence' ou
# string simples) no par (valor, confiança) usado para construir um
# SemanticClaim. Existe para tolerar as duas formas que o backend de
# multimodal reasoning pode retornar. Chamada por _interpret_one_region.
def _label_value_and_confidence(label: Any) -> tuple[str, float]:
    if isinstance(label, dict):
        if "value" not in label:
            raise ValueError("Malformed label hypothesis: missing 'value'.")
        return str(label["value"]), float(label.get("confidence", 1.0))
    if isinstance(label, str) and label:
        return label, 1.0
    raise ValueError(f"Malformed label hypothesis: {label!r}.")


# Valida a forma mínima esperada da resposta bruta do reasoner (é um dict e
# tem uma lista 'labels' não vazia) antes de convertê-la em claims. Existe
# para falhar cedo com um erro acionável quando o backend retorna algo
# malformado. Chamada por _interpret_one_region.
def _validate_region_response(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise ValueError(f"Malformed region response: expected an object, got {type(response)!r}.")
    labels = response.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Malformed region response: 'labels' must be a non-empty list.")


# Extrai a primeira claim de descrição de cena ('scene_description') do
# SceneContext, se existir, para usar como resumo textual passado ao
# reasoner. Existe porque o reasoner de region se beneficia de contexto de
# cena, mas só precisa de um resumo curto, não do SceneContext inteiro.
# Chamada por interpret_regions.
def _summarize_scene(scene_context: SceneContext | None) -> str | None:
    if scene_context is None:
        return None
    descriptions = [claim.value for claim in scene_context.claims if claim.kind.value == "scene_description"]
    return descriptions[0] if descriptions else None
