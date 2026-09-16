"""Descoberta de conceitos concretos em nível de cena para o grounding (#277).

Transforma a resposta bruta do VLM em uma lista curta e auditável de frases nominais.
Os sinais contextuais (dano, obstrução, placa) vêm antes das entidades no teto, e
estruturas genéricas sozinhas (``wall``, ``floor``) são descartadas com motivo.
"""

from __future__ import annotations

import json
import re
from typing import Any

from visual_perception.application.scene_context import scene_view
from visual_perception.application.support import fingerprint_of
from visual_perception.config import MultimodalReasoningConfig, SceneConceptDiscoveryConfig
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.scene_concepts import SceneConcept, SceneConceptKind, SceneConceptSet
from visual_perception.ports.scene_concept_discovery import SceneConceptDiscoverer

#: Frases mais longas que isso deixam de ser um conceito que o grounding localiza.
MAX_CONCEPT_WORDS = 5

_RESPONSE_KEYS = (
    ("contextual_features", SceneConceptKind.CONTEXTUAL_FEATURE),
    ("entities", SceneConceptKind.ENTITY),
)


# Consulta o backend sobre a mesma view de cena do contexto ambiental (sem rig e
# dentro do sensor) e devolve o conjunto de conceitos validado.
def discover_scene_concepts(
    image: ImagePayload,
    discoverer: SceneConceptDiscoverer,
    reasoning_config: MultimodalReasoningConfig,
    config: SceneConceptDiscoveryConfig,
    *,
    area_masks: ImageAreaMasks | None = None,
) -> SceneConceptSet:
    """Descobre conceitos concretos a procurar no frame.

    Argumentos:
        image: frame completo, inalterado.
        discoverer: backend que propõe os conceitos.
        reasoning_config: modelo e transporte do VLM usado.
        config: teto, exclusões e versão do prompt.
        area_masks: áreas declaradas do frame; recortam a view como na cena.
    Retorna:
        conceitos mantidos, descartes com motivo, resposta bruta e proveniência.
    """
    response = discoverer.discover_concepts(
        scene_view(image, area_masks), reasoning_config, max_concepts=config.max_concepts
    )
    concepts, discarded = parse_scene_concepts(response, config)
    provenance = ModelProvenance(
        stage="scene_concept_discovery",
        producer=reasoning_config.backend,
        config_fingerprint=fingerprint_of(config)[:8] + fingerprint_of(reasoning_config)[:8],
        checkpoint=reasoning_config.checkpoint,
        prompt_version=config.prompt_version,
    )
    return SceneConceptSet(
        concepts=concepts,
        discarded=discarded,
        raw_response_json=json.dumps(response, sort_keys=True, default=str),
        provenance=provenance,
    )


# Normaliza, filtra e limita os conceitos da resposta. Pública para ser testada sem
# backend e reaproveitada por benchmarks com respostas gravadas.
def parse_scene_concepts(
    response: Any, config: SceneConceptDiscoveryConfig
) -> tuple[tuple[SceneConcept, ...], tuple[tuple[str, str], ...]]:
    """Converte a resposta bruta em conceitos mantidos e descartes com motivo.

    Argumentos:
        response: objeto JSON devolvido pelo backend.
        config: teto e exclusões estruturais.
    Retorna:
        ``(conceitos, descartes)``, em que cada descarte é ``(texto, motivo)``.
    """
    if not isinstance(response, dict):
        return (), (("<response>", "malformed_response"),)
    kept: list[SceneConcept] = []
    discarded: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, kind in _RESPONSE_KEYS:
        items = response.get(key, [])
        if not isinstance(items, list):
            discarded.append((f"<{key}>", "malformed_list"))
            continue
        for item in items:
            text, confidence = _concept_fields(item)
            if text is None:
                discarded.append((str(item)[:60], "malformed_item"))
                continue
            normalized = _normalize(text)
            reason = _rejection_reason(normalized, seen, config)
            if reason is None and len(kept) >= config.max_concepts:
                reason = "over_budget"
            if reason is not None:
                discarded.append((normalized or text, reason))
                continue
            seen.add(normalized)
            kept.append(SceneConcept(normalized, kind, confidence))
    return tuple(kept), tuple(discarded)


# Aceita um item como string ou objeto com ``concept`` e ``confidence`` opcional.
def _concept_fields(item: Any) -> tuple[str | None, float | None]:
    """Extrai texto e confiança de um item, ou ``(None, None)`` se malformado."""
    if isinstance(item, str):
        return item, None
    if not isinstance(item, dict) or not isinstance(item.get("concept"), str):
        return None, None
    raw = item.get("confidence")
    if isinstance(raw, int | float) and not isinstance(raw, bool) and 0.0 <= raw <= 1.0:
        return item["concept"], float(raw)
    return item["concept"], None


# Deixa o conceito comparável: minúsculas, espaços simples, sem pontuação nas pontas.
def _normalize(text: str) -> str:
    """Retorna o conceito normalizado."""
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .,;:!?\"'")


# Decide se um conceito normalizado deve ser descartado, e por quê.
def _rejection_reason(normalized: str, seen: set[str], config: SceneConceptDiscoveryConfig) -> str | None:
    """Retorna o motivo de descarte, ou ``None`` quando o conceito é mantido."""
    if not normalized:
        return "empty"
    if len(normalized.split()) > MAX_CONCEPT_WORDS:
        return "not_a_noun_phrase"
    if normalized in config.structural_exclusions:
        return "structural"
    if normalized in seen:
        return "duplicate"
    return None
