"""Etapa de análise contextual em nível de cena.

Issues: #164 (contexto de cena), #195 (proibição de confiança inventada),
#202 (contexto ambiental e isolamento do ego-veículo).

Descreve o **ambiente**: tipo de cena, se é interno ou externo, o arranjo
espacial, a iluminação, a visibilidade e a navegabilidade. Nunca toca na
geometria de region, e desde a #202 não produz mais inventário de objetos.

O motivo é medido. Em ``corridor-02-002``, o contract anterior — prosa livre
mais uma lista ``attributes`` — devolveu ``["fisheye lens", "carpeted floor",
"suitcase"]`` e "there is a suitcase on the ground in the foreground". A
"mala" era o próprio quad que carrega a câmera, e aquele texto ia inteiro para
o prompt de **cada** região, condicionando a interpretação local com um objeto
inexistente. Um objeto inferido globalmente não pode induzir a identidade de
uma região local.

A outra metade da correção é visual: quando a sequência declara geometria de
área, a cena é analisada sobre a área válida **menos** a área do ego. É um
recorte, nunca uma pintura: os pixels de origem permanecem intactos, conforme
a regra que a #212 estabeleceu depois de medir o custo de violá-la.

Esta etapa atribuía ``ConfidenceScore(1.0)`` a *todo* claim de cena sempre
que o modelo omitia o score. Agora só ``scene_type`` recebe o score que o
modelo de fato informou, e a resposta bruta é preservada em
``Evidence.raw_response_json`` para que a calibração (#196) possa pontuar os
demais a partir de evidência.
"""

from __future__ import annotations

import json
from typing import Any

from visual_perception.application.support import fingerprint_of
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner

#: Campos ambientais obrigatórios: sem eles não há contexto a propagar.
_REQUIRED_FIELDS = ("scene_type", "environment")

#: Campos ambientais opcionais e o kind de claim de cada um. São opcionais
#: porque o modelo pode legitimamente não saber avaliá-los num frame; a
#: ausência vira ausência de claim, nunca string vazia.
_ENVIRONMENTAL_FIELDS: tuple[tuple[str, ClaimKind], ...] = (
    ("environment", ClaimKind.ENVIRONMENT),
    ("layout", ClaimKind.LAYOUT),
    ("lighting", ClaimKind.LIGHTING),
    ("visibility", ClaimKind.VISIBILITY),
    ("navigability", ClaimKind.NAVIGABILITY),
)

#: Campos do contract anterior que agora são recusados. Aceitá-los em silêncio
#: reabriria o canal pelo qual um objeto alucinado entrava no contexto global.
_REJECTED_FIELDS = ("attributes", "hazards", "description")


# Ponto de entrada público: analisa a cena inteira via multimodal reasoner
# e converte a resposta bruta validada em um SceneContext com claims e
# proveniência. Chamada pelo pipeline principal antes da interpretação por
# region, para fornecer contexto de cena a etapas downstream.
def analyze_scene(
    image: ImagePayload,
    reasoner: MultimodalReasoner,
    config: MultimodalReasoningConfig,
    *,
    area_masks: ImageAreaMasks | None = None,
) -> SceneContext:
    """Produz um :class:`SceneContext` validado a partir de uma resposta multimodal bruta.

    Argumentos:
        image: o frame completo, que permanece inalterado.
        reasoner: o backend multimodal que descreve o ambiente.
        config: a configuração de raciocínio multimodal.
        area_masks: as áreas declaradas do frame. Quando presentes, a cena é
            analisada sobre a área válida menos a área do ego, para que o rig
            não entre no inventário do ambiente.
    Retorna:
        o contexto de cena validado.
    """
    response = reasoner.analyze_scene(_scene_view(image, area_masks), config)
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
        )
    ]
    for field, kind in _ENVIRONMENTAL_FIELDS:
        value = response.get(field)
        if value:
            claims.append(SemanticClaim(kind, str(value), None, evidence, provenance))

    return SceneContext(claims=tuple(claims))


# Recorta a imagem à parte que é cena analisável: dentro do sensor e fora do
# rig. Existe para que o rig deixe de ser inventariado como objeto do ambiente
# sem que nenhum pixel seja alterado — recortar preserva a evidência restante,
# pintar destruiria a que ficasse. Chamada por analyze_scene.
def _scene_view(image: ImagePayload, area_masks: ImageAreaMasks | None) -> ImagePayload:
    """Retorna a view de cena, recortada às áreas declaradas quando existirem."""
    if area_masks is None:
        return image
    box = area_masks.scene_box()
    if box is None:
        return image
    x_min, y_min = int(box.x_min), int(box.y_min)
    x_max, y_max = int(box.x_max), int(box.y_max)
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        return image
    return image.crop(x_min, y_min, x_max, y_max)


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


# Valida a forma da resposta bruta de cena antes de convertê-la em claims:
# campos ambientais obrigatórios presentes, opcionais bem tipados, e campos do
# contract antigo recusados em vez de ignorados. Chamada por analyze_scene.
def _validate_scene_response(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise ValueError(f"Malformed scene response: expected an object, got {type(response)!r}.")
    for required in _REQUIRED_FIELDS:
        value = response.get(required)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Malformed scene response: field {required!r} must be a non-empty string."
            )
    for rejected in _REJECTED_FIELDS:
        if rejected in response:
            raise ValueError(
                f"Malformed scene response: field {rejected!r} is no longer part of the scene "
                "contract. Scene context describes the environment, never an inventory of objects "
                "(see issue #202)."
            )
    for field, _ in _ENVIRONMENTAL_FIELDS:
        value = response.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Malformed scene response: field {field!r} must be a string or absent, got {value!r}."
            )
