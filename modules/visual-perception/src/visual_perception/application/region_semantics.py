"""Etapa de interpretação semântica em nível de region.

Issue: #165.

Interpreta cada region de forma independente. A geometria de uma region
(id, mask, box, geometric confidence, proposals contribuintes) nunca é
modificada aqui: apenas ``claims`` é populado. Uma falha ao interpretar
uma region é isolada e reportada, e nunca invalida as demais regions.

O parsing da resposta bruta do reasoner vive em
:func:`parse_region_interpretation`, uma fronteira pura e sem I/O. Ela
substitui o parser anterior, que atribuía confiança ``1.0`` sempre que o
modelo omitia o score — fazendo todo claim sair com confiança máxima
artificial e tornando impossível medir qualquer melhoria downstream.
Ausência de score agora é ``None``, nunca um número inventado.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from visual_perception.application.support import fingerprint_of
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner

#: Fonte default atribuída a um ConfidenceScore parseado quando o chamador não
#: informa qual backend produziu a resposta. O caller de produção
#: (``_interpret_one_region``) sempre passa ``config.backend``.
_DEFAULT_CONFIDENCE_SOURCE = "vlm"


# Sinaliza que a resposta bruta do reasoner não satisfaz o contrato de
# interpretação de região. Existe separada de RegionInterpretationFailure porque
# esta é uma falha de *schema*, pura e sem região associada — a outra representa
# a falha de uma região concreta e exige region_id + reason.
class InvalidInterpretation(ValueError):
    """A resposta bruta do reasoner não satisfaz o contrato de região."""


# Representa uma hipótese de label com sua confiança opcional. Existe para que
# "o produtor não forneceu score" seja representável (``confidence=None``) em vez
# de virar um 1.0 artificial, que foi a regressão que motivou esta fronteira.
@dataclass(frozen=True)
class LabelHypothesis:
    """Uma hipótese de label e a confiança que o produtor atribuiu a ela."""

    value: str
    confidence: ConfidenceScore | None


# Representa a resposta de região do reasoner já validada e normalizada, antes de
# ganhar evidência e proveniência. Existe como a fronteira pura entre o transporte
# (JSON bruto do VLM) e o modelo de claims auditáveis do domínio.
@dataclass(frozen=True)
class RegionInterpretation:
    """A interpretação de uma região, validada e independente de I/O."""

    category: str | None
    primary: LabelHypothesis
    kind: RegionKind
    alternatives: tuple[LabelHypothesis, ...]

    # Atalho para o valor do label primário, o campo mais consultado por quem
    # converte a interpretação em claims.
    @property
    def label(self) -> str:
        """O valor do label primário."""
        return self.primary.value

    # Atalho para a confiança do label primário, que pode legitimamente não
    # existir quando o produtor não pontuou a hipótese.
    @property
    def confidence(self) -> ConfidenceScore | None:
        """A confiança do label primário, ou ``None`` se não pontuada."""
        return self.primary.confidence


# Converte a resposta bruta do reasoner multimodal em uma RegionInterpretation
# validada. É a fronteira pura do estágio: determinística, sem I/O e sem
# conhecimento de ObservedRegion, ImagePayload, Evidence ou ModelProvenance, o que
# a torna testável sem mocks. Chamada por _interpret_one_region.
def parse_region_interpretation(
    response: Any, *, source: str = _DEFAULT_CONFIDENCE_SOURCE
) -> RegionInterpretation:
    """Valida e normaliza a resposta de região do reasoner multimodal.

    Uma resposta que seja apenas uma string é aceita como um label sem score,
    ``kind`` desconhecido e sem category — nunca como uma hipótese certa.

    Argumentos:
        response: resposta bruta do reasoner (objeto JSON já desserializado).
        source: identidade do produtor, anexada a cada ConfidenceScore.
    Retorna:
        a interpretação validada.
    Levanta:
        InvalidInterpretation: se a resposta não satisfizer o contrato.
    """
    if isinstance(response, str):
        if not response:
            raise InvalidInterpretation("Malformed region response: empty string.")
        return RegionInterpretation(
            category=None,
            primary=LabelHypothesis(response, None),
            kind=RegionKind.UNKNOWN,
            alternatives=(),
        )
    if not isinstance(response, dict):
        raise InvalidInterpretation(
            f"Malformed region response: expected an object or string, got {type(response)!r}."
        )
    if "labels" in response:
        raise InvalidInterpretation(
            "Malformed region response: the legacy 'labels' list is no longer accepted; "
            "expected the 'label'/'kind'/'confidence' contract."
        )

    label = response.get("label")
    if not isinstance(label, str) or not label:
        raise InvalidInterpretation(
            f"Malformed region response: 'label' must be a non-empty string, got {label!r}."
        )

    category = response.get("category")
    if category is not None and not isinstance(category, str):
        raise InvalidInterpretation(
            f"Malformed region response: 'category' must be a string or absent, got {category!r}."
        )

    return RegionInterpretation(
        category=category or None,
        primary=LabelHypothesis(label, _parse_confidence(response.get("confidence"), source)),
        kind=_parse_kind(response.get("kind")),
        alternatives=_parse_alternatives(response.get("alternatives"), source),
    )


# Converte o score bruto em ConfidenceScore, distinguindo ausência (None) de
# valor inválido (erro). Existe para concentrar num só lugar a regra que substitui
# o antigo fallback ``1.0``. Chamada por parse_region_interpretation.
def _parse_confidence(raw: Any, source: str) -> ConfidenceScore | None:
    """Converte um score bruto em ConfidenceScore, ou ``None`` quando ausente."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise InvalidInterpretation(
            f"Malformed region response: 'confidence' must be a number or absent, got {raw!r}."
        )
    try:
        return ConfidenceScore(float(raw), source=source)
    except ValueError as error:
        raise InvalidInterpretation(f"Malformed region response: {error}") from error


# Converte o kind bruto em RegionKind, mapeando ausência para UNKNOWN e um valor
# desconhecido para erro. Existe para que "não informado" nunca seja confundido
# com "informado como thing". Chamada por parse_region_interpretation.
def _parse_kind(raw: Any) -> RegionKind:
    """Converte um kind bruto em RegionKind; ausência vira ``UNKNOWN``."""
    if raw is None:
        return RegionKind.UNKNOWN
    try:
        return RegionKind(raw)
    except ValueError as error:
        raise InvalidInterpretation(
            f"Malformed region response: unrecognized 'kind' {raw!r}; "
            f"expected one of {sorted(kind.value for kind in RegionKind)}."
        ) from error


# Converte a lista bruta de alternativas em hipóteses validadas. Uma alternativa
# malformada invalida a resposta inteira em vez de ser descartada em silêncio.
# Chamada por parse_region_interpretation.
def _parse_alternatives(raw: Any, source: str) -> tuple[LabelHypothesis, ...]:
    """Converte a lista bruta de alternativas em hipóteses validadas."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise InvalidInterpretation(
            f"Malformed region response: 'alternatives' must be a list or absent, got {raw!r}."
        )
    hypotheses: list[LabelHypothesis] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise InvalidInterpretation(
                f"Malformed region response: each alternative must be an object, got {entry!r}."
            )
        value = entry.get("label")
        if not isinstance(value, str) or not value:
            raise InvalidInterpretation(
                f"Malformed region response: alternative 'label' must be a non-empty string, "
                f"got {value!r}."
            )
        hypotheses.append(LabelHypothesis(value, _parse_confidence(entry.get("confidence"), source)))
    return tuple(hypotheses)


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
    """Interpreta todas as regions, isolando falhas por region.

    Uma resposta que viole o contract (:class:`InvalidInterpretation`, subclasse
    de ``ValueError``) derruba apenas a region afetada: ela é preservada com a
    geometria e os claims que outros stages já anexaram, e a falha é reportada.
    """
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
    interpretation = parse_region_interpretation(response, source=config.backend)

    provenance = ModelProvenance(
        stage="region_semantics",
        producer=config.backend,
        config_fingerprint=fingerprint_of(config),
        checkpoint=config.checkpoint,
        prompt_version=config.prompt_version,
    )
    # A resposta literal do modelo acompanha a claim para que a calibração
    # (#196) e a auditoria (#199) possam voltar do score até o texto que o
    # originou, sem depender de logs externos.
    evidence = (
        Evidence(
            description=f"raw multimodal region response for {region.region_id}",
            raw_response_json=_raw_response_json(response),
        ),
    )

    # A hipótese primária e as alternativas viram claims de label irmãos: hipóteses
    # concorrentes coexistem, conforme o design "claims, não labels" (#156). Cada
    # uma carrega a confiança que o produtor informou — ou nenhuma.
    claims: list[SemanticClaim] = [
        SemanticClaim(ClaimKind.LABEL, hypothesis.value, hypothesis.confidence, evidence, provenance)
        for hypothesis in (interpretation.primary, *interpretation.alternatives)
    ]
    claims.extend(_descriptive_claims(response, evidence, provenance))
    return tuple(claims)


# Serializa a resposta bruta do reasoner quando ela é um objeto JSON.
# Existe porque o port permite uma resposta em string simples, que não
# satisfaz o contract de ``Evidence.raw_response_json`` (um objeto); nesse
# caso a resposta é envolvida em um objeto de um campo só, preservando o
# texto literal. Chamada por _interpret_one_region.
def _raw_response_json(response: Any) -> str | None:
    """Serializa a resposta bruta como objeto JSON, ou ``None`` se não for serializável."""
    payload = response if isinstance(response, dict) else {"response": response}
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


# Converte os campos descritivos livres da resposta (description, attributes,
# condition, material) em claims não pontuados. Existe separada da interpretação
# de label porque esses campos não têm contract validável nem consumidor hoje: são
# preservados como estavam, menos o score 1.0 artificial que o parser antigo
# atribuía a eles. Chamada por _interpret_one_region.
def _descriptive_claims(
    response: Any, evidence: tuple[Evidence, ...], provenance: ModelProvenance
) -> tuple[SemanticClaim, ...]:
    """Converte os campos descritivos livres da resposta em claims sem score."""
    if not isinstance(response, dict):
        return ()
    claims: list[SemanticClaim] = []
    if response.get("description"):
        claims.append(
            SemanticClaim(ClaimKind.ATTRIBUTE, str(response["description"]), None, evidence, provenance)
        )
    for attribute in response.get("attributes", []):
        claims.append(SemanticClaim(ClaimKind.ATTRIBUTE, str(attribute), None, evidence, provenance))
    if response.get("condition"):
        claims.append(
            SemanticClaim(ClaimKind.CONDITION, str(response["condition"]), None, evidence, provenance)
        )
    if response.get("material"):
        claims.append(
            SemanticClaim(ClaimKind.MATERIAL, str(response["material"]), None, evidence, provenance)
        )
    return tuple(claims)


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
