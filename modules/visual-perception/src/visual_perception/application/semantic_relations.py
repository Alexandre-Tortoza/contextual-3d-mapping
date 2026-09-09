"""Inferência de relações semânticas candidatas entre regiões canônicas.

Issue: #206.

O caminho geométrico responde "estas duas máscaras se tocam, se sobrepõem, ou
uma contém a outra". Ele é exato e barato, e é **tudo** que o downstream
recebia: no run ``20260908T131207Z``, as 165 regiões produziram 726 relações,
todas geométricas, das quais 432 eram ``near``.

`part_of`, `attached_to`, `supported_by`, `inside`, `covers` e `occludes` —
as arestas que um scene graph realmente usa — não existiam. Esta etapa as
produz, e o desenho dela é governado por uma restrição dura: **custo**.

Perguntar ao VLM sobre todos os pares é O(n²). Num frame de 40 regiões são 780
chamadas para produzir, na esmagadora maioria, "nenhuma relação". A seleção de
pares é, portanto, parte do algoritmo e não um detalhe:

```text
todos os pares                      O(n²)      780 em corridor-02-000
  -> só os que a geometria priorizou            contenção, sobreposição, contato
  -> menos os pares dentro do mesmo grupo       a reconciliação já os descreve
  -> menos os pares de conceito idêntico        "parede encosta em parede"
  -> limitado por max_pairs                     orçamento explícito
```

A poda por prioridade geométrica antes de consultar o modelo é a mesma ideia
que o ConceptGraphs usa (IoU de caixas seguido de árvore geradora mínima); a
diferença é que a nossa poda é intra-frame e 2D, e a deles opera sobre objetos
já fundidos em 3D.

Nada aqui vira fato 3D. Toda relação produzida é ``MODEL_INFERRED``, candidata,
com a resposta bruta preservada, e nunca sobrescreve uma relação geométrica: as
duas fontes coexistem no mesmo grafo com proveniências distintas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from visual_perception.application.reconciliation import region_concept
from visual_perception.application.region_views import build_pair_view
from visual_perception.application.support import fingerprint_of
from visual_perception.config import SemanticRelationConfig
from visual_perception.domain.contextual_entities import ContextualEntityHypothesis
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_reasoning import RegionRelationRequest
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import (
    NO_RELATION_PREDICATE,
    SEMANTIC_RELATION_PREDICATES,
    CandidateRelation,
    RelationSource,
)
from visual_perception.domain.semantics import ConfidenceScore, Evidence, RegionKind
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner

__all__ = [
    "STAGE",
    "RelationCandidate",
    "RelationInferenceFailure",
    "RelationInferenceResult",
    "infer_semantic_relations",
    "parse_relation_response",
    "select_relation_candidates",
]

#: Nome do estágio em ``ModelProvenance``.
STAGE = "semantic_relations"


# Sinaliza que a resposta bruta do reasoner não satisfaz o contrato de relação.
# Existe pela mesma razão que ``InvalidInterpretation`` no estágio de região: é
# uma falha de *schema*, pura e sem par associado.
class InvalidRelation(ValueError):
    """A resposta bruta do reasoner não satisfaz o contrato de relação."""


# Registra a falha de um par específico. Existe para que a falha de um par não
# derrube os demais, no mesmo formato ``(resultados, falhas)`` que os outros
# estágios usam.
@dataclass(frozen=True)
class RelationInferenceFailure:
    """Uma falha isolada ao inferir a relação de um par de regiões."""

    subject_region_id: str
    object_region_id: str
    reason: str


# Descreve um par selecionado e a medida geométrica que o elegeu. Existe porque
# a mesma medida vai para duas coisas: a ordenação por prioridade e o texto que
# o modelo recebe — e recalculá-la nos dois lugares as deixaria divergir.
@dataclass(frozen=True)
class RelationCandidate:
    """Um par de regiões priorizado para inferência, com a medida que o elegeu."""

    subject: ObservedRegion
    target: ObservedRegion
    containment: float
    overlap: float
    touching: bool

    # Ordena candidatos: contenção primeiro, depois sobreposição, depois
    # contato, e o id como desempate para manter a seleção determinística.
    @property
    def priority(self) -> tuple[float, float, int, str, str]:
        """A chave determinística de ordenação sob orçamento."""
        return (
            -self.containment,
            -self.overlap,
            0 if self.touching else 1,
            self.subject.region_id,
            self.target.region_id,
        )

    # Resume, em texto, o que a geometria já mediu sobre o par. Existe para que
    # o modelo não precise reestimar no olho uma grandeza que já calculamos
    # exatamente — e para que o valor que ele recebe seja o mesmo que o
    # artifact registra.
    def geometric_summary(self) -> str:
        """Descreve a relação geométrica medida entre as duas regiões."""
        parts = [
            f"the first region's mask covers {self.containment:.0%} of the second region's mask",
            f"the two masks overlap with IoU {self.overlap:.2f}",
        ]
        parts.append("their bounding boxes touch" if self.touching else "their bounding boxes are apart")
        return "; ".join(parts)


# Agrupa o resultado da etapa: as relações inferidas, as falhas isoladas e
# quantos pares foram efetivamente consultados. A contagem existe para que o
# custo deste estágio seja auditável no manifest.
@dataclass(frozen=True)
class RelationInferenceResult:
    """As relações semânticas inferidas, as falhas isoladas e o custo medido."""

    relations: tuple[CandidateRelation, ...] = ()
    failures: tuple[RelationInferenceFailure, ...] = ()
    model_calls: int = 0
    candidates_considered: int = 0


# Seleciona, de forma determinística, os pares que valem uma chamada de modelo.
# É pura e sem I/O, o que permite testar a política de orçamento sem reasoner.
# Chamada por infer_semantic_relations.
def select_relation_candidates(
    regions: tuple[ObservedRegion, ...],
    config: SemanticRelationConfig,
    *,
    entities: tuple[ContextualEntityHypothesis, ...] = (),
    adjacency_margin_px: float = 5.0,
) -> tuple[RelationCandidate, ...]:
    """Prioriza os pares de regiões cuja geometria sugere uma relação real.

    Argumentos:
        regions: as regiões canônicas do frame, já reconciliadas.
        config: o orçamento e os limiares de candidatura.
        entities: os grupos de mesma superfície; pares internos a um grupo são
            descartados, porque a reconciliação já descreve melhor a relação
            entre eles do que uma aresta descreveria.
        adjacency_margin_px: distância máxima entre caixas para "em contato".
    Retorna:
        os candidatos ordenados por prioridade, limitados a ``max_pairs``.
    """
    grouped: set[frozenset[str]] = {
        frozenset({first, second})
        for entity in entities
        for index, first in enumerate(entity.member_region_ids)
        for second in entity.member_region_ids[index + 1 :]
    }
    candidates: list[RelationCandidate] = []
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            if frozenset({first.region_id, second.region_id}) in grouped:
                continue
            if config.require_distinct_concepts and region_concept(first) == region_concept(second):
                continue
            forward = first.mask.containment_ratio(second.mask)
            backward = second.mask.containment_ratio(first.mask)
            overlap = first.mask.iou(second.mask)
            touching = overlap > 0.0 or _boxes_touch(first, second, adjacency_margin_px)
            # A direção do par não é arbitrária: quem contém mais vira sujeito,
            # para que "A contém B" e "B está dentro de A" não virem duas
            # perguntas diferentes sobre a mesma evidência.
            subject, target, containment = (
                (first, second, forward) if forward >= backward else (second, first, backward)
            )
            if containment < config.min_containment and not (config.include_adjacent and touching):
                continue
            candidates.append(
                RelationCandidate(
                    subject=subject,
                    target=target,
                    containment=containment,
                    overlap=overlap,
                    touching=touching,
                )
            )
    candidates.sort(key=lambda candidate: candidate.priority)
    return tuple(candidates[: config.max_pairs])


# Valida e normaliza a resposta bruta do reasoner sobre um par. É a fronteira
# pura do estágio: determinística, sem I/O e sem conhecimento de região,
# imagem ou proveniência — o mesmo desenho de ``parse_region_interpretation``.
def parse_relation_response(response: Any) -> tuple[str, float | None]:
    """Valida a resposta de relação e devolve ``(predicado, confiança)``.

    Um predicado ``none`` é uma resposta **válida**: ele declara que a evidência
    não sustenta nenhuma aresta do vocabulário. Ele é devolvido como tal, e o
    chamador simplesmente não cria relação — o que é diferente de uma falha.

    Argumentos:
        response: resposta bruta do reasoner, já desserializada.
    Retorna:
        o predicado normalizado e a confiança informada, ou ``None``.
    Levanta:
        InvalidRelation: se a resposta não satisfizer o contract.
    """
    if not isinstance(response, dict):
        raise InvalidRelation(
            f"Malformed relation response: expected an object, got {type(response)!r}."
        )
    predicate = response.get("predicate")
    if not isinstance(predicate, str) or not predicate:
        raise InvalidRelation(
            f"Malformed relation response: 'predicate' must be a non-empty string, got {predicate!r}."
        )
    normalized = predicate.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized != NO_RELATION_PREDICATE and normalized not in SEMANTIC_RELATION_PREDICATES:
        raise InvalidRelation(
            f"Malformed relation response: unknown predicate {predicate!r}; expected one of "
            f"{sorted(SEMANTIC_RELATION_PREDICATES | {NO_RELATION_PREDICATE})}."
        )
    raw = response.get("confidence")
    if raw is None:
        return normalized, None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise InvalidRelation(
            f"Malformed relation response: 'confidence' must be a number or absent, got {raw!r}."
        )
    if not 0.0 <= float(raw) <= 1.0:
        raise InvalidRelation(
            f"Malformed relation response: 'confidence' must be in [0, 1], got {raw!r}."
        )
    return normalized, float(raw)


# Ponto de entrada público da etapa: consulta o reasoner sobre os pares
# priorizados e converte as respostas em relações candidatas. Chamada pelo
# pipeline canônico depois da reconciliação, para que os conceitos usados nas
# perguntas já sejam os reconciliados.
def infer_semantic_relations(
    regions: tuple[ObservedRegion, ...],
    image: ImagePayload,
    reasoner: MultimodalReasoner,
    config: SemanticRelationConfig,
    reasoning_config: Any,
    *,
    entities: tuple[ContextualEntityHypothesis, ...] = (),
    context_expansion: float = 0.25,
) -> RelationInferenceResult:
    """Infere relações semânticas candidatas entre os pares priorizados do frame.

    Argumentos:
        regions: as regiões canônicas já reconciliadas.
        image: o payload da imagem completa, de onde a view de par é recortada.
        reasoner: o backend multimodal consultado por par.
        config: orçamento e limiares de candidatura.
        reasoning_config: a configuração de raciocínio multimodal, repassada ao
            port e usada na proveniência.
        entities: os grupos de mesma superfície, para excluir pares internos.
        context_expansion: margem aplicada à caixa da união do par.
    Retorna:
        as relações inferidas, as falhas isoladas e o custo medido.
    """
    if not config.enabled or config.max_pairs <= 0 or len(regions) < 2:
        return RelationInferenceResult()

    candidates = select_relation_candidates(regions, config, entities=entities)
    provenance = ModelProvenance(
        stage=STAGE,
        producer=reasoning_config.backend,
        config_fingerprint=fingerprint_of(config),
        checkpoint=reasoning_config.checkpoint,
        prompt_version=reasoning_config.prompt_version,
    )

    relations: list[CandidateRelation] = []
    failures: list[RelationInferenceFailure] = []
    calls = 0
    for candidate in candidates:
        try:
            view = build_pair_view(
                candidate.subject, candidate.target, image, expansion=context_expansion
            )
            request = RegionRelationRequest(
                subject_region_id=candidate.subject.region_id,
                object_region_id=candidate.target.region_id,
                subject_concept=region_concept(candidate.subject) or "unlabelled region",
                object_concept=region_concept(candidate.target) or "unlabelled region",
                subject_kind=_kind_of(candidate.subject),
                object_kind=_kind_of(candidate.target),
                views=(view,),
                geometric_summary=candidate.geometric_summary(),
            )
            calls += 1
            response = reasoner.analyze_relation(request, reasoning_config)
            predicate, confidence = parse_relation_response(response)
        except (ValueError, TypeError, KeyError) as error:
            failures.append(
                RelationInferenceFailure(
                    candidate.subject.region_id, candidate.target.region_id, str(error)
                )
            )
            continue
        if predicate == NO_RELATION_PREDICATE:
            continue
        relations.append(
            CandidateRelation(
                relation_id=(
                    f"rel-semantic-{predicate}-{candidate.subject.region_id}-{candidate.target.region_id}"
                ),
                subject_region_id=candidate.subject.region_id,
                predicate=predicate,
                object_region_id=candidate.target.region_id,
                confidence=(
                    None
                    if confidence is None
                    else ConfidenceScore(confidence, source=reasoning_config.backend)
                ),
                source=RelationSource.MODEL_INFERRED,
                evidence=(
                    Evidence(
                        description=(
                            f"pair view of {candidate.subject.region_id} (green outline) and "
                            f"{candidate.target.region_id} (blue outline); {candidate.geometric_summary()}"
                        ),
                        raw_response_json=_raw_relation_json(response),
                    ),
                ),
                provenance=provenance,
            )
        )

    return RelationInferenceResult(
        relations=tuple(relations),
        failures=tuple(failures),
        model_calls=calls,
        candidates_considered=len(candidates),
    )


# Lê a natureza reconciliada de uma região como string para o request. Existe
# porque o request atravessa a fronteira do port, e ali um enum do domínio
# seria acoplamento desnecessário do adapter ao domínio.
def _kind_of(region: ObservedRegion) -> str:
    """Retorna o ``RegionKind`` reconciliado da região como string."""
    from visual_perception.application.reconciliation import reconciled_label_claim

    claim = reconciled_label_claim(region)
    if claim is None or claim.region_kind is None:
        return RegionKind.UNKNOWN.value
    return claim.region_kind.value


# Responde se duas caixas estão a até ``margin_px`` uma da outra.
def _boxes_touch(first: ObservedRegion, second: ObservedRegion, margin_px: float) -> bool:
    """Indica se as caixas das duas regiões estão encostadas dentro da margem."""
    a, b = first.box, second.box
    gap_x = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    gap_y = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    return max(gap_x, gap_y) <= margin_px


# Serializa a resposta bruta do reasoner para acompanhar a relação. Espelha
# ``_raw_response_json`` do estágio de região: a evidência de uma relação
# precisa poder ser auditada até o texto que a originou.
def _raw_relation_json(response: Any) -> str | None:
    """Serializa a resposta bruta como objeto JSON, ou ``None`` se não for serializável."""
    payload = response if isinstance(response, dict) else {"response": response}
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
