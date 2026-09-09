"""Reconciliação contextual intra-frame.

Issue: #205.

Este é o primeiro estágio do módulo que olha **todas as regiões do frame ao
mesmo tempo**. Todos os anteriores decidem região a região, e há perguntas que
uma região isolada não consegue responder:

- ``wall``, ``plain wall`` e ``wall`` em três máscaras que se tocam são três
  coisas ou uma? Medido no run ``20260908T131207Z``: 108 das 165 regiões (65%)
  pertencem a um grupo assim, e o maior tem 13 membros;
- uma região que se diz ``wall`` e ``thing`` está afirmando duas coisas
  incompatíveis. Medido: 121 das 165 regiões (73,3%).

O que este estágio produz, e o que ele deliberadamente **não** produz:

```text
produz                                  não produz
------                                  ----------
claim de identidade RECONCILED          sobrescrita da claim PRIMARY
conceito canônico ao lado do label cru  taxonomia fechada de labels
grupo de mesma superfície (hipótese)    remoção de região, merge de máscara
veredito estrutural registrado          tabela label -> kind aplicada em silêncio
```

Duas regras de método governam o desenho, e as duas vêm de medição:

1. **o agrupamento não é dirigido por similaridade de feature.** Entre pares de
   regiões adjacentes, o cosseno DINOv2 separa "mesmo label" de "label
   diferente" com acurácia balanceada de apenas 0,660 no melhor limiar
   possível. Um gate por similaridade erraria um terço das decisões. O grupo é
   proposto por **conceito compatível mais contato espacial**, e a coerência
   densa entra como corroboração que pode rebaixar o grupo para ``unresolved``;
2. **a canonicalização é lexical e mínima.** Ela colapsa plural e um punhado de
   modificadores não discriminativos (``plain wall`` → ``wall``), e para aí.
   Ela não sabe que ``ceiling tiles`` e ``ceiling`` têm relação, e não deve
   saber: essa é uma afirmação sobre o mundo, não sobre a grafia, e o lugar
   dela é uma relação ``part_of`` sustentada por evidência (#206).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from visual_perception.application.support import fingerprint_of
from visual_perception.config import ReconciliationConfig
from visual_perception.domain.contextual_entities import (
    ContextualEntityHypothesis,
    EntityHypothesisKind,
    EntityHypothesisStatus,
    derive_entity_id,
)
from visual_perception.domain.embeddings import EmbeddingSpace, VisualEmbedding
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantics import (
    ClaimKind,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
    normalize_claim_value,
)
from visual_perception.domain.structural_consistency import (
    StructuralVerdict,
    expected_region_kind,
    head_noun,
    region_kind_verdict,
    singularize,
)

#: Nome do estágio e do produtor em ``ModelProvenance``. Uma claim reconciliada
#: nunca se apresenta como se o VLM a tivesse escrito.
STAGE = "intra_frame_reconciliation"
PRODUCER = "intra_frame_reconciliation"

#: Modificadores que descrevem a aparência do sujeito sem mudar a identidade
#: dele. ``plain wall`` e ``wall`` são a mesma coisa: o ``plain`` é um atributo,
#: e o próprio contract já tem um lugar para atributos. A lista é curta e
#: explícita de propósito — ela existe para colapsar variação de grafia, não
#: para construir uma ontologia, e o label cru continua preservado em qualquer
#: caso.
NON_DISCRIMINATIVE_MODIFIERS = frozenset({"blank", "flat", "plain", "simple"})


# Registra o que a reconciliação decidiu sobre uma região, para que a decisão
# seja auditável sem reabrir os claims. Existe porque "por que esta região
# ganhou uma claim reconciliada?" precisa ter resposta no artifact, e não só no
# código que a produziu.
@dataclass(frozen=True)
class ReconciliationRecord:
    """O que a reconciliação afirmou sobre uma região, e com base em quê."""

    region_id: str
    raw_label: str
    canonical_concept: str
    declared_kind: RegionKind
    reconciled_kind: RegionKind
    structural_verdict: StructuralVerdict
    entity_id: str | None = None


# Agrupa o resultado do estágio: as regiões com a claim reconciliada anexada,
# os grupos propostos e o registro por região. Existe para que o pipeline
# receba um objeto só, no mesmo formato dos demais estágios.
@dataclass(frozen=True)
class ReconciliationResult:
    """As regiões reconciliadas, os grupos propostos e o registro das decisões."""

    regions: tuple[ObservedRegion, ...]
    entities: tuple[ContextualEntityHypothesis, ...] = ()
    records: tuple[ReconciliationRecord, ...] = ()


# Reduz um conceito à sua forma canônica por normalização lexical mínima.
# Existe como função pública porque três consumidores precisam concordar
# exatamente sobre "estes dois labels são o mesmo conceito": a formação de
# grupos, a claim reconciliada e a seleção de pares para relações semânticas.
def canonical_concept(label: str) -> str:
    """Retorna a forma canônica de ``label``: caixa, plural e modificadores vazios.

    A transformação é deliberadamente pobre. Ela remove vírgulas, colapsa
    espaços, singulariza cada palavra e descarta modificadores que não
    discriminam identidade. Ela **não** consulta ontologia, não usa embedding e
    não conhece sinônimos: qualquer uma dessas coisas transformaria um sistema
    open-vocabulary em uma taxonomia fechada por dentro.

    Argumentos:
        label: o label cru, exatamente como o produtor o escreveu.
    Retorna:
        o conceito canônico, ou o label normalizado quando não há o que reduzir.
    """
    tokens = [token for token in normalize_claim_value(label).replace(",", " ").split() if token]
    reduced = [singularize(token) for token in tokens]
    meaningful = [token for token in reduced if token not in NON_DISCRIMINATIVE_MODIFIERS]
    return " ".join(meaningful or reduced)


# Ponto de entrada público do estágio: reconcilia identidade e natureza entre
# todas as regiões do frame e propõe grupos de mesma superfície. Chamada pelo
# pipeline canônico depois do refinamento e antes das relações semânticas, que
# consomem os conceitos reconciliados e os grupos.
def reconcile_observation(
    regions: tuple[ObservedRegion, ...],
    observation_id: str,
    config: ReconciliationConfig,
    *,
    visual_embeddings: tuple[VisualEmbedding, ...] = (),
    visual_space: EmbeddingSpace | None = None,
) -> ReconciliationResult:
    """Reconcilia as hipóteses de um frame sem alterar nenhuma geometria.

    Argumentos:
        regions: as regiões já interpretadas, calibradas e eventualmente
            refinadas.
        observation_id: identidade da observação, usada para derivar ids de
            grupo estáveis.
        config: limiares de adjacência e de corroboração.
        visual_embeddings: os embeddings densos por região, usados apenas como
            corroboração da coerência de um grupo.
        visual_space: o espaço em que essa coerência é medida. Obrigatório
            quando há embeddings: uma similaridade sem espaço declarado não é
            comparável a nada.
    Retorna:
        as regiões com a claim reconciliada anexada, os grupos e os registros.
    """
    if not config.enabled or not regions:
        return ReconciliationResult(regions=regions)

    fingerprint = fingerprint_of(config)
    provenance = ModelProvenance(
        stage=STAGE, producer=PRODUCER, config_fingerprint=fingerprint
    )

    decisions = {
        region.region_id: _decide(region, config) for region in regions
    }
    groups = _same_surface_groups(regions, decisions, config)
    entity_by_region = {
        member: entity_id for entity_id, members in groups for member in members
    }

    updated: list[ObservedRegion] = []
    records: list[ReconciliationRecord] = []
    for region in regions:
        decision = decisions[region.region_id]
        if decision is None:
            updated.append(region)
            continue
        entity_id = entity_by_region.get(region.region_id)
        records.append(dataclasses.replace(decision.record, entity_id=entity_id))
        if not decision.changes_anything:
            updated.append(region)
            continue
        updated.append(
            dataclasses.replace(
                region, claims=region.claims + (_reconciled_claim(decision, provenance, entity_id),)
            )
        )

    entities = _build_entities(
        groups, decisions, observation_id, provenance, config, visual_embeddings, visual_space
    )
    return ReconciliationResult(regions=tuple(updated), entities=entities, records=tuple(records))


# Reúne, para uma região, tudo que a reconciliação concluiu sobre ela. Existe
# como estrutura intermediária para que a formação de grupos e a construção da
# claim leiam a mesma decisão, em vez de recalculá-la de formas que podem
# divergir.
@dataclass(frozen=True)
class _Decision:
    """O que a reconciliação concluiu sobre uma única região."""

    claim: SemanticClaim
    record: ReconciliationRecord

    # Responde se há algo a acrescentar. Uma região cujo conceito já é canônico
    # e cuja natureza já é coerente não ganha claim nova: acrescentar uma cópia
    # do que já existe seria ruído no artifact.
    @property
    def changes_anything(self) -> bool:
        """Indica se a reconciliação encontrou algo que o produtor não afirmou."""
        return (
            self.record.canonical_concept != normalize_claim_value(self.record.raw_label)
            or self.record.reconciled_kind is not self.record.declared_kind
        )


# Decide o conceito canônico e a natureza reconciliada de uma região. Helper de
# reconcile_observation; devolve ``None`` quando a região não tem hipótese
# primária, caso em que não há o que reconciliar.
def _decide(region: ObservedRegion, config: ReconciliationConfig) -> _Decision | None:
    """Deriva conceito canônico e natureza reconciliada de uma região."""
    claim = primary_label_claim(region)
    if claim is None:
        return None
    concept = (
        canonical_concept(claim.value) if config.canonicalize_labels else normalize_claim_value(claim.value)
    )
    declared = claim.region_kind or RegionKind.UNKNOWN
    verdict = region_kind_verdict(claim)
    expected = expected_region_kind(claim)
    if verdict is StructuralVerdict.CONTRADICTS and expected is not None or declared is RegionKind.UNKNOWN and expected is not None:
        reconciled_kind = expected
    else:
        reconciled_kind = declared
    return _Decision(
        claim=claim,
        record=ReconciliationRecord(
            region_id=region.region_id,
            raw_label=claim.value,
            canonical_concept=concept,
            declared_kind=declared,
            reconciled_kind=reconciled_kind,
            structural_verdict=verdict,
        ),
    )


# Constrói a claim reconciliada de uma região. Ela é um claim **novo**, com
# produtor próprio, e a sua evidência nomeia exatamente o que a produziu: o
# label cru de origem, o veredito estrutural, e o grupo a que a região pertence.
def _reconciled_claim(
    decision: _Decision, provenance: ModelProvenance, entity_id: str | None
) -> SemanticClaim:
    """Constrói a claim de identidade reconciliada, sem tocar na original."""
    record = decision.record
    parts = [f"reconciled from raw label {record.raw_label!r}"]
    if record.canonical_concept != normalize_claim_value(record.raw_label):
        parts.append(f"canonical concept {record.canonical_concept!r}")
    if record.reconciled_kind is not record.declared_kind:
        parts.append(
            f"structural evidence rules {record.declared_kind.value!r} out for this concept; "
            f"reconciled to {record.reconciled_kind.value!r}"
        )
    if entity_id is not None:
        parts.append(f"member of contextual entity {entity_id}")
    return SemanticClaim(
        ClaimKind.LABEL,
        record.canonical_concept,
        None,
        (Evidence(description="; ".join(parts)),),
        provenance,
        role=HypothesisRole.RECONCILED,
        category=decision.claim.category,
        region_kind=record.reconciled_kind,
    )


# Agrupa regiões que se tocam e compartilham o conceito reconciliado, por
# union-find determinístico. Helper de reconcile_observation; devolve pares
# ``(entity_id_placeholder, membros)`` ainda sem corroboração densa.
def _same_surface_groups(
    regions: tuple[ObservedRegion, ...],
    decisions: dict[str, _Decision | None],
    config: ReconciliationConfig,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Encontra os grupos de regiões contíguas com o mesmo conceito reconciliado."""
    eligible = [
        region
        for region in regions
        if (decision := decisions.get(region.region_id)) is not None
        # Um ``thing`` não é agrupado como superfície: duas cadeiras encostadas
        # continuam sendo duas cadeiras, e agrupá-las inventaria identidade.
        and decision.record.reconciled_kind is not RegionKind.THING
    ]
    parent = {region.region_id: region.region_id for region in eligible}

    # Busca da raiz com path compression; espelha o union-find de
    # ``region_merge`` de propósito, porque a garantia exigida é a mesma:
    # agrupamento determinístico e independente da ordem de entrada.
    def find(region_id: str) -> str:
        while parent[region_id] != region_id:
            parent[region_id] = parent[parent[region_id]]
            region_id = parent[region_id]
        return region_id

    for index, first in enumerate(eligible):
        first_concept = _concept_of(decisions, first.region_id)
        for second in eligible[index + 1 :]:
            if first_concept != _concept_of(decisions, second.region_id):
                continue
            if not _touches(first, second, config.adjacency_margin_px):
                continue
            root_a, root_b = find(first.region_id), find(second.region_id)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

    grouped: dict[str, list[str]] = {}
    for region in eligible:
        grouped.setdefault(find(region.region_id), []).append(region.region_id)
    return tuple(
        (root, tuple(sorted(members))) for root, members in sorted(grouped.items()) if len(members) >= 2
    )


# Converte os grupos em hipóteses de entidade, medindo a coerência densa média
# quando há embeddings. Helper de reconcile_observation.
def _build_entities(
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    decisions: dict[str, _Decision | None],
    observation_id: str,
    provenance: ModelProvenance,
    config: ReconciliationConfig,
    visual_embeddings: tuple[VisualEmbedding, ...],
    visual_space: EmbeddingSpace | None,
) -> tuple[ContextualEntityHypothesis, ...]:
    """Constrói as hipóteses de entidade contextual a partir dos grupos encontrados."""
    vectors = {
        embedding.region_id: np.asarray(embedding.vector) for embedding in visual_embeddings
    }
    entities: list[ContextualEntityHypothesis] = []
    for _, members in groups:
        coherence = _mean_pairwise_similarity(members, vectors) if visual_space is not None else None
        if coherence is None:
            status = EntityHypothesisStatus.UNRESOLVED
            reason = "no dense evidence available to corroborate the group"
        elif coherence >= config.min_group_coherence:
            status, reason = EntityHypothesisStatus.SUPPORTED, None
        else:
            status = EntityHypothesisStatus.UNRESOLVED
            reason = (
                f"mean dense coherence {coherence:.3f} is below the configured "
                f"{config.min_group_coherence:.3f}"
            )
        first = decisions[members[0]]
        assert first is not None  # imposto por _same_surface_groups
        entities.append(
            ContextualEntityHypothesis(
                entity_id=derive_entity_id(observation_id, members),
                kind=EntityHypothesisKind.SAME_SURFACE,
                canonical_concept=first.record.canonical_concept,
                region_kind=first.record.reconciled_kind,
                member_region_ids=members,
                status=status,
                evidence=(
                    Evidence(
                        description=(
                            f"{len(members)} touching regions share the reconciled concept "
                            f"{first.record.canonical_concept!r}"
                        )
                    ),
                ),
                provenance=provenance,
                feature_coherence=coherence,
                space=None if coherence is None else visual_space,
                reason=reason,
                member_raw_labels=tuple(
                    _raw_label_of(decisions, member) for member in members
                ),
            )
        )
    return tuple(entities)


# Calcula a similaridade média entre todos os pares de membros de um grupo.
# Devolve ``None`` quando falta embedding de algum membro: uma média sobre um
# subconjunto silencioso seria uma corroboração que não corresponde ao grupo.
def _mean_pairwise_similarity(
    members: tuple[str, ...], vectors: dict[str, np.ndarray]
) -> float | None:
    """Retorna a coerência densa média do grupo, ou ``None`` sem cobertura completa."""
    present = [vectors[member] for member in members if member in vectors]
    if len(present) != len(members) or len(present) < 2:
        return None
    similarities = [
        float(present[i] @ present[j])
        for i in range(len(present))
        for j in range(i + 1, len(present))
    ]
    return float(np.clip(np.mean(similarities), -1.0, 1.0))


# Responde se duas regiões se tocam no plano da imagem. Existe aqui porque a
# reconciliação precisa de uma resposta binária com a sua própria margem
# configurável, enquanto ``relation_generation`` precisa da proximidade
# normalizada para pontuar ``near``.
def _touches(first: ObservedRegion, second: ObservedRegion, margin_px: float) -> bool:
    """Indica se as máscaras se sobrepõem ou as caixas estão a até ``margin_px``."""
    if first.mask.iou(second.mask) > 0.0:
        return True
    a, b = first.box, second.box
    gap_x = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    gap_y = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    return max(gap_x, gap_y) <= margin_px


# Lê o conceito reconciliado de uma região a partir das decisões já tomadas.
def _concept_of(decisions: dict[str, _Decision | None], region_id: str) -> str:
    """Retorna o conceito canônico da região, ou string vazia se não houver decisão."""
    decision = decisions.get(region_id)
    return "" if decision is None else decision.record.canonical_concept


# Lê o label cru de uma região a partir das decisões já tomadas.
def _raw_label_of(decisions: dict[str, _Decision | None], region_id: str) -> str:
    """Retorna o label cru da região, ou string vazia se não houver decisão."""
    decision = decisions.get(region_id)
    return "" if decision is None else decision.record.raw_label


# Retorna a claim de identidade reconciliada de uma região, quando existe.
# Existe para que os consumidores tenham uma única forma de perguntar "qual é a
# melhor interpretação atual desta região", em vez de cada um decidir sozinho
# entre a primária e a reconciliada. A precedência é explícita: a reconciliada
# quando ela existe, a primária caso contrário — e as duas continuam na região.
def reconciled_label_claim(region: ObservedRegion) -> SemanticClaim | None:
    """Retorna a claim reconciliada da região, ou a primária quando não houver.

    Argumentos:
        region: a região observada cujos claims serão inspecionados.
    Retorna:
        a interpretação mais informada disponível, ou ``None`` sem identidade.
    """
    for claim in region.claims:
        if claim.kind is ClaimKind.LABEL and claim.role is HypothesisRole.RECONCILED:
            return claim
    return primary_label_claim(region)


# Deriva o conceito pelo qual uma região deve ser referida por consumidores de
# alto nível (relações semânticas, reports). Existe para que essa escolha seja
# feita em um lugar só, e não replicada por cada consumidor.
def region_concept(region: ObservedRegion) -> str:
    """Retorna o conceito reconciliado da região, ou string vazia sem identidade."""
    claim = reconciled_label_claim(region)
    return "" if claim is None else claim.value


# Expõe o núcleo nominal de um conceito para consumidores que precisam agrupar
# por ele. Reexportado daqui para que quem já importa a reconciliação não
# precise conhecer ``domain/structural_consistency.py``.
__all__ = [
    "NON_DISCRIMINATIVE_MODIFIERS",
    "PRODUCER",
    "STAGE",
    "ReconciliationRecord",
    "ReconciliationResult",
    "canonical_concept",
    "head_noun",
    "reconcile_observation",
    "reconciled_label_claim",
    "region_concept",
]
