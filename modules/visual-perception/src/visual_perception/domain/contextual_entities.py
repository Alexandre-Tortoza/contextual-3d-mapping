"""Contract de hipótese de entidade contextual intra-frame.

Issue: #205.

O SAM é class-agnostic e recorta superfícies contínuas em pedaços. Medido no
run ``20260908T131207Z``: **108 de 165 regiões** (65%) pertencem a algum grupo
de regiões que se tocam e compartilham o label primário — ``wall`` com 13
membros adjacentes em ``corridor-02-000``, ``ceiling`` com 8. Nenhum estágio
do módulo dizia isso ao downstream.

:class:`ContextualEntityHypothesis` é o registro dessa observação, e o nome de
cada palavra dele é deliberado:

- **contextual**, porque só existe olhando todas as regiões do frame ao mesmo
  tempo; nenhuma região isolada consegue afirmá-lo;
- **entity**, porque é sobre identidade, não sobre geometria: a geometria de
  cada membro continua exatamente onde estava;
- **hypothesis**, porque é 2D e de um frame só. Se as três paredes são a mesma
  parede *no mundo* é uma pergunta que exige geometria 3D, e essa pergunta
  pertence a ``sensor-association``/``semantic-fusion``.

Um grupo **nunca** substitui seus membros. O downstream recebe as 13 regiões e,
ao lado delas, a informação de que provavelmente são manifestações da mesma
superfície — que é estritamente mais do que ele recebia antes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from visual_perception.domain.embeddings import EmbeddingSpace
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import Evidence, RegionKind

__all__ = [
    "ContextualEntityHypothesis",
    "EntityHypothesisKind",
    "EntityHypothesisStatus",
    "derive_entity_id",
]


# Nomeia o tipo de afirmação que um grupo faz. É um vocabulário fechado e
# versionado, e não uma string livre, porque o campo é serializado e o
# downstream decide comportamento a partir dele. Hoje há um valor só: é o único
# para o qual existe evidência 2D e consumidor concreto. Novos valores entram
# quando as duas coisas existirem — não antes.
class EntityHypothesisKind(StrEnum):
    """O que um grupo de regiões afirma sobre seus membros."""

    #: Os membros se tocam no plano da imagem, compartilham o conceito
    #: reconciliado e são matéria contínua: provavelmente são recortes da mesma
    #: superfície física.
    SAME_SURFACE = "same_surface"


# Distingue um grupo corroborado por evidência densa de um grupo apenas
# proposto. Existe porque a medição foi explícita quanto ao limite do sinal:
# entre pares adjacentes, o cosseno denso separa mesmo-label de label-diferente
# com acurácia balanceada de apenas 0,660 no melhor limiar. Um grupo cuja
# coerência é baixa continua sendo evidência — mas evidência que o downstream
# precisa poder tratar diferente.
class EntityHypothesisStatus(StrEnum):
    """Quanto a evidência densa corrobora um grupo proposto."""

    #: A coerência densa média entre os membros ficou no nível esperado para
    #: uma superfície contínua.
    SUPPORTED = "supported"
    #: O grupo é consistente em conceito e contato, mas a evidência densa não
    #: o corrobora; a identidade permanece em aberto.
    UNRESOLVED = "unresolved"


# Deriva um identificador estável de grupo a partir dos membros. Existe pela
# mesma razão que ``derive_region_id``: o mesmo conjunto de regiões, na mesma
# observação, precisa produzir sempre o mesmo id, para que dois runs da mesma
# configuração sejam comparáveis linha a linha.
def derive_entity_id(observation_id: str, member_region_ids: tuple[str, ...]) -> str:
    """Deriva um id de entidade contextual estável e independente de ordem.

    Argumentos:
        observation_id: identidade da observação que contém o grupo.
        member_region_ids: ids das regiões que compõem o grupo.
    Retorna:
        um identificador ``entity-<digest>`` determinístico.
    Levanta:
        ValueError: se o grupo não tiver membros.
    """
    if not member_region_ids:
        raise ValueError("A contextual entity must be derived from at least one region.")
    canonical = "|".join(sorted(member_region_ids))
    digest = hashlib.sha256(f"{observation_id}:{canonical}".encode()).hexdigest()[:16]
    return f"entity-{digest}"


# Agrupa regiões que provavelmente são manifestações da mesma entidade dentro
# de um frame, preservando todos os membros. Existe como o produto da
# reconciliação intra-frame (#205); consumido pela serialização canônica e, em
# seguida, por ``sensor-association``, que é quem tem geometria para promover
# ou refutar a hipótese.
@dataclass(frozen=True)
class ContextualEntityHypothesis:
    """Uma hipótese de que várias regiões do mesmo frame são a mesma entidade.

    ``feature_coherence`` é a similaridade densa média entre os membros, no
    espaço declarado por ``space``. Ela **corrobora**, e não decide: o grupo é
    proposto por compatibilidade de conceito mais contato espacial, porque a
    medição mostrou que a similaridade densa sozinha não separa bem o caso.
    """

    entity_id: str
    kind: EntityHypothesisKind
    canonical_concept: str
    region_kind: RegionKind
    member_region_ids: tuple[str, ...]
    status: EntityHypothesisStatus
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance
    feature_coherence: float | None = None
    space: EmbeddingSpace | None = None
    reason: str | None = None
    #: Os labels crus que os membros afirmaram, na ordem dos membros. Existe
    #: para que a canonicalização continue auditável: o grupo diz o conceito a
    #: que chegou **e** de quais afirmações originais ele partiu.
    member_raw_labels: tuple[str, ...] = field(default_factory=tuple)

    # Impõe que o grupo seja uma afirmação sobre mais de uma região, que os
    # membros sejam distintos e ordenados de forma determinística, e que uma
    # coerência declarada seja um cosseno válido acompanhado do seu espaço.
    def __post_init__(self) -> None:
        """Valida identidade, membros, coerência e proveniência do grupo."""
        validate_identifier(self.entity_id, field="entity_id")
        if len(self.member_region_ids) < 2:
            raise ValueError(
                f"ContextualEntityHypothesis({self.entity_id!r}) needs at least two members: a group "
                "of one is the region itself, and asserting it would add nothing."
            )
        if len(set(self.member_region_ids)) != len(self.member_region_ids):
            raise ValueError(f"ContextualEntityHypothesis({self.entity_id!r}) repeats a member region.")
        if tuple(sorted(self.member_region_ids)) != self.member_region_ids:
            raise ValueError(
                f"ContextualEntityHypothesis({self.entity_id!r}) must list its members in sorted order, "
                "so that two runs of the same configuration are comparable line by line."
            )
        for member in self.member_region_ids:
            validate_identifier(member, field="member_region_ids")
        if not self.canonical_concept.strip():
            raise ValueError("ContextualEntityHypothesis.canonical_concept must not be empty.")
        if not self.evidence:
            raise ValueError(
                f"ContextualEntityHypothesis({self.entity_id!r}) must reference at least one Evidence."
            )
        if self.member_raw_labels and len(self.member_raw_labels) != len(self.member_region_ids):
            raise ValueError(
                f"ContextualEntityHypothesis({self.entity_id!r}) must record one raw label per member "
                "or none at all."
            )
        if self.feature_coherence is not None:
            if not -1.0 <= self.feature_coherence <= 1.0:
                raise ValueError(
                    f"feature_coherence must be a cosine in [-1, 1], got {self.feature_coherence}."
                )
            if self.space is None:
                raise ValueError(
                    "A declared feature_coherence requires the embedding space it was measured in: "
                    "a similarity without a space is not comparable to anything."
                )
        if self.status is EntityHypothesisStatus.UNRESOLVED and not self.reason:
            raise ValueError(
                f"An unresolved ContextualEntityHypothesis({self.entity_id!r}) must explain itself."
            )

    # Responde se uma região participa do grupo. Existe para que consumidores
    # (relações semânticas, auditoria, downstream) não varram a tupla na mão.
    def contains(self, region_id: str) -> bool:
        """Indica se ``region_id`` é um dos membros deste grupo."""
        return region_id in self.member_region_ids
