"""Contract de relação candidata estruturada.

Issue: #166.

Relações são candidatas apenas em nível de imagem 2D. Elas nunca são
tratadas como relações 3D verificadas: essa validação é de posse de módulos
downstream (scene-graph / context-reasoning), depois que a geometria foi
associada entre sensores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import ConfidenceScore, Evidence

_PREDICATE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


# Distingue se uma relação foi derivada de geometria 2D ou inferida por um
# modelo. Existe para manter essas duas origens distinguíveis (#167) em vez
# de fundir a proveniência.
class RelationSource(StrEnum):
    """Se uma relação foi derivada de geometria 2D ou inferida por um modelo.

    Issue: #167 mantém essas origens distinguíveis em vez de fundir a
    proveniência.
    """

    GEOMETRIC_2D = "geometric_2d"
    MODEL_INFERRED = "model_inferred"


# Representa uma relação não verificada, em nível de imagem, entre duas
# regiões canônicas. Existe como a unidade de relação candidata que
# scene-graph/context-reasoning eventualmente verificam em 3D.
@dataclass(frozen=True)
class CandidateRelation:
    """Uma relação não verificada, em nível de imagem, entre duas regiões canônicas."""

    relation_id: str
    subject_region_id: str
    predicate: str
    object_region_id: str
    confidence: ConfidenceScore
    source: RelationSource
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance

    # Valida os ids envolvidos, rejeita auto-relações, exige um predicate em
    # snake_case normalizado, e exige ao menos uma Evidence.
    def __post_init__(self) -> None:
        validate_identifier(self.relation_id, field="relation_id")
        validate_identifier(self.subject_region_id, field="subject_region_id")
        validate_identifier(self.object_region_id, field="object_region_id")
        if self.subject_region_id == self.object_region_id:
            raise ValueError(
                f"CandidateRelation({self.relation_id!r}) is a self-relation, which is not "
                "supported: subject_region_id and object_region_id must differ."
            )
        if not _PREDICATE_PATTERN.match(self.predicate):
            raise ValueError(
                f"predicate must be normalized snake_case matching {_PREDICATE_PATTERN.pattern!r}, "
                f"got {self.predicate!r}."
            )
        if not self.evidence:
            raise ValueError(
                f"CandidateRelation({self.relation_id!r}) must reference at least one Evidence."
            )


# Rejeita relações que referenciam regiões fora do conjunto conhecido.
# Existe para detectar referências penduradas (ex: a uma região descartada
# durante merge/refinamento) explicitamente, em vez de deixá-las passar
# silenciosamente; usada por VisualObservation.__post_init__.
def validate_relation_references(
    relations: tuple[CandidateRelation, ...], known_region_ids: frozenset[str]
) -> None:
    """Rejeita relações que referenciam regiões fora de ``known_region_ids``.

    Uma referência pendurada (ex: para uma região descartada durante
    merge/refinamento) falha explicitamente em vez de ser descartada
    silenciosamente.
    """
    for relation in relations:
        for region_id, role in (
            (relation.subject_region_id, "subject_region_id"),
            (relation.object_region_id, "object_region_id"),
        ):
            if region_id not in known_region_ids:
                raise ValueError(
                    f"CandidateRelation({relation.relation_id!r}).{role} references unknown "
                    f"region {region_id!r}."
                )
