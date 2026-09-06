"""Geração de relations em nível de imagem.

Issues: #167 (geração de relations), #195 (proibição de confiança inventada).

Gera relations 2D-geométricas diretamente a partir da geometria de region
já mesclada, e opcionalmente aceita relations inferidas por modelo que uma
etapa multimodal já resolveu para region ids canônicos (a resolução
subject/object a partir de texto bruto de modelo não é responsabilidade
desta função). Ambos os tipos mantêm sua proveniência separadamente, e
ambos ainda são explicitamente relations *candidatas*, não verificadas
(#166): nenhuma validação 3D acontece aqui.

A confiança de uma relation geométrica é a *própria medida* que a produziu
(IoU, containment ratio, proximidade normalizada), e não o ``1.0`` fixo que
esta etapa atribuía antes: uma sobreposição de 1% e uma de 99% geravam a
mesma certeza máxima. Uma relation inferida por modelo sem score fica com
``confidence=None``, nunca com um número inventado.
"""

from __future__ import annotations

from typing import Any

from visual_perception.application.support import fingerprint_of
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import (
    CandidateRelation,
    RelationSource,
    validate_relation_references,
)
from visual_perception.domain.semantics import ConfidenceScore, Evidence

_CONTAINMENT_THRESHOLD = 0.9
_ADJACENCY_MARGIN_PX = 5.0


# Ponto de entrada público: gera todas as relations candidatas de uma
# observation, combinando relations geométricas (calculadas par a par entre
# regions) com relations inferidas por modelo (já resolvidas para region ids
# canônicos), e valida que todas as referências apontam para regions
# conhecidas. Chamada pelo pipeline principal após o merge de regions.
def generate_relations(
    regions: tuple[ObservedRegion, ...],
    config: RegionMergeConfig,
    inferred_relations: tuple[dict[str, Any], ...] = (),
) -> tuple[CandidateRelation, ...]:
    """Gera relations candidatas, referenciando apenas as regions canônicas fornecidas."""
    geometric_provenance = ModelProvenance(
        stage="relation_generation", producer="geometric_2d", config_fingerprint=fingerprint_of(config)
    )
    relations: list[CandidateRelation] = []
    for i, subject in enumerate(regions):
        for target in regions[i + 1 :]:
            relations.extend(
                _geometric_relations(subject, target, geometric_provenance)
            )

    for index, raw in enumerate(inferred_relations):
        relations.append(_model_inferred_relation(raw, index))

    known_ids = frozenset(region.region_id for region in regions)
    result = tuple(relations)
    validate_relation_references(result, known_ids)
    return result


# Calcula as relations 2D-geométricas entre um par de regions (overlaps,
# contains, near) a partir de IoU, containment ratio e proximidade de box.
# Existe como o núcleo geométrico da geração de relations, sem depender de
# nenhum modelo. Chamada por generate_relations para cada par de regions.
def _geometric_relations(
    subject: ObservedRegion, target: ObservedRegion, provenance: ModelProvenance
) -> list[CandidateRelation]:
    """Calcula overlaps/contains/near entre duas regions, pontuadas pela própria medida."""
    relations: list[CandidateRelation] = []
    iou = subject.mask.iou(target.mask)
    evidence = (Evidence(description=f"mask overlap between {subject.region_id} and {target.region_id}"),)

    if iou > 0.0:
        relations.append(
            CandidateRelation(
                relation_id=f"rel-overlaps-{subject.region_id}-{target.region_id}",
                subject_region_id=subject.region_id,
                predicate="overlaps",
                object_region_id=target.region_id,
                confidence=ConfidenceScore(iou, source="geometric_2d"),
                source=RelationSource.GEOMETRIC_2D,
                evidence=evidence,
                provenance=provenance,
            )
        )

    forward = subject.mask.containment_ratio(target.mask)
    backward = target.mask.containment_ratio(subject.mask)
    if forward >= _CONTAINMENT_THRESHOLD and forward > backward:
        relations.append(
            _contains(subject, target, ConfidenceScore(forward, source="geometric_2d"), provenance)
        )
    elif backward >= _CONTAINMENT_THRESHOLD and backward > forward:
        relations.append(
            _contains(target, subject, ConfidenceScore(backward, source="geometric_2d"), provenance)
        )

    proximity = _box_proximity(subject, target)
    if iou == 0.0 and proximity is not None:
        relations.append(
            CandidateRelation(
                relation_id=f"rel-near-{subject.region_id}-{target.region_id}",
                subject_region_id=subject.region_id,
                predicate="near",
                object_region_id=target.region_id,
                confidence=ConfidenceScore(proximity, source="geometric_2d"),
                source=RelationSource.GEOMETRIC_2D,
                evidence=evidence,
                provenance=provenance,
            )
        )
    return relations


# Constrói a CandidateRelation "contains" entre uma region container e uma
# region part. Existe para não duplicar a construção do objeto nos dois
# sentidos possíveis (subject contém target, ou vice-versa) dentro de
# _geometric_relations, que é quem chama esta função.
def _contains(
    container: ObservedRegion, part: ObservedRegion, confidence: ConfidenceScore, provenance: ModelProvenance
) -> CandidateRelation:
    """Constrói a relation ``contains`` já pontuada pelo containment ratio medido."""
    return CandidateRelation(
        relation_id=f"rel-contains-{container.region_id}-{part.region_id}",
        subject_region_id=container.region_id,
        predicate="contains",
        object_region_id=part.region_id,
        confidence=confidence,
        source=RelationSource.GEOMETRIC_2D,
        evidence=(Evidence(description=f"{container.region_id} contains most of {part.region_id}"),),
        provenance=provenance,
    )


# Mede quão próximos estão os boxes de duas regions, normalizando o maior
# vão para [0, 1] dentro de _ADJACENCY_MARGIN_PX. Existe para que a relation
# "near" carregue a distância medida como confiança em vez de um valor fixo;
# chamada por _geometric_relations quando as masks não se sobrepõem.
def _box_proximity(subject: ObservedRegion, target: ObservedRegion) -> float | None:
    """Retorna a proximidade normalizada em ``[0, 1]``, ou ``None`` se os boxes estão longe.

    ``1.0`` significa boxes encostados; ``0.0``, exatamente na margem de
    adjacência. Acima da margem não há relation "near" a gerar.
    """
    a, b = subject.box, target.box
    gap_x = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
    gap_y = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
    gap = max(gap_x, gap_y)
    if gap > _ADJACENCY_MARGIN_PX:
        return None
    return 1.0 - gap / _ADJACENCY_MARGIN_PX


# Converte uma relation bruta inferida por modelo (dict solto vindo do
# reasoner) em uma CandidateRelation validada, verificando os campos
# obrigatórios e os identifiers antes de aceitar o dado externo. Chamada
# por generate_relations para cada relation em inferred_relations.
def _model_inferred_relation(raw: dict[str, Any], index: int) -> CandidateRelation:
    try:
        subject_id = str(raw["subject_region_id"])
        predicate = str(raw["predicate"])
        object_id = str(raw["object_region_id"])
    except KeyError as error:
        raise ValueError(f"Malformed inferred relation at index {index}: missing {error}.") from error
    validate_identifier(subject_id, field="subject_region_id")
    validate_identifier(object_id, field="object_region_id")
    provenance = ModelProvenance(
        stage="relation_generation",
        producer=str(raw.get("producer", "multimodal_reasoner")),
        config_fingerprint=str(raw.get("config_fingerprint", "unknown")),
    )
    raw_confidence = raw.get("confidence")
    if raw_confidence is not None and (
        isinstance(raw_confidence, bool) or not isinstance(raw_confidence, int | float)
    ):
        raise ValueError(
            f"Malformed inferred relation at index {index}: 'confidence' must be a number or absent."
        )
    return CandidateRelation(
        relation_id=f"rel-inferred-{index}-{subject_id}-{object_id}",
        subject_region_id=subject_id,
        predicate=predicate,
        object_region_id=object_id,
        confidence=(
            None if raw_confidence is None else ConfidenceScore(float(raw_confidence), provenance.producer)
        ),
        source=RelationSource.MODEL_INFERRED,
        evidence=(Evidence(description="model-inferred relation from contextual analysis"),),
        provenance=provenance,
    )
