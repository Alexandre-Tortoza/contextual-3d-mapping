"""Funde regiões com o mesmo label predominante que se sobrepõem geometricamente.

Roda depois da semântica de região (#165), como um passo opcional de
pós-processamento: reduz o over-segmentation do SAM em superfícies uniformes
(teto, parede, piso) onde múltiplas propostas sobrevivem ao merge geométrico
por IoU/containment (#160, ``region_merge.py``) — que roda antes da semântica,
sobre a geometria pura — por terem sobreposição abaixo do threshold geométrico,
mas que se revelam a mesma superfície uma vez que o VLM as rotula com o mesmo
label.

Não é chamado por ``run_canonical_pipeline`` (#169): é um passo de
pós-processamento explícito, usado hoje pelo harness de validação
(``benchmarks/validate_reference_pipeline.py``, #190). Regiões fundidas
perdem ``visual_embedding_ref``/``language_embedding_ref`` (a máscara unida
não corresponde mais ao embedding calculado sobre a máscara original) — quem
chamar este passo deve recalcular relações sobre o resultado, já que a
identidade das regiões mudou.
"""

from __future__ import annotations

import dataclasses

from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, most_confident_claim


# Extrai o valor do claim de label de maior confiança de uma região, ou None
# se a região não tiver nenhum claim de label pontuado (nunca funde regiões sem
# label decidível — não há base para agrupá-las). A política de "pontuada vence
# não pontuada" mora em most_confident_claim, não aqui.
def _top_label(region: ObservedRegion) -> str | None:
    label_claims = tuple(claim for claim in region.claims if claim.kind is ClaimKind.LABEL)
    winner = most_confident_claim(label_claims)
    return None if winner is None else winner.value


# Funde, dentro de cada grupo de mesmo label predominante, as regiões cujas
# máscaras se sobrepõem (IoU > 0) — usa union-find para fundir cadeias de
# sobreposição (A sobrepõe B, B sobrepõe C -> A/B/C viram uma só região).
def merge_same_label_regions(regions: tuple[ObservedRegion, ...]) -> tuple[ObservedRegion, ...]:
    """Funde regiões com o mesmo label predominante cujas máscaras se sobrepõem."""
    groups: dict[str | None, list[ObservedRegion]] = {}
    for region in regions:
        groups.setdefault(_top_label(region), []).append(region)

    merged: list[ObservedRegion] = []
    for label, group in groups.items():
        if label is None or len(group) == 1:
            merged.extend(group)
            continue
        merged.extend(_merge_overlapping(group))
    return tuple(merged)


# Agrupa regiões por componente conexo de sobreposição (union-find sobre
# pares com IoU > 0) e funde cada componente em uma única região.
def _merge_overlapping(group: list[ObservedRegion]) -> list[ObservedRegion]:
    parent = list(range(len(group)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if group[i].mask.iou(group[j].mask) > 0.0:
                union(i, j)

    clusters: dict[int, list[ObservedRegion]] = {}
    for index, region in enumerate(group):
        clusters.setdefault(find(index), []).append(region)

    return [cluster[0] if len(cluster) == 1 else _union_regions(cluster) for cluster in clusters.values()]


# Constrói uma única ObservedRegion a partir de um cluster de regiões
# sobrepostas: máscara é a união pixel a pixel, claims e proposals
# contribuintes são a união deduplicada, confiança geométrica é o máximo do
# cluster. Escolhe o menor region_id do cluster (determinístico, estável
# entre execuções com a mesma entrada) como identidade da região fundida.
def _union_regions(cluster: list[ObservedRegion]) -> ObservedRegion:
    base = min(cluster, key=lambda region: region.region_id)
    data = base.mask.data.copy()
    for region in cluster:
        data |= region.mask.data
    mask = Mask(data, base.mask.image_width, base.mask.image_height)

    contributing = tuple(
        dict.fromkeys(pid for region in cluster for pid in region.contributing_proposal_ids)
    )
    claims = tuple(dict.fromkeys(claim for region in cluster for claim in region.claims))
    geometric_confidence = max(region.geometric_confidence for region in cluster)

    return dataclasses.replace(
        base,
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=geometric_confidence,
        contributing_proposal_ids=contributing,
        claims=claims,
        visual_embedding_ref=None,
        language_embedding_ref=None,
    )
