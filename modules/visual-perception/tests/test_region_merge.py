"""Testes determinísticos de merge de região cross-scale (#160)."""

from __future__ import annotations

import numpy as np

from visual_perception.application.region_merge import merge_regions
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import RegionProposal, TileProvenance


# Helper que monta uma RegionProposal retangular simples a partir de uma bounding box,
# reutilizado por todos os testes de merge abaixo.
def _proposal(proposal_id: str, box: tuple[int, int, int, int], size: int = 16) -> RegionProposal:
    data = np.zeros((size, size), dtype=np.bool_)
    x0, y0, x1, y1 = box
    data[y0:y1, x0:x1] = True
    mask = Mask(data, size, size)
    tile = TileProvenance("full", "whole")
    return RegionProposal(proposal_id, mask, mask.bounding_box(), 0.8, "fake", tile)


# Verifica o caso central de merge: duas proposals idênticas (mesma geometria) colapsam
# em uma única região, mas ambas ficam registradas como contribuintes.
def test_duplicate_proposals_merge_into_one_region() -> None:
    proposals = (_proposal("p1", (2, 2, 6, 6)), _proposal("p2", (2, 2, 6, 6)))
    regions = merge_regions("obs-1", proposals, RegionMergeConfig())
    assert len(regions) == 1
    assert set(regions[0].contributing_proposal_ids) == {"p1", "p2"}


# Garante que proposals sem nenhuma sobreposição permanecem como regiões separadas.
def test_disjoint_proposals_stay_separate() -> None:
    proposals = (_proposal("p1", (0, 0, 2, 2)), _proposal("p2", (10, 10, 12, 12)))
    regions = merge_regions("obs-1", proposals, RegionMergeConfig())
    assert len(regions) == 2


# Confirma que uma proposal pequena inteiramente contida em uma maior NÃO é removida por
# ser "redundante" — containment não é motivo suficiente para descartar uma parte.
def test_contained_part_is_not_removed() -> None:
    large = _proposal("large", (0, 0, 10, 10))
    small_part = _proposal("small", (1, 1, 3, 3))
    regions = merge_regions("obs-1", (large, small_part), RegionMergeConfig())
    assert len(regions) == 2


# Verifica o critério de merge por IoU: duas proposals com sobreposição alta (acima do
# threshold configurado) são fundidas em uma só região.
def test_high_overlap_merges_via_iou() -> None:
    proposals = (_proposal("p1", (0, 0, 10, 10)), _proposal("p2", (0, 0, 9, 10)))
    regions = merge_regions("obs-1", proposals, RegionMergeConfig(iou_merge_threshold=0.85))
    assert len(regions) == 1


# Garante que os region_id gerados são estáveis: a mesma entrada, processada duas vezes,
# produz o mesmo id de região (necessário para o cache de estágio e para reprodutibilidade).
def test_region_ids_are_stable_for_identical_input() -> None:
    proposals = (_proposal("p1", (0, 0, 4, 4)),)
    first = merge_regions("obs-1", proposals, RegionMergeConfig())
    second = merge_regions("obs-1", proposals, RegionMergeConfig())
    assert first[0].region_id == second[0].region_id
