"""Testes determinísticos de merge de região cross-scale (#160)."""

from __future__ import annotations

import numpy as np
import pytest

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


# Regressão: o union-find era indexado por proposal_id e duas propostas de
# fontes diferentes com o mesmo ID sobrescreviam uma à outra silenciosamente.
def test_duplicate_proposal_ids_are_rejected_before_merge() -> None:
    """Recusa IDs repetidos para não perder proveniência no merge."""
    proposals = (_proposal("same-id", (2, 2, 6, 6)), _proposal("same-id", (2, 2, 6, 6)))

    with pytest.raises(ValueError, match="must be unique"):
        merge_regions("obs-1", proposals, RegionMergeConfig())


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


# Protege a fronteira de tiling: duas metades quase disjuntas de uma mesma
# estrutura não são duplicatas geométricas. Reconstruí-las exigiria uma regra
# nova de stitching, que não pertence ao merge atual de IoU/containment.
def test_adjacent_tile_fragments_are_not_merged_without_geometric_overlap() -> None:
    """Mantém fragmentos adjacentes separados quando não há evidência de duplicidade."""
    left = _proposal("tile-left", (0, 2, 7, 10))
    right = _proposal("tile-right", (7, 2, 14, 10))

    regions = merge_regions("obs-1", (left, right), RegionMergeConfig())

    assert len(regions) == 2


# Garante que os region_id gerados são estáveis: a mesma entrada, processada duas vezes,
# produz o mesmo id de região (necessário para o cache de estágio e para reprodutibilidade).
def test_region_ids_are_stable_for_identical_input() -> None:
    proposals = (_proposal("p1", (0, 0, 4, 4)),)
    first = merge_regions("obs-1", proposals, RegionMergeConfig())
    second = merge_regions("obs-1", proposals, RegionMergeConfig())
    assert first[0].region_id == second[0].region_id
