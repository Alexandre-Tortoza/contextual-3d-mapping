"""Testes determinísticos do merge semântico pós-interpretação (over-segmentation)."""

from __future__ import annotations

import numpy as np

from visual_perception.application.semantic_merge import merge_same_label_regions
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim


# Helper que monta uma Mask retangular numa imagem 10x10, reutilizado por todos os
# testes deste arquivo.
def _mask(box: tuple[int, int, int, int]) -> Mask:
    x0, y0, x1, y1 = box
    data = np.zeros((10, 10), dtype=np.bool_)
    data[y0:y1, x0:x1] = True
    return Mask(data, 10, 10)


# Helper que monta um SemanticClaim de label mínimo com o value/confiança dados.
def _label(value: str, confidence: float = 0.8) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(confidence, source="fake"),
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
    )


# Helper que monta uma ObservedRegion com a box e os claims dados.
def _region(
    region_id: str, box: tuple[int, int, int, int], claims: tuple[SemanticClaim, ...]
) -> ObservedRegion:
    mask = _mask(box)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",), claims=claims)


# Confirma o caso central: duas regiões com o mesmo label predominante e máscaras
# sobrepostas colapsam em uma única região cuja máscara é a união das duas.
def test_overlapping_same_label_regions_merge() -> None:
    a = _region("a", (0, 0, 6, 6), (_label("floor"),))
    b = _region("b", (4, 4, 8, 8), (_label("floor"),))

    merged = merge_same_label_regions((a, b))

    assert len(merged) == 1
    expected_union = np.logical_or(a.mask.data, b.mask.data)
    assert np.array_equal(merged[0].mask.data, expected_union)
    assert merged[0].mask.data[5, 5]  # pixel de sobreposição continua ocupado
    assert set(merged[0].contributing_proposal_ids) == {"a-p", "b-p"}


# Garante que regiões do mesmo label mas sem nenhuma sobreposição de máscara
# permanecem separadas — merge semântico não funde por rótulo sozinho.
def test_same_label_without_overlap_stays_separate() -> None:
    a = _region("a", (0, 0, 2, 2), (_label("floor"),))
    b = _region("b", (8, 8, 10, 10), (_label("floor"),))

    merged = merge_same_label_regions((a, b))

    assert len(merged) == 2


# Garante que regiões sobrepostas com labels diferentes NÃO são fundidas — o
# merge é estritamente por label predominante igual, nunca por geometria sozinha
# (essa responsabilidade já é do merge geométrico #160, anterior à semântica).
def test_overlapping_different_labels_stay_separate() -> None:
    a = _region("a", (0, 0, 6, 6), (_label("floor"),))
    b = _region("b", (4, 4, 8, 8), (_label("wall"),))

    merged = merge_same_label_regions((a, b))

    assert len(merged) == 2


# Confirma que regiões sem nenhum claim de label nunca são fundidas entre si,
# mesmo se sobrepostas — não há base semântica para agrupá-las.
def test_regions_without_label_claims_never_merge() -> None:
    a = _region("a", (0, 0, 6, 6), ())
    b = _region("b", (4, 4, 8, 8), ())

    merged = merge_same_label_regions((a, b))

    assert len(merged) == 2


# Verifica que a região fundida preserva a união deduplicada de claims de
# ambas as regiões originais, sem duplicar o claim de label compartilhado.
def test_merged_region_deduplicates_shared_claims() -> None:
    shared = _label("floor")
    a = _region("a", (0, 0, 6, 6), (shared, _label("worn", confidence=0.6)))
    b = _region("b", (4, 4, 8, 8), (shared,))

    merged = merge_same_label_regions((a, b))

    assert len(merged) == 1
    assert merged[0].claims.count(shared) == 1


# region_id da região fundida é determinístico (o menor id do cluster), para
# que a mesma entrada produza sempre o mesmo resultado.
def test_merged_region_id_is_deterministic() -> None:
    a = _region("region-b", (0, 0, 6, 6), (_label("floor"),))
    b = _region("region-a", (4, 4, 8, 8), (_label("floor"),))

    merged = merge_same_label_regions((a, b))

    assert merged[0].region_id == "region-a"
