"""Testes de pooling de região mask-aware em alta resolução (#162)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.application.pooling import BASELINE, HIGH_RESOLUTION, pool_region_vector, pool_regions
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion


# Helper que monta um FeatureMap sintético com valores crescentes, para tornar o
# resultado do pooling previsível e fácil de verificar nos testes abaixo.
def _feature_map(grid: int = 4, dim: int = 3) -> FeatureMap:
    data = np.arange(grid * grid * dim, dtype=np.float64).reshape(grid, grid, dim) + 1.0
    return FeatureMap(data, stride_x=8.0 / grid * 4, stride_y=8.0 / grid * 4, dimension=dim, model_id="fake")


# Helper que monta uma Mask retangular simples a partir de uma bounding box, evitando
# repetir a construção de array booleano em cada teste.
def _mask(width: int, height: int, box: tuple[int, int, int, int]) -> Mask:
    data = np.zeros((height, width), dtype=np.bool_)
    x0, y0, x1, y1 = box
    data[y0:y1, x0:x1] = True
    return Mask(data, width, height)


# Confirma a limitação conhecida da estratégia BASELINE: uma mask menor que uma célula
# da grade não tem nenhum centro de célula dentro dela, então o pooling falha
# explicitamente em vez de retornar um vetor sem sentido.
def test_baseline_rejects_masks_smaller_than_one_cell() -> None:
    feature_map = FeatureMap(
        np.ones((2, 2, 2)), stride_x=16.0, stride_y=16.0, dimension=2, model_id="fake"
    )
    tiny_mask = _mask(32, 32, (0, 0, 2, 2))
    with pytest.raises(ValueError):
        pool_region_vector(tiny_mask, feature_map, BASELINE)


# Verifica que a estratégia HIGH_RESOLUTION resolve exatamente a limitação acima: uma
# mask sub-célula ainda produz um vetor de pooling válido (é o motivo dela existir).
def test_high_resolution_represents_small_masks() -> None:
    feature_map = FeatureMap(
        np.ones((2, 2, 2)), stride_x=16.0, stride_y=16.0, dimension=2, model_id="fake"
    )
    tiny_mask = _mask(32, 32, (0, 0, 2, 2))
    vector = pool_region_vector(tiny_mask, feature_map, HIGH_RESOLUTION)
    assert len(vector) == 2


# Garante que todo vetor de pooling produzido é finito e normalizado (norma L2 == 1),
# invariante exigido por consumidores que fazem busca por similaridade de cosseno.
def test_pooled_vector_is_finite_and_normalized() -> None:
    data = np.random.default_rng(0).normal(size=(4, 4, 3))
    feature_map = FeatureMap(data, stride_x=8.0, stride_y=8.0, dimension=3, model_id="fake")
    mask = _mask(32, 32, (4, 4, 20, 20))
    vector = np.array(pool_region_vector(mask, feature_map, HIGH_RESOLUTION))
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1.0)


# Confirma que pool_regions (a função de mais alto nível, usada pelo pipeline) mapeia
# corretamente cada embedding de volta ao region_id de origem.
def test_pool_regions_maps_embeddings_to_correct_region_ids() -> None:
    feature_map = _feature_map()
    mask = _mask(32, 32, (4, 4, 20, 20))
    region = ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p1",))
    embeddings = pool_regions((region,), feature_map)
    assert embeddings[0].region_id == "region-a"
