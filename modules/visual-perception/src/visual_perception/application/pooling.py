"""Pooling de região em alta resolução, consciente de máscara.

Issue: #162.

Dois métodos de pooling são suportados atrás de uma única função:

- ``patch_grid_baseline``: inclui uma célula do grid de features apenas
  quando o *centro* do seu pixel cai dentro da máscara da região, depois
  faz a média sem peso. Simples, mas uma máscara menor que uma célula do
  grid não tem centro de célula dentro dela e é rejeitada.
- ``pixel_nearest_highres``: reúne, para cada pixel da máscara, o vetor de
  feature da célula do grid mais próxima, depois faz a média. Isso mantém
  regiões pequenas representáveis sempre que pelo menos um pixel da
  máscara tiver suporte de feature.

Os dois caminhos normalizam L2 o vetor resultante (comportamento de
normalização documentado); uma normalização só é pulada quando
inalcançável (entrada finita e não-zero), o que não pode acontecer depois
que um método já aceitou uma máscara.
"""

from __future__ import annotations

import numpy as np

from visual_perception.domain.embeddings import VisualEmbedding
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion

BASELINE = "patch_grid_baseline"
HIGH_RESOLUTION = "pixel_nearest_highres"
_METHODS = frozenset({BASELINE, HIGH_RESOLUTION})


# Agrega as dense features de uma única região em um vetor final, finito e
# normalizado L2, escolhendo entre os métodos BASELINE e HIGH_RESOLUTION;
# usada por pool_regions para cada região do batch.
def pool_region_vector(mask: Mask, feature_map: FeatureMap, method: str) -> tuple[float, ...]:
    """Agrega as dense features de uma região em um único vetor finito e normalizado L2."""
    if method not in _METHODS:
        raise ValueError(f"Unknown pooling method {method!r}, expected one of {sorted(_METHODS)}.")
    stride_x = mask.image_width / feature_map.grid_width
    stride_y = mask.image_height / feature_map.grid_height
    if method == BASELINE:
        vector = _pool_patch_grid_baseline(mask, feature_map, stride_x, stride_y)
    else:
        vector = _pool_pixel_nearest(mask, feature_map, stride_x, stride_y)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Pooled vector has zero norm and cannot be normalized.")
    return tuple((vector / norm).tolist())


# Implementa o método BASELINE: inclui uma célula do grid só quando o
# centro do seu pixel cai dentro da máscara, depois faz a média simples;
# helper interno de pool_region_vector, mais barato porém mais grosseiro
# para regiões pequenas.
def _pool_patch_grid_baseline(
    mask: Mask, feature_map: FeatureMap, stride_x: float, stride_y: float
) -> np.ndarray:
    included = []
    for grid_y in range(feature_map.grid_height):
        center_y = int((grid_y + 0.5) * stride_y)
        if not 0 <= center_y < mask.image_height:
            continue
        for grid_x in range(feature_map.grid_width):
            center_x = int((grid_x + 0.5) * stride_x)
            if 0 <= center_x < mask.image_width and mask.data[center_y, center_x]:
                included.append(feature_map.data[grid_y, grid_x])
    if not included:
        raise ValueError(
            "No feature-grid cell center falls inside the mask; the region cannot be aligned "
            f"with the '{BASELINE}' method. Try '{HIGH_RESOLUTION}' for small regions."
        )
    return np.mean(np.stack(included), axis=0)


# Implementa o método HIGH_RESOLUTION: para cada pixel da máscara, busca a
# feature da célula do grid mais próxima e faz a média; helper interno de
# pool_region_vector, preferido para regiões pequenas que o BASELINE
# rejeitaria.
def _pool_pixel_nearest(
    mask: Mask, feature_map: FeatureMap, stride_x: float, stride_y: float
) -> np.ndarray:
    ys, xs = np.where(mask.data)
    if ys.size == 0:
        raise ValueError("Cannot pool an empty mask.")
    grid_ys = np.clip((ys / stride_y).astype(np.int64), 0, feature_map.grid_height - 1)
    grid_xs = np.clip((xs / stride_x).astype(np.int64), 0, feature_map.grid_width - 1)
    gathered = feature_map.data[grid_ys, grid_xs]
    return np.mean(gathered, axis=0)


# Agrupa pool_region_vector sobre todas as regiões de uma observação
# contra um único feature map denso, produzindo os VisualEmbedding
# consumidos pelo pipeline canônico logo após o merge de regiões.
def pool_regions(
    regions: tuple[ObservedRegion, ...],
    feature_map: FeatureMap,
    method: str = HIGH_RESOLUTION,
) -> tuple[VisualEmbedding, ...]:
    """Faz pooling de cada região contra um feature map denso em um :class:`VisualEmbedding`."""
    embeddings = []
    for region in regions:
        vector = pool_region_vector(region.mask, feature_map, method)
        embeddings.append(
            VisualEmbedding(
                embedding_id=f"visual-{region.region_id}",
                region_id=region.region_id,
                vector=vector,
                dimension=len(vector),
                pooling_method=method,
                feature_resolution=f"{feature_map.grid_width}x{feature_map.grid_height}",
                model_id=feature_map.model_id,
                normalized=True,
            )
        )
    return tuple(embeddings)
