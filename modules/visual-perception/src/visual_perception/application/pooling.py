"""Pooling de região em alta resolução, consciente de máscara.

Issues: #162 (pooling mask-aware), #191 (amostragem de evidência densa),
#192 (comparação entre caminhos de resolução).

Três métodos de pooling são suportados atrás de uma única função:

- ``patch_grid_baseline``: inclui uma célula do grid de features apenas
  quando o *centro* do seu pixel cai dentro da máscara da região, depois
  faz a média sem peso. Simples, mas uma máscara menor que uma célula do
  grid não tem centro de célula dentro dela e é rejeitada. É o baseline
  contra o qual a #192 compara;
- ``pixel_nearest_highres``: reúne, para cada pixel da máscara, o vetor de
  feature da célula mais próxima, depois faz a média. Isso mantém regiões
  pequenas representáveis sempre que pelo menos um pixel da máscara tiver
  suporte de feature;
- ``pixel_bilinear_highres``: o mesmo, interpolando as quatro células
  vizinhas. Pixels na meia-célula de borda não têm vizinhança completa e
  são explicitamente descartados em vez de extrapolados.

Os dois caminhos pixel-aligned amostram o mapa denso sob demanda (ver
``application/dense_evidence.py``): eles nunca materializam o mapa em
resolução de pixel, porque uma região cobre uma fração pequena da imagem.

A geometria vem inteiramente do :class:`FeatureMap` (stride e origin), e
não das dimensões da máscara: um mapa extraído de um tile ou crop tem
origin diferente de zero, e recomputar o stride a partir da máscara
alinharia a região no lugar errado.

Os três caminhos normalizam L2 o vetor resultante (comportamento de
normalização documentado); uma normalização só é pulada quando
inalcançável (entrada finita e não-zero), o que não pode acontecer depois
que um método já aceitou uma máscara.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_perception.domain.embeddings import VisualEmbedding
from visual_perception.domain.feature_map import FeatureMap, SamplingRule, sample_feature_map
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion

BASELINE = "patch_grid_baseline"
HIGH_RESOLUTION = "pixel_nearest_highres"
PIXEL_BILINEAR = "pixel_bilinear_highres"
_METHODS = frozenset({BASELINE, HIGH_RESOLUTION, PIXEL_BILINEAR})

#: Como cada valor de ``FeatureExtractionConfig.upsampling`` se traduz em um
#: método de pooling. Mantido aqui para que a config selecione o caminho de
#: resolução em um único lugar, e a ablation da #192 não precise de um
#: switch paralelo.
_METHOD_BY_UPSAMPLING = {
    "patch_grid": BASELINE,
    "nearest": HIGH_RESOLUTION,
    "bilinear": PIXEL_BILINEAR,
}


# Agrupa o vetor agregado de uma região com quanto suporte de feature ele
# realmente teve. Existe porque a #191 exige que a qualidade do suporte seja
# reportável separada da confiança semântica: um vetor pooled a partir de
# 3% da máscara não é o mesmo dado que um pooled a partir de 100% dela.
@dataclass(frozen=True)
class PooledRegion:
    """O vetor agregado de uma região e a cobertura de suporte que o produziu."""

    vector: tuple[float, ...]
    support_ratio: float
    supported_pixels: int
    method: str

    # Valida a cobertura declarada, para que um consumidor possa usá-la
    # diretamente como sinal de qualidade sem reconferir a faixa.
    def __post_init__(self) -> None:
        """Rejeita coberturas fora de ``[0, 1]`` e contagens negativas."""
        if not 0.0 <= self.support_ratio <= 1.0:
            raise ValueError(f"PooledRegion.support_ratio must be in [0, 1], got {self.support_ratio}.")
        if self.supported_pixels < 0:
            raise ValueError("PooledRegion.supported_pixels must not be negative.")


# Traduz a configuração de amostragem densa no método de pooling
# correspondente. Existe para que a escolha de resolução seja feita pela
# config (#192) em vez de ficar fixa no pipeline; usada pelo pipeline
# canônico e pelo harness de ablation.
def pooling_method_for(upsampling: str) -> str:
    """Retorna o método de pooling correspondente a ``upsampling``.

    Argumentos:
        upsampling: valor de ``FeatureExtractionConfig.upsampling``.
    Retorna:
        o identificador de método aceito por :func:`pool_region_vector`.
    Levanta:
        ValueError: se ``upsampling`` não for um caminho conhecido.
    """
    method = _METHOD_BY_UPSAMPLING.get(upsampling)
    if method is None:
        raise ValueError(
            f"Unknown upsampling {upsampling!r}, expected one of {sorted(_METHOD_BY_UPSAMPLING)}."
        )
    return method


# Agrega as dense features de uma única região em um vetor final, finito e
# normalizado L2, junto com a cobertura de suporte medida. É o núcleo desta
# etapa; chamada por pool_region_vector e por pool_regions.
def pool_region_evidence(mask: Mask, feature_map: FeatureMap, method: str) -> PooledRegion:
    """Agrega as dense features de uma região e reporta quanto suporte havia.

    Argumentos:
        mask: máscara da região, alinhada à imagem original.
        feature_map: o mapa denso a ser amostrado.
        method: um dos três métodos declarados neste módulo.
    Retorna:
        o :class:`PooledRegion` com vetor normalizado e cobertura de suporte.
    Levanta:
        ValueError: se o método for desconhecido, a máscara estiver vazia,
            nenhum pixel tiver suporte, ou o vetor agregado tiver norma zero.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown pooling method {method!r}, expected one of {sorted(_METHODS)}.")
    if mask.is_empty:
        raise ValueError("Cannot pool an empty mask.")
    if method == BASELINE:
        vector, supported, total = _pool_patch_grid_baseline(mask, feature_map)
    else:
        rule = SamplingRule.NEAREST if method == HIGH_RESOLUTION else SamplingRule.BILINEAR
        vector, supported, total = _pool_sampled(mask, feature_map, rule)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Pooled vector has zero norm and cannot be normalized.")
    return PooledRegion(
        vector=tuple((vector / norm).tolist()),
        support_ratio=supported / total if total else 0.0,
        supported_pixels=supported,
        method=method,
    )


# Mantém a assinatura histórica desta etapa (#162) para os chamadores que
# só precisam do vetor. Delega a pool_region_evidence para que exista uma
# única implementação da agregação.
def pool_region_vector(mask: Mask, feature_map: FeatureMap, method: str) -> tuple[float, ...]:
    """Agrega as dense features de uma região em um único vetor finito e normalizado L2."""
    return pool_region_evidence(mask, feature_map, method).vector


# Implementa o método BASELINE: inclui uma célula do grid só quando o
# centro do seu pixel cai dentro da máscara, depois faz a média simples;
# helper interno de pool_region_evidence, mais barato porém mais grosseiro
# para regiões pequenas.
def _pool_patch_grid_baseline(mask: Mask, feature_map: FeatureMap) -> tuple[np.ndarray, int, int]:
    """Faz a média das células cujo centro cai dentro da máscara."""
    support = feature_map.valid_support
    included = []
    for grid_y in range(feature_map.grid_height):
        center_y = int(feature_map.origin_y + (grid_y + 0.5) * feature_map.stride_y)
        if not 0 <= center_y < mask.image_height:
            continue
        for grid_x in range(feature_map.grid_width):
            center_x = int(feature_map.origin_x + (grid_x + 0.5) * feature_map.stride_x)
            if not 0 <= center_x < mask.image_width or not mask.data[center_y, center_x]:
                continue
            if support is not None and not support[grid_y, grid_x]:
                continue
            included.append(feature_map.data[grid_y, grid_x])
    if not included:
        raise ValueError(
            "No feature-grid cell center falls inside the mask; the region cannot be aligned "
            f"with the '{BASELINE}' method. Try '{HIGH_RESOLUTION}' for small regions."
        )
    return np.mean(np.stack(included), axis=0), len(included), len(included)


# Implementa os métodos pixel-aligned: amostra o mapa denso em cada pixel da
# máscara com a regra pedida e faz a média apenas do que tem suporte válido.
# Helper interno de pool_region_evidence, preferido para regiões pequenas que
# o BASELINE rejeitaria.
def _pool_sampled(
    mask: Mask, feature_map: FeatureMap, rule: SamplingRule
) -> tuple[np.ndarray, int, int]:
    """Amostra o mapa denso em cada pixel da máscara, descartando o que não tem suporte."""
    ys, xs = np.where(mask.data)
    # O centro do pixel inteiro (x, y) fica em (x + 0.5, y + 0.5) na
    # convenção semi-aberta do módulo (ver domain/geometry.py).
    values, valid = sample_feature_map(
        feature_map, xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5, interpolation=rule
    )
    supported = int(valid.sum())
    if supported == 0:
        raise ValueError(
            f"No mask pixel has dense feature support under the '{rule.value}' sampling rule; "
            "the region falls outside the feature map's covered area."
        )
    return np.mean(values[valid], axis=0), supported, int(ys.size)


# Agrupa pool_region_evidence sobre todas as regiões de uma observação
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
        pooled = pool_region_evidence(region.mask, feature_map, method)
        embeddings.append(
            VisualEmbedding(
                embedding_id=f"visual-{region.region_id}",
                region_id=region.region_id,
                vector=pooled.vector,
                dimension=len(pooled.vector),
                pooling_method=method,
                feature_resolution=f"{feature_map.grid_width}x{feature_map.grid_height}",
                model_id=feature_map.model_id,
                normalized=True,
            )
        )
    return tuple(embeddings)
