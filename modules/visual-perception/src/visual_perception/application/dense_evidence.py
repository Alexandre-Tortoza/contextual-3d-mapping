"""Elevação de features densas para resolução de pixel.

Issues: #191 (contract de evidência densa), #192 (benchmark de upsampling).

Um backbone como o DINOv2 devolve uma grade de patches: cada célula cobre
14x14 pixels ou mais da imagem original. Regiões pequenas, bordas e
estruturas finas caem inteiras dentro de uma célula e perdem qualquer
detalhe espacial. Elevar a grade para resolução de pixel recupera esse
detalhe, ao custo de memória proporcional a ``largura * altura * dimensão``.

Existem dois caminhos, e eles servem a propósitos diferentes:

- :func:`upsample_feature_map` materializa o mapa pixel-aligned completo.
  É o que a #191 exige como representação distinguível e o que o benchmark
  #192 compara, mas em 8 GB de VRAM ele só cabe em resoluções modestas —
  por isso a materialização é protegida por um limite explícito de bytes;
- :func:`~visual_perception.domain.feature_map.sample_feature_map` amostra
  as mesmas coordenadas sob demanda, sem materializar nada. É o caminho que
  o pooling mask-aware usa em produção, porque uma região cobre uma fração
  pequena da imagem e materializar o resto seria desperdício.

Os dois produzem os mesmos valores para as mesmas coordenadas: a diferença
é apenas de custo de memória, o que o benchmark #192 mede.
"""

from __future__ import annotations

import numpy as np

from visual_perception.domain.feature_map import (
    FeatureMap,
    FeatureRepresentation,
    SamplingRule,
    sample_feature_map,
)

#: Nome do caminho que deixa a grade de patches como está. É o baseline
#: contra o qual a #192 compara os caminhos pixel-aligned.
PATCH_GRID = "patch_grid"

#: Caminhos que produzem um mapa pixel-aligned, nomeados pela regra de
#: amostragem usada em cada pixel.
NEAREST = "nearest"
BILINEAR = "bilinear"

_UPSAMPLING_METHODS = {NEAREST: SamplingRule.NEAREST, BILINEAR: SamplingRule.BILINEAR}

#: Teto default de memória para materializar um mapa pixel-aligned. Em uma
#: RTX 3060 de 8 GB, um mapa DINOv2-base (768 dims) a 1280x720 em float32
#: pediria ~2,8 GB só para os valores; recusar explicitamente é melhor do
#: que derrubar o processo por falta de memória no meio de um benchmark.
DEFAULT_MAX_UPSAMPLED_BYTES = 1 << 30


# Estima quantos bytes a materialização de um mapa pixel-aligned ocuparia.
# Existe para que chamadores (benchmark #192, pipeline) decidam antes de
# alocar, em vez de descobrir o custo ao estourar a memória.
def upsampled_bytes(feature_map: FeatureMap, width: int, height: int, *, itemsize: int = 4) -> int:
    """Retorna o tamanho em bytes de um mapa pixel-aligned ``width x height``.

    Argumentos:
        feature_map: o mapa de origem, que define a dimensão do vetor.
        width: largura alvo em pixels.
        height: altura alvo em pixels.
        itemsize: bytes por componente do vetor de feature.
    Retorna:
        o número de bytes que o array denso ocuparia.
    """
    return int(width) * int(height) * int(feature_map.dimension) * int(itemsize)


# Materializa um FeatureMap pixel-aligned a partir de uma grade de patches.
# Existe porque a #191 exige que os dois níveis de resolução sejam
# representações distinguíveis e serializáveis, e porque a #192 precisa
# comparar o custo real de cada caminho. A construção é feita linha a linha
# para que apenas o array de saída, e não uma malha intermediária completa,
# fique residente.
def upsample_feature_map(
    feature_map: FeatureMap,
    *,
    method: str,
    width: int,
    height: int,
    max_bytes: int = DEFAULT_MAX_UPSAMPLED_BYTES,
) -> FeatureMap:
    """Eleva ``feature_map`` para um mapa pixel-aligned de ``width x height``.

    Argumentos:
        feature_map: grade de patches produzida pelo backbone.
        method: ``"nearest"`` ou ``"bilinear"``.
        width: largura alvo em pixels da imagem original.
        height: altura alvo em pixels da imagem original.
        max_bytes: teto de memória para o array resultante.
    Retorna:
        um :class:`FeatureMap` com ``representation`` ``pixel_aligned`` e
        stride unitário, carregando a máscara de suporte válido.
    Levanta:
        ValueError: se o método for desconhecido, as dimensões não forem
            positivas, ou a materialização exceder ``max_bytes``.
    """
    rule = _UPSAMPLING_METHODS.get(method)
    if rule is None:
        raise ValueError(
            f"Unknown upsampling method {method!r}, expected one of {sorted(_UPSAMPLING_METHODS)}."
        )
    if width <= 0 or height <= 0:
        raise ValueError("Upsampling target dimensions must be positive.")
    required = upsampled_bytes(feature_map, width, height)
    if required > max_bytes:
        raise ValueError(
            f"Materializing a {width}x{height}x{feature_map.dimension} pixel-aligned feature map needs "
            f"{required / 1e9:.2f} GB, above the {max_bytes / 1e9:.2f} GB budget. Sample the coordinates "
            "you actually need with sample_feature_map instead of materializing the full map."
        )

    data = np.empty((height, width, feature_map.dimension), dtype=np.float32)
    support = np.empty((height, width), dtype=np.bool_)
    columns = np.arange(width, dtype=np.float64) + 0.5
    for row in range(height):
        rows = np.full(width, row + 0.5, dtype=np.float64)
        values, valid = sample_feature_map(feature_map, columns, rows, interpolation=rule)
        data[row] = values.astype(np.float32, copy=False)
        support[row] = valid

    return FeatureMap(
        data=data,
        stride_x=1.0,
        stride_y=1.0,
        dimension=feature_map.dimension,
        model_id=feature_map.model_id,
        representation=FeatureRepresentation.PIXEL_ALIGNED,
        interpolation=rule,
        checkpoint=feature_map.checkpoint,
        preprocessing=feature_map.preprocessing,
        upsampling_method=method,
        valid_support=support,
    )
