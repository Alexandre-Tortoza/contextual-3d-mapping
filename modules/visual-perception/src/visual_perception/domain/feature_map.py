"""Contract do mapa de features visuais denso e alinhado a pixel.

Issues: #161 (grade de features densa), #191 (evidência densa pixel-aligned).

A #161 definiu a grade espacial mínima que o pooling mask-aware precisa. A
#191 completa esse contract com o que um consumidor precisa para amarrar um
valor de feature de volta à sua coordenada na imagem *original*:

- ``representation`` distingue uma grade grosseira de patches
  (``patch_grid``) de um mapa já elevado para resolução de pixel
  (``pixel_aligned``), para que os dois nunca sejam confundidos;
- ``interpolation`` declara a regra de amostragem válida para este mapa;
- ``stride_*`` mais ``origin_*`` formam um :class:`CoordinateTransform`
  invertível entre coordenada de grade e coordenada da imagem original,
  cobrindo resize, crop e tiling;
- ``valid_support`` marca explicitamente quais células têm suporte real,
  para que borda e área não suportada sejam identificáveis em vez de
  silenciosamente extrapoladas;
- ``checkpoint``, ``preprocessing`` e ``upsampling_method`` preservam a
  proveniência numérica do backbone que produziu os valores.

O array denso em si nunca é serializado inline (ver
``docs/artifacts.md``): :func:`feature_map_spec_to_dict` serializa apenas a
metadata, e os valores viajam por referência de artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from visual_perception.domain.geometry import CoordinateTransform


# Distingue uma grade grosseira de patches de um mapa já elevado para
# resolução de pixel. Existe porque a #191 exige que os dois permaneçam
# distinguíveis: eles têm custo, resolução e garantias de suporte
# diferentes, e um benchmark (#192) que os comparasse sem poder separá-los
# não teria como atribuir um resultado a um caminho.
class FeatureRepresentation(StrEnum):
    """Qual é a natureza espacial de um :class:`FeatureMap`."""

    PATCH_GRID = "patch_grid"
    ELEVATED_GRID = "elevated_grid"
    PIXEL_ALIGNED = "pixel_aligned"


# Registra como os valores espaciais foram produzidos, separadamente de sua
# resolução final. Existe para impedir que interpolação geométrica seja
# reportada como detalhe aprendido ou extraído nativamente (#208).
class FeatureGeneration(StrEnum):
    """Origem dos valores de um :class:`FeatureMap`."""

    BACKBONE_NATIVE = "backbone_native"
    RESAMPLED = "resampled"
    LEARNED_UPSAMPLER = "learned_upsampler"


# Declara qual regra de amostragem é válida para ler um mapa. Existe para
# que a interpolação seja parte do contract (e do report de benchmark), e
# não uma escolha implícita do consumidor.
class SamplingRule(StrEnum):
    """Como uma coordenada de imagem é convertida em um vetor de feature."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"


# Representa uma grade espacial de features visuais densas alinhada a uma
# imagem, junto com a metadata que permite mapear qualquer coordenada da
# imagem original para o suporte de feature correspondente. Existe para dar
# ao pooling mask-aware (#162) o stride de que ele precisa e ao consumidor
# de evidência densa (#191/#194) uma proveniência completa.
@dataclass(frozen=True, eq=False)
class FeatureMap:
    """Uma grade espacial de features visuais densas alinhada a uma imagem.

    ``stride_x``/``stride_y`` descrevem quantos pixels da imagem original
    uma célula da grade cobre; ``origin_x``/``origin_y`` descrevem onde a
    grade começa na imagem original (diferente de zero quando o mapa veio de
    um tile ou crop). Juntos eles formam :attr:`transform`.
    """

    data: np.ndarray  # shape (grid_height, grid_width, dimension)
    stride_x: float
    stride_y: float
    dimension: int
    model_id: str
    representation: FeatureRepresentation = FeatureRepresentation.PATCH_GRID
    interpolation: SamplingRule = SamplingRule.NEAREST
    origin_x: float = 0.0
    origin_y: float = 0.0
    checkpoint: str | None = None
    preprocessing: str | None = None
    upsampling_method: str | None = None
    generation: FeatureGeneration = FeatureGeneration.BACKBONE_NATIVE
    source_grid_width: int | None = None
    source_grid_height: int | None = None
    upsampler_checkpoint: str | None = None
    upsampler_checkpoint_digest: str | None = None
    fallback_reason: str | None = None
    valid_support: np.ndarray | None = None  # shape (grid_height, grid_width), bool

    # Valida shape, consistência de dimensão, strides positivos, coerência
    # da metadata de representação e o shape/dtype do suporte, para que
    # consumidores downstream possam confiar cegamente no FeatureMap.
    def __post_init__(self) -> None:
        """Valida geometria, proveniência e suporte declarados pelo mapa."""
        if self.data.ndim != 3:
            raise ValueError(f"FeatureMap.data must have shape (H, W, D), got {self.data.shape}.")
        if self.data.shape[2] != self.dimension:
            raise ValueError(
                f"FeatureMap dimension mismatch: declared {self.dimension}, "
                f"got last axis {self.data.shape[2]}."
            )
        if self.stride_x <= 0 or self.stride_y <= 0:
            raise ValueError("FeatureMap strides must be positive.")
        if not np.isfinite(self.data).all():
            raise ValueError("FeatureMap.data must be finite (no NaN/Inf).")
        if self.representation is FeatureRepresentation.PIXEL_ALIGNED and (
            self.stride_x != 1.0 or self.stride_y != 1.0
        ):
            raise ValueError("A pixel_aligned FeatureMap must have unit strides.")
        if self.representation is FeatureRepresentation.PIXEL_ALIGNED and self.upsampling_method is None:
            raise ValueError("A pixel_aligned FeatureMap must record its upsampling_method.")
        if self.generation is FeatureGeneration.LEARNED_UPSAMPLER:
            if self.representation is not FeatureRepresentation.ELEVATED_GRID:
                raise ValueError("A learned upsampler must produce an elevated_grid FeatureMap.")
            if (
                not self.upsampling_method
                or not self.upsampler_checkpoint
                or not self.upsampler_checkpoint_digest
            ):
                raise ValueError(
                    "A learned upsampler must record method, checkpoint and checkpoint digest."
                )
            if not self.source_grid_width or not self.source_grid_height:
                raise ValueError("A learned upsampler must record its positive source grid resolution.")
        if self.valid_support is not None:
            if self.valid_support.dtype != np.bool_:
                raise ValueError(
                    f"FeatureMap.valid_support dtype must be bool, got {self.valid_support.dtype}."
                )
            if self.valid_support.shape != self.data.shape[:2]:
                raise ValueError(
                    "FeatureMap.valid_support shape must match the feature grid "
                    f"{self.data.shape[:2]}, got {self.valid_support.shape}."
                )

    # Expõe a altura da grade de features sem expor o array numpy bruto
    # diretamente.
    @property
    def grid_height(self) -> int:
        """Número de células da grade no eixo vertical."""
        return int(self.data.shape[0])

    # Expõe a largura da grade de features sem expor o array numpy bruto
    # diretamente.
    @property
    def grid_width(self) -> int:
        """Número de células da grade no eixo horizontal."""
        return int(self.data.shape[1])

    # Deriva o transform invertível entre coordenada de grade e coordenada
    # da imagem original. Existe para que crop/tiling/resize sejam
    # representados por um único objeto já existente no módulo (#155), em
    # vez de aritmética de stride replicada em cada consumidor.
    @property
    def transform(self) -> CoordinateTransform:
        """O mapeamento afim ``imagem = grade * stride + origin``."""
        return CoordinateTransform(self.stride_x, self.stride_y, self.origin_x, self.origin_y)

    # Descreve a área da imagem original efetivamente coberta por esta
    # grade. Usada para decidir se uma coordenada está fora do mapa antes
    # mesmo de consultar valid_support.
    @property
    def covered_region(self) -> tuple[float, float, float, float]:
        """A caixa semi-aberta ``(x_min, y_min, x_max, y_max)`` coberta na imagem original."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.grid_width * self.stride_x,
            self.origin_y + self.grid_height * self.stride_y,
        )

    # Resume, em um único número, quanta da grade tem suporte real. Existe
    # para que a qualidade de *suporte de feature* seja reportável separada
    # da confiança semântica, como a #191 exige.
    @property
    def support_ratio(self) -> float:
        """Fração de células com suporte válido (``1.0`` quando não há máscara de suporte)."""
        if self.valid_support is None:
            return 1.0
        return float(self.valid_support.mean())


# Amostra o mapa denso em coordenadas da imagem *original*, devolvendo os
# vetores e uma máscara de validade explícita. Existe como a única forma
# pública de ler um FeatureMap sem depender de tensors privados do backend
# (#191); usada pelo pooling mask-aware (#162/#194) e pelo benchmark #192.
def sample_feature_map(
    feature_map: FeatureMap,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    interpolation: SamplingRule | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra ``feature_map`` nas coordenadas de imagem ``(xs, ys)``.

    Coordenadas fora da área coberta, sem vizinhança completa (no caso
    bilinear) ou sem suporte válido nunca são extrapoladas: elas voltam
    marcadas como inválidas, com vetor zero.

    Argumentos:
        feature_map: o mapa denso a ser lido.
        xs: coordenadas ``x`` em pixels da imagem original.
        ys: coordenadas ``y`` em pixels da imagem original.
        interpolation: regra de amostragem; o default é a declarada no mapa.
    Retorna:
        ``(values, valid)`` com shapes ``(N, dimension)`` e ``(N,)``.
    Levanta:
        ValueError: se ``xs`` e ``ys`` não tiverem o mesmo shape unidimensional.
    """
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("sample_feature_map expects xs and ys to be 1-D arrays of the same length.")
    rule = interpolation or feature_map.interpolation
    # Centro da célula (i, j) fica em origin + (i + 0.5) * stride, então a
    # coordenada contínua de grade é a inversa exata dessa relação.
    grid_x = (xs.astype(np.float64) - feature_map.origin_x) / feature_map.stride_x - 0.5
    grid_y = (ys.astype(np.float64) - feature_map.origin_y) / feature_map.stride_y - 0.5
    if rule is SamplingRule.NEAREST:
        return _sample_nearest(feature_map, grid_x, grid_y)
    return _sample_bilinear(feature_map, grid_x, grid_y)


# Implementa a amostragem nearest-neighbour: arredonda para a célula mais
# próxima e rejeita o que cai fora da grade ou sem suporte. Helper interno
# de sample_feature_map.
def _sample_nearest(
    feature_map: FeatureMap, grid_x: np.ndarray, grid_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra a célula mais próxima, sem extrapolar para fora da grade."""
    columns = np.rint(grid_x).astype(np.int64)
    rows = np.rint(grid_y).astype(np.int64)
    valid = (
        (columns >= 0)
        & (columns < feature_map.grid_width)
        & (rows >= 0)
        & (rows < feature_map.grid_height)
    )
    safe_columns = np.clip(columns, 0, feature_map.grid_width - 1)
    safe_rows = np.clip(rows, 0, feature_map.grid_height - 1)
    if feature_map.valid_support is not None:
        valid &= feature_map.valid_support[safe_rows, safe_columns]
    values = feature_map.data[safe_rows, safe_columns].astype(np.float64, copy=True)
    values[~valid] = 0.0
    return values, valid


# Implementa a amostragem bilinear: combina as quatro células vizinhas e
# invalida coordenadas da meia-célula de borda, que não têm vizinhança
# completa. Helper interno de sample_feature_map.
def _sample_bilinear(
    feature_map: FeatureMap, grid_x: np.ndarray, grid_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpola as quatro células vizinhas, invalidando a borda sem vizinhança completa."""
    left = np.floor(grid_x).astype(np.int64)
    top = np.floor(grid_y).astype(np.int64)
    valid = (
        (left >= 0)
        & (left + 1 < feature_map.grid_width)
        & (top >= 0)
        & (top + 1 < feature_map.grid_height)
    )
    safe_left = np.clip(left, 0, feature_map.grid_width - 2) if feature_map.grid_width > 1 else left * 0
    safe_top = np.clip(top, 0, feature_map.grid_height - 2) if feature_map.grid_height > 1 else top * 0
    safe_right = np.minimum(safe_left + 1, feature_map.grid_width - 1)
    safe_bottom = np.minimum(safe_top + 1, feature_map.grid_height - 1)

    weight_x = np.clip(grid_x - safe_left, 0.0, 1.0)[:, None]
    weight_y = np.clip(grid_y - safe_top, 0.0, 1.0)[:, None]
    data = feature_map.data.astype(np.float64, copy=False)
    top_row = data[safe_top, safe_left] * (1.0 - weight_x) + data[safe_top, safe_right] * weight_x
    bottom_row = (
        data[safe_bottom, safe_left] * (1.0 - weight_x) + data[safe_bottom, safe_right] * weight_x
    )
    values = top_row * (1.0 - weight_y) + bottom_row * weight_y

    if feature_map.valid_support is not None:
        support = feature_map.valid_support
        valid &= (
            support[safe_top, safe_left]
            & support[safe_top, safe_right]
            & support[safe_bottom, safe_left]
            & support[safe_bottom, safe_right]
        )
    values = np.asarray(values, dtype=np.float64)
    values[~valid] = 0.0
    return values, valid


# Serializa apenas a metadata de um FeatureMap (nunca o array denso), para
# que resolução, interpolação, transform e validade sejam persistíveis e
# validáveis junto de uma referência de artifact. Usada por quem grava
# evidência densa (#191) e pelo report de benchmark (#192).
def feature_map_spec_to_dict(feature_map: FeatureMap) -> dict[str, Any]:
    """Converte a metadata de ``feature_map`` em um dict serializável.

    Argumentos:
        feature_map: o mapa cuja metadata deve ser preservada.
    Retorna:
        dict com a especificação densa, sem nenhum valor de feature.
    """
    return {
        "representation": feature_map.representation.value,
        "interpolation": feature_map.interpolation.value,
        "grid_width": feature_map.grid_width,
        "grid_height": feature_map.grid_height,
        "dimension": feature_map.dimension,
        "stride_x": feature_map.stride_x,
        "stride_y": feature_map.stride_y,
        "origin_x": feature_map.origin_x,
        "origin_y": feature_map.origin_y,
        "model_id": feature_map.model_id,
        "checkpoint": feature_map.checkpoint,
        "preprocessing": feature_map.preprocessing,
        "upsampling_method": feature_map.upsampling_method,
        "generation": feature_map.generation.value,
        "source_grid_width": feature_map.source_grid_width,
        "source_grid_height": feature_map.source_grid_height,
        "upsampler_checkpoint": feature_map.upsampler_checkpoint,
        "upsampler_checkpoint_digest": feature_map.upsampler_checkpoint_digest,
        "fallback_reason": feature_map.fallback_reason,
        "dtype": str(feature_map.data.dtype),
        "support_ratio": feature_map.support_ratio,
        "has_valid_support": feature_map.valid_support is not None,
    }


# Reconstrói e valida a especificação densa a partir do dict — lado inverso
# de feature_map_spec_to_dict. Existe para que a metadata gravada seja
# relida com as mesmas invariantes, em vez de aceita como dict solto.
def feature_map_spec_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Valida e normaliza uma especificação densa lida de disco.

    Argumentos:
        payload: dict produzido por :func:`feature_map_spec_to_dict`.
    Retorna:
        a mesma especificação, com enums e números já validados.
    Levanta:
        ValueError: se algum campo obrigatório faltar ou for inválido.
    """
    required = ("representation", "interpolation", "grid_width", "grid_height", "dimension")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Dense feature spec is missing required fields: {missing}.")
    spec = dict(payload)
    spec["representation"] = FeatureRepresentation(payload["representation"]).value
    spec["interpolation"] = SamplingRule(payload["interpolation"]).value
    spec["generation"] = FeatureGeneration(
        payload.get("generation", FeatureGeneration.BACKBONE_NATIVE.value)
    ).value
    for key in ("grid_width", "grid_height", "dimension"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"Dense feature spec field {key!r} must be a positive integer.")
    for key in ("stride_x", "stride_y"):
        if key in spec and float(spec[key]) <= 0.0:
            raise ValueError(f"Dense feature spec field {key!r} must be positive.")
    return spec
