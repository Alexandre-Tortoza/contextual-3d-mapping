"""Contracts de geometria em espaço de imagem e invariantes de coordenadas.

Issue: #155.

Convenção de coordenadas (vinculante para todo o módulo):

- a origem do pixel ``(0, 0)`` é o canto superior esquerdo da imagem;
- ``x`` aumenta para a direita, ``y`` aumenta para baixo;
- uma :class:`BoundingBox` é armazenada como ``(x_min, y_min, x_max, y_max)``
  em um intervalo semi-aberto: ``x_min``/``y_min`` são inclusivos,
  ``x_max``/``y_max`` são exclusivos, combinando com o slicing
  ``array[y_min:y_max, x_min:x_max]``;
- uma :class:`Mask` é um array booleano com shape ``(height, width)``
  alinhado a uma resolução de imagem específica registrada na própria
  máscara.

Toda região emitida por region discovery, tiling, merge, ou serialização
deve satisfazer esses invariantes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Representa uma caixa retangular de pixels alinhada aos eixos, em intervalo
# semi-aberto. Existe como a representação canônica de bounding box usada em
# todo o módulo, garantindo que toda largura/altura seja sempre positiva.
@dataclass(frozen=True)
class BoundingBox:
    """Uma caixa de pixel semi-aberta e alinhada aos eixos: ``[x_min, x_max) x [y_min, y_max)``."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # Garante que a caixa tenha largura e altura positivas, rejeitando
    # caixas degeneradas ou invertidas na criação.
    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                "BoundingBox must have positive width and height: "
                f"({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})."
            )

    # Calcula a largura da caixa a partir de x_min/x_max, evitando que
    # consumidores dupliquem essa subtração.
    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    # Calcula a altura da caixa a partir de y_min/y_max, evitando que
    # consumidores dupliquem essa subtração.
    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    # Recorta a caixa aos limites de uma imagem de resolução dada. Usada
    # quando uma caixa remapeada de um tile/crop pode extrapolar a imagem
    # original.
    def clipped_to(self, *, width: int, height: int) -> BoundingBox:
        """Recorta a caixa para uma imagem da resolução dada."""
        x_min = max(0.0, min(self.x_min, width))
        y_min = max(0.0, min(self.y_min, height))
        x_max = max(0.0, min(self.x_max, width))
        y_max = max(0.0, min(self.y_max, height))
        return BoundingBox(x_min, y_min, x_max, y_max)


# Representa uma máscara booleana de ocupação alinhada a uma resolução de
# imagem específica. Existe como a representação canônica de máscara usada
# por region discovery, merge e refinamento.
@dataclass(frozen=True, eq=False)
class Mask:
    """Uma máscara booleana de ocupação alinhada a uma resolução de imagem específica."""

    data: np.ndarray
    image_width: int
    image_height: int

    # Valida que o dtype é bool e que o shape do array bate com
    # (image_height, image_width), para que toda Mask seja internamente
    # consistente com a resolução que ela declara.
    def __post_init__(self) -> None:
        if self.data.dtype != np.bool_:
            raise ValueError(f"Mask dtype must be bool, got {self.data.dtype}.")
        if self.data.shape != (self.image_height, self.image_width):
            raise ValueError(
                "Mask shape must be (image_height, image_width) = "
                f"({self.image_height}, {self.image_width}), got {self.data.shape}."
            )

    # Compara duas máscaras por conteúdo (resolução + dados), já que
    # dataclass(eq=False) desativa a comparação automática por causa do
    # array numpy não hasheável/comparável por padrão.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return (
            self.image_width == other.image_width
            and self.image_height == other.image_height
            and bool(np.array_equal(self.data, other.data))
        )

    # Indica se a máscara não ocupa nenhum pixel. Usada por operações que
    # não fazem sentido em uma máscara vazia (ex: bounding_box).
    @property
    def is_empty(self) -> bool:
        return not bool(self.data.any())

    # Calcula a bounding box justa dos pixels ocupados. Usada para derivar
    # uma BoundingBox a partir de uma Mask quando só a máscara está
    # disponível.
    def bounding_box(self) -> BoundingBox:
        """Calcula a bounding box justa (tight) dos pixels ocupados."""
        if self.is_empty:
            raise ValueError("Cannot compute a bounding box for an empty mask.")
        rows = np.any(self.data, axis=1)
        cols = np.any(self.data, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return BoundingBox(float(x_min), float(y_min), float(x_max) + 1.0, float(y_max) + 1.0)

    # Conta os pixels ocupados. Usada, por exemplo, como denominador em
    # containment_ratio.
    def area(self) -> int:
        return int(self.data.sum())

    # Calcula a intersecção-sobre-união com outra máscara da mesma
    # resolução. Usada por merge/refinamento de região para decidir se duas
    # propostas se sobrepõem o suficiente para serem fundidas.
    def iou(self, other: Mask) -> float:
        """Intersection-over-union com outra máscara da mesma resolução."""
        self._require_compatible(other)
        intersection = np.logical_and(self.data, other.data).sum()
        union = np.logical_or(self.data, other.data).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    # Calcula a fração de ``other`` coberta por ``self``. Usada para decidir
    # se uma região está contida em outra (diferente de IoU, que penaliza
    # tamanhos diferentes).
    def containment_ratio(self, other: Mask) -> float:
        """Fração de ``other`` coberta por ``self`` (0 quando ``other`` está vazia)."""
        self._require_compatible(other)
        other_area = other.area()
        if other_area == 0:
            return 0.0
        intersection = np.logical_and(self.data, other.data).sum()
        return float(intersection) / float(other_area)

    # Garante que duas máscaras compartilham a mesma resolução antes de
    # compará-las pixel a pixel; evita comparações sem sentido entre
    # máscaras de imagens diferentes.
    def _require_compatible(self, other: Mask) -> None:
        if (self.image_width, self.image_height) != (other.image_width, other.image_height):
            raise ValueError("Masks must share the same image resolution to be compared.")


# Representa um mapeamento afim entre um frame local (tile, crop, resize) e a
# imagem original. Existe para que boxes/máscaras calculadas em coordenadas
# locais de tile possam ser remapeadas para coordenadas globais da imagem.
@dataclass(frozen=True)
class CoordinateTransform:
    """Um mapeamento afim entre um frame local (tile, crop, resize) e a
    imagem original, expresso como ``global = local * scale + offset``.
    """

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    # Garante que os fatores de escala sejam positivos, rejeitando
    # transforms degenerados que inverteriam ou colapsariam coordenadas.
    def __post_init__(self) -> None:
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("CoordinateTransform scale factors must be positive.")

    # Constrói o transform identidade (sem escala nem offset). Usada como
    # valor default para regiões que não passaram por tiling/resize.
    @staticmethod
    def identity() -> CoordinateTransform:
        return CoordinateTransform(1.0, 1.0, 0.0, 0.0)

    # Remapeia uma BoundingBox de coordenadas locais para coordenadas
    # globais da imagem, aplicando escala e offset.
    def box_to_global(self, box: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x_min=box.x_min * self.scale_x + self.offset_x,
            y_min=box.y_min * self.scale_y + self.offset_y,
            x_max=box.x_max * self.scale_x + self.offset_x,
            y_max=box.y_max * self.scale_y + self.offset_y,
        )

    # Remapeia uma BoundingBox de coordenadas globais da imagem de volta
    # para coordenadas locais; inverso de box_to_global.
    def box_to_local(self, box: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x_min=(box.x_min - self.offset_x) / self.scale_x,
            y_min=(box.y_min - self.offset_y) / self.scale_y,
            x_max=(box.x_max - self.offset_x) / self.scale_x,
            y_max=(box.y_max - self.offset_y) / self.scale_y,
        )

    # Posiciona uma máscara local em um canvas em tamanho real, em
    # coordenadas globais. Usada para reconstruir a máscara de uma região
    # detectada em um tile/crop sobre a imagem completa.
    def mask_to_global(self, mask: Mask, *, global_width: int, global_height: int) -> Mask:
        """Posiciona uma máscara local em um canvas de tamanho real, em coordenadas globais.

        Usa posicionamento nearest-neighbor, que é exato para offsets de tile
        inteiros/scale=1 (tiling) e aproximado para escala não unitária
        (resize).
        """
        canvas = np.zeros((global_height, global_width), dtype=np.bool_)
        local_ys, local_xs = np.where(mask.data)
        if local_ys.size == 0:
            return Mask(canvas, global_width, global_height)
        global_xs = np.clip(
            np.round(local_xs * self.scale_x + self.offset_x).astype(np.int64),
            0,
            global_width - 1,
        )
        global_ys = np.clip(
            np.round(local_ys * self.scale_y + self.offset_y).astype(np.int64),
            0,
            global_height - 1,
        )
        canvas[global_ys, global_xs] = True
        return Mask(canvas, global_width, global_height)
