"""Geometria declarada das áreas de imagem que limitam a evidência visual.

Issue: #202. Duas áreas limitam o que, num frame, é evidência utilizável:

- a **área válida** do sensor — em ``corridor-02``, o círculo útil da lente
  fisheye, fora do qual os pixels são o corpo preto da objetiva;
- a área ocupada pelo **ego-veículo** — a silhueta do rig, fixa em todo frame.

A geometria é *declarada e versionada*, nunca inferida por limiar de cor em
runtime. Os intrínsecos MEI de ``corridor-02`` (ξ = 1,563) não delimitam o anel
preto: tanto o limite de FOV quanto o de desprojeção excedem a diagonal da
imagem, de modo que a vinheta é óptica e não derivável da calibração. A
alternativa determinística é esta: uma forma medida uma vez, versionada como
artifact e rasterizada igual em toda execução.

Aplicar uma área **nunca** altera os pixels de origem. A #212 mediu o custo de
fazer isso: zerar a faixa do rig antes do SAM corrompia a análise de cena e
colapsava 45 de 45 regiões em ``curved wall``. A exclusão acontece na
filtragem de proposals, sobre a máscara — ver
``application/proposal_filtering.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from visual_perception.domain.geometry import BoundingBox, Mask

__all__ = ["CircleArea", "ImageAreaGeometry", "ImageAreaMasks"]

#: Um vértice de polígono em coordenadas de imagem ``(x, y)``.
Vertex = tuple[float, float]


# Descreve um disco em coordenadas de imagem. Existe porque a área útil de uma
# lente é bem aproximada por um círculo, e três números versionados são
# auditáveis num diff de forma que uma máscara binária não é.
@dataclass(frozen=True)
class CircleArea:
    """Um disco fechado em coordenadas de imagem.

    Argumentos:
        center_x: coordenada horizontal do centro, em pixels.
        center_y: coordenada vertical do centro, em pixels.
        radius: raio em pixels; um pixel exatamente sobre ele pertence à área.
    """

    center_x: float
    center_y: float
    radius: float

    # Rejeita um raio não positivo, que não descreve disco nenhum e produziria
    # uma área vazia silenciosa.
    def __post_init__(self) -> None:
        """Valida que o raio é positivo."""
        if self.radius <= 0.0:
            raise ValueError(f"CircleArea.radius must be positive, got {self.radius}.")


# Agrupa as formas que compõem uma área de imagem e sabe rasterizá-las. Existe
# como o contract único de "que parte do frame é esta área", consumido pela
# configuração (que a declara) e pela filtragem de proposals (que a aplica).
@dataclass(frozen=True)
class ImageAreaGeometry:
    """Uma área de imagem declarada como união de um círculo e/ou polígonos.

    Argumentos:
        circle: disco opcional, tipicamente a área útil de uma lente.
        polygons: polígonos opcionais, tipicamente a silhueta de um rig.
    """

    circle: CircleArea | None = None
    polygons: tuple[tuple[Vertex, ...], ...] = field(default_factory=tuple)

    # Exige ao menos uma forma e polígonos com interior. Uma geometria vazia
    # rasterizaria como máscara toda falsa e, usada como área válida,
    # rejeitaria o frame inteiro sem que nada no artifact revelasse a causa.
    def __post_init__(self) -> None:
        """Valida que a geometria declara ao menos uma forma com interior."""
        if self.circle is None and not self.polygons:
            raise ValueError(
                "ImageAreaGeometry must declare at least one shape: a circle, a polygon, or both."
            )
        for polygon in self.polygons:
            if len(polygon) < 3:
                raise ValueError(
                    f"An ImageAreaGeometry polygon needs at least three vertices, got {len(polygon)}."
                )

    # Converte a geometria declarada na máscara booleana daquela resolução.
    # Reutiliza o Mask do domínio em vez de devolver um ndarray solto, para que
    # a área componha com as máscaras de proposal pelas mesmas operações.
    def rasterize(self, width: int, height: int) -> Mask:
        """Rasteriza a área na resolução pedida, como união das formas declaradas.

        Argumentos:
            width: largura da imagem em pixels.
            height: altura da imagem em pixels.
        Retorna:
            a máscara booleana da área, na resolução pedida.
        Levanta:
            ValueError: se a resolução não for positiva.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Image resolution must be positive, got {width}x{height}.")
        rows, columns = np.mgrid[0:height, 0:width]
        inside = np.zeros((height, width), dtype=np.bool_)
        if self.circle is not None:
            distance = np.hypot(columns - self.circle.center_x, rows - self.circle.center_y)
            inside |= distance <= self.circle.radius
        for polygon in self.polygons:
            inside |= _polygon_mask(polygon, columns, rows)
        return Mask(inside, width, height)


# Agrupa as máscaras já rasterizadas de um frame. Existe para que os estágios
# que precisam das duas áreas — filtragem de proposals, contexto de cena e
# diagnóstico — recebam um único objeto, em vez de dois parâmetros opcionais
# que podem ser trocados de posição sem o tipo reclamar.
@dataclass(frozen=True)
class ImageAreaMasks:
    """As máscaras de área válida e de ego-veículo de um frame.

    Ambas são opcionais: uma sequência sem geometria declarada roda sem
    exclusão nenhuma, e o diagnóstico registra essa ausência em vez de fingir
    que a filtragem foi aplicada.

    Argumentos:
        valid_area: área utilizável do sensor, ou ``None`` se não declarada.
        ego_vehicle: área ocupada pelo rig, ou ``None`` se não declarada.
    """

    valid_area: Mask | None = None
    ego_vehicle: Mask | None = None

    # Indica se alguma exclusão foi de fato declarada, para que o chamador não
    # precise testar os dois campos separadamente.
    @property
    def is_empty(self) -> bool:
        """Indica que nenhuma das duas áreas foi declarada."""
        return self.valid_area is None and self.ego_vehicle is None

    # Calcula a janela da imagem que é cena analisável: dentro do sensor e fora
    # do rig. Existe para que a análise de cena receba uma *view* recortada em
    # vez de a imagem inteira — o modelo não pode inventariar um objeto que não
    # recebeu, e recortar não altera pixel nenhum, ao contrário de pintar.
    def scene_box(self) -> BoundingBox | None:
        """Retorna a caixa da área analisável como cena, ou ``None`` sem áreas.

        Retorna:
            a menor caixa que contém a área válida menos a área do ego, ou
            ``None`` quando nada foi declarado ou nada sobra.
        """
        if self.is_empty:
            return None
        reference = self.valid_area or self.ego_vehicle
        assert reference is not None
        analysable = (
            np.ones((reference.image_height, reference.image_width), dtype=np.bool_)
            if self.valid_area is None
            else self.valid_area.data.copy()
        )
        if self.ego_vehicle is not None:
            analysable &= ~self.ego_vehicle.data
        rows, columns = np.nonzero(analysable)
        if rows.size == 0:
            return None
        return BoundingBox(
            x_min=float(columns.min()),
            y_min=float(rows.min()),
            x_max=float(columns.max() + 1),
            y_max=float(rows.max() + 1),
        )


# Rasteriza um polígono pela regra de cruzamento (even-odd), vetorizada sobre a
# grade de pixels. Existe local ao domínio para que a geometria não dependa de
# uma biblioteca de imagem só para desenhar um polígono; chamada por
# ImageAreaGeometry.rasterize.
def _polygon_mask(
    polygon: tuple[Vertex, ...], columns: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Retorna a máscara booleana do interior do polígono na grade dada."""
    inside = np.zeros(columns.shape, dtype=np.bool_)
    count = len(polygon)
    for index in range(count):
        x_start, y_start = polygon[index]
        x_end, y_end = polygon[(index + 1) % count]
        # Conta os cruzamentos da semirreta horizontal que parte de cada pixel:
        # um número ímpar de arestas cruzadas significa interior.
        straddles = (y_start > rows) != (y_end > rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection = (x_end - x_start) * (rows - y_start) / (y_end - y_start) + x_start
        inside ^= straddles & (columns < intersection)
    return inside
