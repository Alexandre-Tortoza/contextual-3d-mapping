"""Primitivas de desenho para inspeção visual de um frame.

Cada primitiva recebe e devolve uma ``Image``, de modo que elas compõem por
encadeamento e cada camada do pipeline pode ser desenhada isoladamente:
máscaras sem caixas, caixas sem labels, ou tudo junto. Isso é o que permite
comparar o estágio de proposals com o de regiões sem duplicar código de
desenho.

As primitivas não conhecem ``ObservedRegion`` nem ``RegionProposal`` — só
``DrawableShape``. O mapeamento dos tipos de domínio para essa forma comum vive
em ``render_overlay.py``. Nada aqui é contract público do módulo: é ferramenta
de inspeção local.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_perception.domain.geometry import BoundingBox, Mask

#: Número máximo de formas que recebem rótulo de texto. Com dezenas de regiões
#: (comum com SAM automático), rotular todas empilha texto ilegível; rotula-se
#: só as maiores, mas a máscara continua desenhada para todas.
MAX_LABELED_SHAPES = 20

try:
    _FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
except OSError:
    _FONT = ImageFont.load_default()

_PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (170, 110, 40),
]


# Forma desenhável comum a proposals e regiões. Existe porque os dois estágios
# têm a mesma geometria mas tipos diferentes, e sem essa adaptação o renderer
# precisaria ser duplicado — ou conhecer os dois tipos de domínio.
# Proposals preenchem apenas geometria: elas não têm semântica, e o artifact
# precisa mostrar isso em vez de fingir um label ausente.
@dataclass(frozen=True)
class DrawableShape:
    """Geometria e rótulo opcional de uma forma a desenhar sobre o frame.

    Argumentos:
        shape_id: identificador estável, usado para destacar formas escolhidas.
        mask: máscara booleana em resolução plena do frame.
        box: bounding box da forma.
        label: texto semântico, ou ``None`` quando a forma não tem semântica.
        semantic_confidence: confiança do label, ou ``None`` se não pontuado.
        geometric_confidence: confiança de máscara/box, ou ``None``.
    """

    shape_id: str
    mask: Mask
    box: BoundingBox
    label: str | None = None
    semantic_confidence: float | None = None
    geometric_confidence: float | None = None


# Escolhe a cor de uma forma pelo seu índice, ciclando a paleta. Centralizada
# aqui para que máscara, caixa e rótulo da mesma forma recebam a mesma cor em
# camadas desenhadas em chamadas separadas.
def color_for(index: int) -> tuple[int, int, int]:
    """Retorna a cor determinística da forma de índice informado."""
    return _PALETTE[index % len(_PALETTE)]


# Converte uma máscara booleana em imagem preto-e-branco. Usada para persistir
# máscaras de exclusão (ego-veículo) como artifact independente, sem alterar os
# pixels do frame.
def binary_mask_image(mask: np.ndarray) -> Image.Image:
    """Retorna a máscara booleana como imagem de 1 byte por pixel."""
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


# Compõe as máscaras semitransparentes sobre a imagem. A mistura é vetorizada
# em um buffer numpy porque alocar uma camada PIL por região é proibitivo com
# dezenas de regiões em imagens de centenas de milhares de pixels.
def blend_masks(
    image: Image.Image, shapes: Sequence[DrawableShape], *, alpha: float = 0.35
) -> Image.Image:
    """Retorna a imagem com as máscaras das formas mescladas por cima.

    Argumentos:
        image: imagem base, não modificada.
        shapes: formas cujas máscaras serão mescladas, na ordem de cor.
        alpha: opacidade da mistura, em ``[0, 1]``.
    Retorna:
        nova imagem RGB com as máscaras compostas.
    """
    canvas = np.array(image.convert("RGB"), dtype=np.float64)
    for index, shape in enumerate(shapes):
        color = np.array(color_for(index), dtype=np.float64)
        selected = shape.mask.data
        canvas[selected] = canvas[selected] * (1 - alpha) + color * alpha
    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


# Desenha o contorno da bounding box de cada forma, engrossando as destacadas.
# Separada de blend_masks para que exista um artifact só de caixas, no qual a
# over-segmentação aparece sem a máscara colorida escondendo os limites.
def draw_boxes(
    image: Image.Image,
    shapes: Sequence[DrawableShape],
    *,
    width: int = 1,
    emphasized: frozenset[str] = frozenset(),
) -> Image.Image:
    """Retorna a imagem com o contorno das caixas das formas desenhado.

    Argumentos:
        image: imagem base, não modificada.
        shapes: formas cujas caixas serão desenhadas.
        width: espessura do contorno das formas não destacadas.
        emphasized: ``shape_id`` que recebem contorno mais grosso.
    Retorna:
        nova imagem RGB com as caixas desenhadas.
    """
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    for index, shape in enumerate(shapes):
        box = shape.box
        outline_width = width + 1 if shape.shape_id in emphasized else width
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max - 1, box.y_max - 1],
            outline=color_for(index),
            width=outline_width,
        )
    return result


# Formata os dois eixos de confiança lado a lado e nomeados. Existe porque o
# overlay imprimia apenas a geometric_confidence colada ao label semântico, o
# que fazia um número de qualidade de máscara ser lido como certeza semântica
# (docs/known-limitations.md, seção 3). Nomear os dois eixos torna a leitura
# errada impossível, e "?" distingue ausência de score de score baixo.
def confidence_caption(semantic: float | None, geometric: float | None) -> str:
    """Retorna a legenda ``"sem=... geom=..."`` dos dois eixos de confiança."""
    semantic_text = "?" if semantic is None else f"{semantic:.2f}"
    geometric_text = "?" if geometric is None else f"{geometric:.2f}"
    return f"sem={semantic_text} geom={geometric_text}"


# Monta o texto de duas linhas de uma forma: o label, e as duas confianças
# nomeadas embaixo. Usada tanto pelo overlay completo quanto pelos testes de
# regressão que impedem o formato antigo, "label (0.97)", de voltar.
def label_caption(shape: DrawableShape) -> str:
    """Retorna o rótulo de duas linhas da forma, com label e confianças."""
    label = shape.label or "?"
    return f"{label}\n{confidence_caption(shape.semantic_confidence, shape.geometric_confidence)}"


# Calcula o centro de massa da máscara, com recuo para a caixa quando a máscara
# está vazia. O rótulo é posicionado aí, e não no canto da caixa, porque uma
# máscara côncava ou muito alongada tem o canto da caixa longe do que ela de
# fato cobre — e o texto acaba anotando a região errada.
def _anchor(shape: DrawableShape) -> tuple[float, float]:
    """Retorna o ponto de ancoragem do rótulo da forma."""
    rows, columns = np.nonzero(shape.mask.data)
    if rows.size == 0:
        return (shape.box.x_min, shape.box.y_min)
    return (float(columns.mean()), float(rows.mean()))


# Desenha os rótulos das maiores formas sobre a imagem. O teto existe porque o
# SAM automático produz dezenas de regiões pequenas, e rotular todas transforma
# o artifact em massa ilegível — exatamente o que os artifacts separados por
# camada existem para evitar.
def draw_labels(
    image: Image.Image, shapes: Sequence[DrawableShape], *, max_labels: int = MAX_LABELED_SHAPES
) -> Image.Image:
    """Retorna a imagem com o rótulo das maiores formas desenhado.

    Argumentos:
        image: imagem base, não modificada.
        shapes: formas candidatas a rótulo.
        max_labels: quantas formas, das maiores por área, recebem rótulo.
    Retorna:
        nova imagem RGB com os rótulos desenhados.
    """
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    largest_first = sorted(shapes, key=lambda shape: shape.mask.area(), reverse=True)
    labeled = {shape.shape_id for shape in largest_first[:max_labels]}
    for index, shape in enumerate(shapes):
        if shape.shape_id not in labeled:
            continue
        text = label_caption(shape)
        anchor_x, anchor_y = _anchor(shape)
        text_box = draw.textbbox((0, 0), text, font=_FONT)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        origin_x = max(0.0, anchor_x - text_width / 2)
        origin_y = max(0.0, anchor_y - text_height / 2)
        draw.rectangle(
            [origin_x, origin_y, origin_x + text_width + 4, origin_y + text_height + 4],
            fill=(0, 0, 0),
        )
        draw.text((origin_x + 2, origin_y + 1), text, fill=color_for(index), font=_FONT)
    return result
