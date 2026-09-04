"""Renderiza uma VisualObservation sobre a imagem original para revisão visual.

Usado por ``validate_reference_pipeline.py`` (#190) para gerar as amostras
(overlay + JSON + resumo) que acompanham a validação do pipeline real. Não é
parte do contract público do módulo — é uma ferramenta de inspeção local.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_perception.domain.visual_observation import VisualObservation

#: Número máximo de regiões que recebem rótulo de texto no overlay. Com
#: dezenas de regiões (comum com SAM automático), rotular todas empilha
#: texto ilegível; rotula-se só as maiores/mais visíveis, mas a máscara
#: colorida continua desenhada para todas as regiões.
_MAX_LABELED_REGIONS = 20

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


# Extrai o primeiro claim do kind "label" de uma região, se houver, para
# rotular a máscara no overlay sem precisar reimplementar a lógica de
# desambiguação de labels da camada application.
def _first_label(region) -> str | None:  # noqa: ANN001
    for claim in region.claims:
        if claim.kind.value == "label":
            return claim.value
    return None


# Desenha máscaras semitransparentes coloridas, contorno da box e um rótulo
# curto (region_id + label + geometric_confidence) para cada região da
# observação, sobre uma cópia da imagem original.
def render_overlay(image: Image.Image, observation: VisualObservation) -> Image.Image:
    """Retorna uma cópia de ``image`` com as regiões da observação desenhadas por cima."""
    base = image.convert("RGBA")

    # Composição vetorizada das máscaras: alpha-blend direto em um buffer RGB
    # numpy, sem alocar uma camada PIL por região (proibitivo com dezenas de
    # regiões em imagens de centenas de milhares de pixels).
    canvas = np.array(base.convert("RGB"), dtype=np.float64)
    for index, region in enumerate(observation.regions):
        color = np.array(_PALETTE[index % len(_PALETTE)], dtype=np.float64)
        mask = region.mask.data
        alpha = 90 / 255
        canvas[mask] = canvas[mask] * (1 - alpha) + color * alpha
    overlay = Image.fromarray(canvas.astype(np.uint8), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Contorno fino em todas as regiões (mostra a segmentação inteira), mas
    # rótulo de texto só nas _MAX_LABELED_REGIONS maiores (por área de
    # máscara) — evita empilhar dezenas de textos ilegíveis uns sobre os
    # outros quando o SAM produz muitas regiões pequenas.
    largest_first = sorted(observation.regions, key=lambda r: r.mask.area(), reverse=True)
    labeled_ids = {region.region_id for region in largest_first[:_MAX_LABELED_REGIONS]}

    for index, region in enumerate(observation.regions):
        color = _PALETTE[index % len(_PALETTE)]
        box = region.box
        width = 2 if region.region_id in labeled_ids else 1
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max - 1, box.y_max - 1], outline=(*color, 255), width=width
        )
        if region.region_id not in labeled_ids:
            continue
        label = _first_label(region)
        confidence = f"{region.geometric_confidence:.2f}"
        text = f"{label or '?'} ({confidence})"
        text_box = draw.textbbox((0, 0), text, font=_FONT)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        text_y = max(0, box.y_min - text_height - 4)
        draw.rectangle(
            [box.x_min, text_y, box.x_min + text_width + 4, text_y + text_height + 4],
            fill=(0, 0, 0, 230),
        )
        draw.text((box.x_min + 2, text_y + 1), text, fill=(255, 255, 255, 255), font=_FONT)

    return overlay.convert("RGB")
