"""Construção das views em pixels de cada região.

Issue: #203 (evidência mask-aware multi-contexto).

Este módulo é o dono único da geometria de recorte da região. Ele existe para
que a evidência que vai ao encoder alinhado a linguagem e a evidência que vai
ao reasoner multimodal sejam **o mesmo recorte**, e não dois recortes
calculados em lugares diferentes que podem divergir com o tempo.

As views são um canal em memória, válido apenas durante o processamento de um
frame. Elas nunca são serializadas: o que persiste é o
:class:`~visual_perception.domain.region_evidence.RegionEvidenceSlot`, que
guarda a mesma ``crop_box`` e o mesmo ``transform`` por referência de
artifact.

Uma view de foreground é sempre produzida, mesmo quando o slot denso está
desabilitado, porque ela é a evidência local primária da região: sem ela, a
interpretação descreveria o entorno.
"""

from __future__ import annotations

import numpy as np

from visual_perception.config import ModuleConfig
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.region_reasoning import RegionView
from visual_perception.domain.regions import ObservedRegion


# Constrói as views em pixels de cada região, na ordem canônica dos slots.
# Existe para que o recorte da região seja calculado uma única vez por frame e
# compartilhado entre a evidência de embedding (#194) e o raciocínio semântico
# (#203). Chamada por application/multi_context.py::extract_region_evidence.
def build_region_views(
    regions: tuple[ObservedRegion, ...], image: ImagePayload, config: ModuleConfig
) -> dict[str, tuple[RegionView, ...]]:
    """Produz as views em pixels de cada região, indexadas por ``region_id``.

    Um slot desabilitado por configuração não gera view, para que o perfil de
    ablation continue medindo a ausência daquela evidência. A view de
    foreground é a exceção: ela depende apenas da máscara, e é sempre
    tentada. Uma view que não pode ser recortada é omitida em vez de
    interromper as demais.

    Argumentos:
        regions: as regiões canônicas já mescladas.
        image: o payload de pixels da imagem completa.
        config: a configuração do módulo (slots habilitados, margem de contexto).
    Retorna:
        um mapa de ``region_id`` para as views daquela região, em ordem de slot.
    """
    settings = config.multi_context
    scene_view = _scene_view(image) if settings.scene_conditioned_enabled else None
    views: dict[str, tuple[RegionView, ...]] = {}

    for region in regions:
        built: list[RegionView] = []
        # O foreground suprime tudo fora da máscara, independente de
        # ``masked_tight_crop``: aquele flag governa o slot de embedding, e
        # confundir os dois faria a evidência primária do reasoner incluir
        # fundo — exatamente o que a #203 proíbe.
        _append_crop_view(
            built, region, image, EvidenceSlot.FOREGROUND_DENSE, expansion=0.0, masked=True
        )
        if settings.tight_crop_enabled:
            _append_crop_view(
                built,
                region,
                image,
                EvidenceSlot.TIGHT_CROP,
                expansion=0.0,
                masked=settings.masked_tight_crop,
            )
        if settings.contextual_crop_enabled:
            _append_crop_view(
                built,
                region,
                image,
                EvidenceSlot.CONTEXTUAL_CROP,
                expansion=settings.context_expansion,
                masked=False,
            )
        if scene_view is not None:
            built.append(scene_view)
        views[region.region_id] = tuple(built)

    return views


# Recorta uma view e a acrescenta à lista, silenciando apenas o caso em que a
# caixa é degenerada. Existe para que a falha de uma view não interrompa as
# outras views da mesma região (critério de isolamento da #203); chamada por
# build_region_views uma vez por slot.
def _append_crop_view(
    built: list[RegionView],
    region: ObservedRegion,
    image: ImagePayload,
    slot: EvidenceSlot,
    *,
    expansion: float,
    masked: bool,
) -> None:
    """Recorta a view de ``slot`` e a acrescenta, ou a omite se a caixa for degenerada."""
    try:
        box = expanded_box(region.box, expansion, image.width, image.height)
        payload = crop_payload(image, region, box, masked=masked)
    except ValueError:
        return
    built.append(
        RegionView(
            slot=slot,
            payload=payload,
            crop_box=box,
            transform=CoordinateTransform(1.0, 1.0, box.x_min, box.y_min),
            masked=masked,
        )
    )


# Produz a view de cena: a imagem inteira, compartilhada por todas as regiões
# do frame. Existe separada porque não depende da região, e recomputá-la por
# região seria custo puro de memória.
def _scene_view(image: ImagePayload) -> RegionView:
    """Retorna a view de cena, com a caixa da imagem inteira."""
    box = BoundingBox(0.0, 0.0, float(image.width), float(image.height))
    return RegionView(
        slot=EvidenceSlot.SCENE_CONDITIONED,
        payload=image,
        crop_box=box,
        transform=CoordinateTransform.identity(),
        masked=False,
    )


# Expande a caixa da região pela fração pedida e a recorta aos limites da
# imagem. Vive aqui, junto do recorte, porque as duas operações definem a
# geometria de uma view; usada por _append_crop_view e por
# application/multi_context.py.
def expanded_box(box: BoundingBox, expansion: float, width: int, height: int) -> BoundingBox:
    """Retorna ``box`` expandida por ``expansion`` e recortada à imagem."""
    if expansion <= 0.0:
        return box.clipped_to(width=width, height=height)
    margin_x = box.width * expansion / 2.0
    margin_y = box.height * expansion / 2.0
    return BoundingBox(
        x_min=box.x_min - margin_x,
        y_min=box.y_min - margin_y,
        x_max=box.x_max + margin_x,
        y_max=box.y_max + margin_y,
    ).clipped_to(width=width, height=height)


# Recorta os pixels da caixa pedida, opcionalmente zerando tudo que está fora
# da máscara da região. Existe para que o crop mascarado seja uma opção de
# geometria e não uma variante duplicada; usada por _append_crop_view e por
# application/multi_context.py.
def crop_payload(
    image: ImagePayload, region: ObservedRegion, box: BoundingBox, *, masked: bool
) -> ImagePayload:
    """Recorta ``box`` da imagem, zerando o fundo quando ``masked``."""
    x_min, y_min = int(box.x_min), int(box.y_min)
    x_max, y_max = int(box.x_max), int(box.y_max)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Region {region.region_id!r} has a degenerate crop box after clipping.")
    crop = image.crop(x_min, y_min, x_max, y_max)
    if not masked:
        return crop
    window = region.mask.data[y_min:y_max, x_min:x_max]
    pixels = np.where(window[:, :, None], crop.pixels, 0).astype(crop.pixels.dtype)
    return ImagePayload(pixels, width=crop.width, height=crop.height)
