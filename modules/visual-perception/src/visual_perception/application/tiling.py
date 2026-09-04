"""Tiling multi-escala de imagem e remapeamento global.

Issue: #159.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.config import TilingConfig
from visual_perception.domain.geometry import CoordinateTransform
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal, RegionProposal, TileProvenance


# Representa uma view tile-local da imagem completa, junto com o transform
# que leva de volta às coordenadas globais. Existe para que cada tile
# carregue consigo a informação necessária para remap_to_global reconverter
# suas proposals locais em coordenadas da imagem original.
@dataclass(frozen=True)
class Tile:
    """Uma view tile-local da imagem completa, com seu transform de volta às coordenadas globais."""

    scale_id: str
    tile_id: str
    payload: ImagePayload
    transform: CoordinateTransform


# Constrói a lista de tiles a processar: sempre a imagem completa, e,
# quando multi-escala está habilitado, também um grid de tiles sobrepostos.
# Existe para permitir region discovery em múltiplas escalas sem duplicar a
# lógica de particionamento em cada chamador. Usada pelo pipeline principal
# antes da etapa de region discovery.
def build_tiles(image: ImagePayload, config: TilingConfig) -> tuple[Tile, ...]:
    """Produz o tile da imagem completa mais, quando habilitado, um grid de tiles sobrepostos.

    A imagem completa é sempre incluída como scale ``full``/tile ``whole``,
    para que um chamador sempre possa rodar discovery não-tiled mesmo com
    multi-escala ativado.
    """
    tiles = [Tile("full", "whole", image, CoordinateTransform.identity())]
    if not config.multi_scale_enabled:
        return tuple(tiles)

    rows_str, _, cols_str = config.tile_grid.partition("x")
    rows, cols = int(rows_str), int(cols_str)
    base_w = image.width / cols
    base_h = image.height / rows
    overlap_w = base_w * config.overlap_ratio
    overlap_h = base_h * config.overlap_ratio

    for row in range(rows):
        for col in range(cols):
            x_min = max(0, int(col * base_w - overlap_w))
            y_min = max(0, int(row * base_h - overlap_h))
            x_max = min(image.width, int((col + 1) * base_w + overlap_w))
            y_max = min(image.height, int((row + 1) * base_h + overlap_h))
            crop = image.crop(x_min, y_min, x_max, y_max)
            transform = CoordinateTransform(
                scale_x=1.0, scale_y=1.0, offset_x=float(x_min), offset_y=float(y_min)
            )
            tiles.append(Tile("tile", f"r{row}c{col}", crop, transform))
    return tuple(tiles)


# Remapeia uma proposal tile-local exatamente de volta às coordenadas da
# imagem original, usando o transform do tile, e anexa a proveniência do
# tile (scale_id/tile_id) à RegionProposal resultante. Chamada pelo
# pipeline principal para cada proposal produzida dentro de um tile.
def remap_to_global(
    local_proposal: LocalRegionProposal,
    tile: Tile,
    *,
    image_width: int,
    image_height: int,
) -> RegionProposal:
    """Remapeia uma proposal tile-local exatamente de volta às coordenadas da imagem original."""
    global_box = tile.transform.box_to_global(local_proposal.box)
    global_mask = tile.transform.mask_to_global(
        local_proposal.mask, global_width=image_width, global_height=image_height
    )
    return RegionProposal(
        proposal_id=f"{tile.scale_id}-{tile.tile_id}-{local_proposal.local_id}",
        mask=global_mask,
        box=global_box,
        geometric_confidence=local_proposal.geometric_confidence,
        source=local_proposal.source,
        tile=TileProvenance(scale_id=tile.scale_id, tile_id=tile.tile_id),
    )
