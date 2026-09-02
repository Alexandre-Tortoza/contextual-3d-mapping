"""Multi-scale image tiling and global remapping.

Issue: #159.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.config import TilingConfig
from visual_perception.domain.geometry import CoordinateTransform
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal, RegionProposal, TileProvenance


@dataclass(frozen=True)
class Tile:
    """One tile-local view of the full image, with its transform back to global coordinates."""

    scale_id: str
    tile_id: str
    payload: ImagePayload
    transform: CoordinateTransform


def build_tiles(image: ImagePayload, config: TilingConfig) -> tuple[Tile, ...]:
    """Produce the full-image tile plus, when enabled, an overlapping tile grid.

    The full image is always included as scale ``full``/tile ``whole`` so a
    caller can always run non-tiled discovery even when multi-scale is on.
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


def remap_to_global(
    local_proposal: LocalRegionProposal,
    tile: Tile,
    *,
    image_width: int,
    image_height: int,
) -> RegionProposal:
    """Remap one tile-local proposal exactly back to original image coordinates."""
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
