"""Testes de tiling multi-scale de imagem e remapeamento global (#159)."""

from __future__ import annotations

import numpy as np

from fixtures import payload_with_blobs
from visual_perception.application.tiling import build_tiles, remap_to_global
from visual_perception.config import TilingConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import LocalRegionProposal


# Com tiling multi-scale desabilitado, build_tiles deve produzir só o tile "full"
# (a imagem inteira), sem subdivisão.
def test_full_image_only_when_multi_scale_disabled() -> None:
    payload = payload_with_blobs()
    tiles = build_tiles(payload, TilingConfig(multi_scale_enabled=False))
    assert [tile.scale_id for tile in tiles] == ["full"]


# Com uma grade 2x2, build_tiles deve produzir o tile "full" mais os 4 tiles da
# grade, cada um com um tile_id de linha/coluna previsível (r{row}c{col}).
def test_two_by_two_overlapping_tiling() -> None:
    payload = payload_with_blobs(width=32, height=32)
    tiles = build_tiles(payload, TilingConfig(multi_scale_enabled=True, tile_grid="2x2"))
    assert [tile.scale_id for tile in tiles] == ["full", "tile", "tile", "tile", "tile"]
    assert {tile.tile_id for tile in tiles[1:]} == {"r0c0", "r0c1", "r1c0", "r1c1"}


# remap_to_global deve converter uma proposal local (coordenadas do tile) para
# coordenadas globais da imagem, aplicando o offset do CoordinateTransform do tile.
def test_remapping_masks_and_boxes_to_original_coordinates() -> None:
    payload = payload_with_blobs(width=32, height=32)
    tiles = build_tiles(payload, TilingConfig(multi_scale_enabled=True, tile_grid="2x2"))
    tile = next(tile for tile in tiles if tile.tile_id == "r1c1")

    local_data = np.zeros((tile.payload.height, tile.payload.width), dtype=np.bool_)
    local_data[0, 0] = True
    local_mask = Mask(local_data, tile.payload.width, tile.payload.height)
    local_proposal = LocalRegionProposal("p0", local_mask, local_mask.bounding_box(), 0.9, "fake")

    global_proposal = remap_to_global(local_proposal, tile, image_width=32, image_height=32)

    assert global_proposal.mask.image_width == 32
    assert global_proposal.tile.tile_id == "r1c1"
    assert global_proposal.box.x_min >= tile.transform.offset_x


# build_tiles deve ser determinístico mesmo em dimensões não divisíveis pela grade
# (bordas irregulares) e com overlap configurado: duas execuções com o mesmo input
# devem produzir tiles idênticos (mesmo scale_id/tile_id/transform).
def test_remapping_is_deterministic_for_edge_and_overlap_cases() -> None:
    payload = payload_with_blobs(width=33, height=33)
    config = TilingConfig(multi_scale_enabled=True, tile_grid="2x2", overlap_ratio=0.25)
    tiles_a = build_tiles(payload, config)
    tiles_b = build_tiles(payload, config)
    assert [(t.scale_id, t.tile_id, t.transform) for t in tiles_a] == [
        (t.scale_id, t.tile_id, t.transform) for t in tiles_b
    ]
