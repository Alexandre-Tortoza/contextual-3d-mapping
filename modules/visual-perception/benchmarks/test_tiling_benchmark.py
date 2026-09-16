"""Testes da métrica de emendas do benchmark de tiling, sem GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

from fixtures import payload_with_blobs  # noqa: E402
from tiling_benchmark import count_seam_regions  # noqa: E402
from visual_perception.application.tiling import build_tiles  # noqa: E402
from visual_perception.config import TilingConfig  # noqa: E402
from visual_perception.domain.geometry import Mask  # noqa: E402
from visual_perception.domain.regions import ObservedRegion  # noqa: E402


# Cria uma região canônica retangular em coordenadas globais.
def _region(region_id: str, x_min: int, y_min: int, x_max: int, y_max: int, size: int = 96) -> ObservedRegion:
    """Retorna uma região cuja máscara cobre o retângulo pedido."""
    data = np.zeros((size, size), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    mask = Mask(data, size, size)
    return ObservedRegion(
        region_id=region_id, mask=mask, box=mask.bounding_box(), geometric_confidence=0.9,
        contributing_proposal_ids=(region_id,),
    )


# Um fragmento que termina exatamente na coluna onde um tile começa é o artefato
# medido; uma região que termina longe das emendas não conta.
def test_seam_regions_count_only_straight_cuts_on_tile_lines() -> None:
    payload = payload_with_blobs(width=96, height=96)
    tiles = build_tiles(payload, TilingConfig(multi_scale_enabled=True, tile_grid="2x2"))
    seam_column = int(tiles[2].transform.offset_x)
    cut = _region("cut", 4, 10, seam_column, 90)
    inside = _region("inside", 5, 5, 20, 20)
    whole = _region("whole", 0, 0, 96, 96)

    assert count_seam_regions((cut,), tiles, 96, 96) == 1
    assert count_seam_regions((inside, whole), tiles, 96, 96) == 0
