"""Compara variantes de tiling da region discovery sem rodar o VLM (#277).

Mede só discovery → filtro de área → merge, que é onde o tiling atua, para decidir o
grid antes de pagar as chamadas de VLM por região. Cada tile único é inferido uma vez
e reaproveitado pelas variantes que o contêm.

Métricas por frame e variante:

- ``proposals`` e ``regions``: regiões canônicas equivalem a chamadas de VLM de região;
- ``truncated_discarded``: propostas de tile descartadas por tocar borda interna;
- ``seam_regions``: regiões com um corte reto sobre uma emenda de tile, o artefato
  visto no outdoor do corridor-02;
- ``discovery_latency_s``: soma da inferência dos tiles usados pela variante.

Uso, a partir de ``modules/visual-perception``:

    python benchmarks/tiling_benchmark.py --frames-dir <frames> --frame-id <id> ... \\
      --output-dir benchmarks/.local/tiling-<data>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from validate_reference_pipeline import (  # noqa: E402  (também prepara o import path)
    SEQUENCE_MASKS_DIR,
    TILING_VARIANTS,
    load_image_area_config,
)
from visual_perception.application.execution_profile import research_quality_config  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.application.proposal_filtering import filter_proposals  # noqa: E402
from visual_perception.application.region_merge import merge_regions  # noqa: E402
from visual_perception.application.tiling import (  # noqa: E402
    Tile,
    build_tiles,
    is_truncated_by_tile,
    remap_to_global,
)
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.infrastructure.adapters.region_discovery_backend import (  # noqa: E402
    RealRegionDiscoveryAdapter,
)

#: Um corte só conta como emenda quando é longo o bastante para não ser coincidência
#: de borda natural alinhada à coluna do tile.
SEAM_MIN_PIXELS = 24
SEAM_MIN_FRACTION = 0.5


# Identifica um tile pela geometria, para reaproveitar a inferência entre variantes
# que compartilham o mesmo recorte (a imagem completa aparece em todas).
def _tile_key(tile: Tile) -> tuple[int, int, int, int]:
    """Retorna o retângulo global do tile."""
    left, top = int(tile.transform.offset_x), int(tile.transform.offset_y)
    return (left, top, left + tile.payload.width, top + tile.payload.height)


# Conta regiões com um corte reto sobre uma linha interna de tile. Existe porque o
# artefato das emendas é geométrico: uma borda de máscara vertical ou horizontal
# longa, exatamente onde um tile começa ou termina.
def count_seam_regions(regions: tuple, tiles: tuple[Tile, ...], width: int, height: int) -> int:  # type: ignore[type-arg]
    """Retorna quantas regiões têm uma borda reta coincidente com uma emenda de tile.

    Argumentos:
        regions: regiões canônicas depois do merge.
        tiles: tiles da variante, incluindo a imagem completa.
        width: largura da imagem.
        height: altura da imagem.
    Retorna:
        número de regiões com ao menos um corte de emenda.
    """
    columns = sorted({x for tile in tiles for x in _tile_key(tile)[0::2] if 0 < x < width})
    rows = sorted({y for tile in tiles for y in _tile_key(tile)[1::2] if 0 < y < height})
    seams = 0
    for region in regions:
        data = region.mask.data
        row_span = int(data.any(axis=1).sum())
        column_span = int(data.any(axis=0).sum())
        vertical = any(
            _cut_length(data[:, x - 1], data[:, x]) >= max(SEAM_MIN_PIXELS, SEAM_MIN_FRACTION * row_span)
            for x in columns
        )
        horizontal = any(
            _cut_length(data[y - 1, :], data[y, :]) >= max(SEAM_MIN_PIXELS, SEAM_MIN_FRACTION * column_span)
            for y in rows
        )
        seams += int(vertical or horizontal)
    return seams


# Mede quantos pixels de uma linha mudam de dentro para fora da máscara na linha vizinha.
def _cut_length(before: np.ndarray, after: np.ndarray) -> int:
    """Retorna o comprimento da transição entre duas linhas adjacentes da máscara."""
    return int(np.logical_xor(before, after).sum())


# Pinta as regiões canônicas sobre o frame para inspeção lado a lado das variantes.
def render_regions(pixels: np.ndarray, regions: tuple) -> Image.Image:  # type: ignore[type-arg]
    """Retorna o frame com cada região em uma cor determinística."""
    image = pixels.astype(np.float32).copy()
    rng = np.random.default_rng(0)
    for region in sorted(regions, key=lambda item: -item.mask.area()):
        color = rng.integers(40, 255, size=3)
        selected = region.mask.data
        image[selected] = image[selected] * 0.45 + color * 0.55
    return Image.fromarray(image.astype(np.uint8))


# Executa as variantes sobre os frames pedidos e grava JSON e overlays.
def run(frames: list[Path], sequence_masks: Path | None, output_dir: Path) -> dict[str, object]:
    """Roda discovery real uma vez por tile único e avalia cada variante.

    Argumentos:
        frames: PNGs de entrada.
        sequence_masks: geometria de área da sequência, ou ``None``.
        output_dir: destino de ``results.json`` e dos overlays.
    Retorna:
        o relatório gravado em ``results.json``.
    """
    base = research_quality_config(real_backends=True)
    config = dataclasses.replace(base, image_area=load_image_area_config(sequence_masks))
    discoverer = RealRegionDiscoveryAdapter(ModelLifecycleManager())
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "region_discovery": dataclasses.asdict(config.region_discovery),
        "merge": dataclasses.asdict(config.merge),
        "variants": {name: dataclasses.asdict(tiling) for name, tiling in TILING_VARIANTS.items()},
        "frames": {},
    }
    for frame in frames:
        pixels = np.array(Image.open(frame).convert("RGB"))
        height, width = pixels.shape[:2]
        payload = ImagePayload(pixels, width=width, height=height)
        area_masks = config.image_area.rasterize(width, height)
        inferred: dict[tuple[int, int, int, int], tuple[tuple, float]] = {}  # type: ignore[type-arg]
        frame_report: dict[str, object] = {}
        for name, tiling in TILING_VARIANTS.items():
            tiles = build_tiles(payload, tiling)
            proposals, truncated, latency = [], 0, 0.0
            for tile in tiles:
                key = _tile_key(tile)
                if key not in inferred:
                    start = time.perf_counter()
                    local = discoverer.discover(tile.payload, config.region_discovery)
                    inferred[key] = (local, time.perf_counter() - start)
                local, elapsed = inferred[key]
                latency += elapsed
                for proposal in local:
                    if tiling.discard_tile_border_truncations and is_truncated_by_tile(
                        proposal, tile, image_width=width, image_height=height
                    ):
                        truncated += 1
                        continue
                    proposals.append(remap_to_global(proposal, tile, image_width=width, image_height=height))
            kept, rejected = filter_proposals(tuple(proposals), area_masks=area_masks, config=config.proposal_filter)
            regions = merge_regions(frame.stem, kept, config.merge)
            frame_report[name] = {
                "tiles": len(tiles),
                "proposals": len(proposals),
                "kept_after_area_filter": len(kept),
                "truncated_discarded": truncated,
                "regions": len(regions),
                "seam_regions": count_seam_regions(regions, tiles, width, height),
                "discovery_latency_s": round(latency, 2),
            }
            render_regions(pixels, regions).save(output_dir / f"{frame.stem}-{name}.png")
            print(frame.stem, name, json.dumps(frame_report[name]), flush=True)
        report["frames"][frame.stem] = frame_report  # type: ignore[index]
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# Ponto de entrada do benchmark.
def main(argv: list[str] | None = None) -> None:
    """Lê argumentos e roda o benchmark de tiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-id", action="append", required=True)
    parser.add_argument("--sequence-masks", type=Path, default=SEQUENCE_MASKS_DIR / "corridor-02.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    frames = [arguments.frames_dir / f"{frame_id}.png" for frame_id in arguments.frame_id]
    run(frames, arguments.sequence_masks, arguments.output_dir)


if __name__ == "__main__":
    main()
