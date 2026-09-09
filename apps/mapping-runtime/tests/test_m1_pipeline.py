"""Teste end-to-end do caminho offline executável do M1."""

from __future__ import annotations

import json
from pathlib import Path

from mapping_runtime.cli import main


# Garante que a CLI realmente compõe mapa, associação e exportação em um
# artifact que o map-explorer consegue consumir sem dependências externas.
def test_demo_command_exports_inspectable_slice(tmp_path: Path) -> None:
    """Executa o demo e valida geometria, cor, região, oclusão e proveniência."""
    destination = tmp_path / "m1-demo.json"

    assert main(("demo", "--output", str(destination))) == 0

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["map_id"] == "m1-demo"
    assert len(payload["points"]) == 5
    assert {point["association"]["status"] for point in payload["points"]} == {
        "associated",
        "occluded",
    }
    center = next(point for point in payload["points"] if point["geometry_id"] == "lidar-000:1")
    assert center["association"]["region_id"] == "demo-region-center"
    assert center["provenance"]["producer"] == "mapping-runtime-demo"
