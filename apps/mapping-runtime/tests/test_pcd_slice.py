"""Testes da importação limitada de mapas PCD para o viewer."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from mapping_runtime.pcd_slice import export_pcd_slice


# Cria um PCD binário mínimo e independente de PCL. Existe para testar offsets,
# amostragem e campos opcionais com o mesmo layout produzido pelo FAST-LIO.
def _write_pcd(destination: Path) -> None:
    """Persiste três pontos ``x y z intensity`` em formato PCD binário."""
    header = """# .PCD v0.7
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 3
HEIGHT 1
POINTS 3
DATA binary
""".encode("ascii")
    destination.write_bytes(
        header
        + struct.pack("<ffff", 0.0, 1.0, -1.0, 10.0)
        + struct.pack("<ffff", 2.0, 3.0, 0.0, 20.0)
        + struct.pack("<ffff", 4.0, 5.0, 1.0, 30.0)
    )


# Confirma que o adapter limita a carga do navegador e preserva coordenadas,
# intensidade, proveniência do arquivo e cor geométrica de visualização.
def test_export_pcd_slice_samples_binary_map(tmp_path: Path) -> None:
    """Exporta uma amostra determinística no schema aceito pelo viewer."""
    source = tmp_path / "segment.pcd"
    destination = tmp_path / "segment.json"
    _write_pcd(source)

    result = export_pcd_slice(
        source,
        destination,
        map_id="segment",
        map_frame="map",
        max_points=2,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "geometric_pcd_slice"
    assert payload["source"]["point_count"] == 3
    assert payload["source"]["sampling_stride"] == 2
    assert [point["coordinates_m"] for point in payload["points"]] == [
        [0.0, 1.0, -1.0],
        [4.0, 5.0, 1.0],
    ]
    assert payload["points"][0]["intensity"] == 10.0
    assert payload["points"][0]["display_color_rgb"] != payload["points"][1]["display_color_rgb"]


# Rejeita variantes que exigiriam um decoder diferente. O erro explícito evita
# interpretar bytes comprimidos como registros e produzir geometria corrompida.
def test_export_pcd_slice_rejects_compressed_payload(tmp_path: Path) -> None:
    """Recusa PCD binary_compressed com diagnóstico direto."""
    source = tmp_path / "compressed.pcd"
    source.write_bytes(
        b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 1\nDATA binary_compressed\n"
    )

    with pytest.raises(ValueError, match="only DATA binary"):
        export_pcd_slice(source, tmp_path / "map.json", map_id="map", map_frame="map")
