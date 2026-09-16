"""Testes locais do protocolo de comparação de region discovery."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from compare_region_discovery import require_same_input, write_side_by_side_overlay  # noqa: E402


# Verifica que o artifact visual mantém os dois painéis e o cabeçalho de
# identificação, sem exigir modelos, GPU ou um run completo da pipeline.
def test_write_side_by_side_overlay_combines_two_same_sized_images(tmp_path: Path) -> None:
    """Grava uma comparação visual com ambos os overlays lado a lado."""
    baseline = tmp_path / "baseline.png"
    candidate = tmp_path / "candidate.png"
    output = tmp_path / "comparison.png"
    Image.new("RGB", (4, 3), "red").save(baseline)
    Image.new("RGB", (4, 3), "blue").save(candidate)

    write_side_by_side_overlay(
        baseline, candidate, baseline_name="SAM3", candidate_name="Florence-2", output=output
    )

    image = Image.open(output)
    assert image.size == (8, 31)
    assert image.getpixel((1, 29)) == (255, 0, 0)
    assert image.getpixel((5, 29)) == (0, 0, 255)


# A comparação não pode continuar quando os braços não vieram do mesmo frame,
# porque toda diferença posterior deixaria de poder ser atribuída ao backend.
def test_require_same_input_rejects_different_hashes(tmp_path: Path) -> None:
    """Recusa manifests que apontam para SHA-256s de entrada distintos."""
    for run, digest in ((tmp_path / "sam3", "a"), (tmp_path / "florence2", "b")):
        frame_dir = run / "frames" / "frame-a"
        frame_dir.mkdir(parents=True)
        (frame_dir / "diagnostics.json").write_text("{}")
        (run / "manifest.json").write_text(
            '{"frames": [{"frame_id": "frame-a", "input": {"sha256": "' + digest + '"}}]}'
        )

    with pytest.raises(ValueError, match="mesmo input"):
        require_same_input(tmp_path / "sam3", tmp_path / "florence2", "frame-a")
