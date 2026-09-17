"""Testes do ``debug_manifest`` estruturado do artifact contextual (schema_version 3).

Cobrem só a montagem do manifest a partir de arquivos já escritos em disco —
não o pipeline completo de ``export_corridor02_context``, que exige bag, PCD e
calibração reais. ``_copy_debug_asset`` e ``_visual_perception_debug_manifest``
são puras o bastante para testar isoladamente com fixtures de diretório.
"""

from __future__ import annotations

import json
from pathlib import Path

from mapping_runtime.corridor02_context import (
    _copy_debug_asset,
    _pipeline_backends,
    _visual_perception_debug_manifest,
)


# A promoção de qualquer artifact de debug é best-effort: a ausência da origem
# nunca é erro, só significa que a chave correspondente não aparece.
def test_copy_debug_asset_is_best_effort(tmp_path: Path) -> None:
    """Sem a origem, retorna ``None``; com ela, copia e devolve a URI relativa."""
    assets = tmp_path / "run-assets"
    assets.mkdir()

    assert _copy_debug_asset(tmp_path / "missing.json", assets, "missing.json") is None

    source = tmp_path / "diagnostics.json"
    source.write_text("{}", encoding="utf-8")
    uri = _copy_debug_asset(source, assets, "frame-0-diagnostics.json")

    assert uri == "run-assets/frame-0-diagnostics.json"
    assert (assets / "frame-0-diagnostics.json").read_text(encoding="utf-8") == "{}"


# Caso (a) do plano: sem with_debug_images e sem os artifacts ricos, o
# manifest da etapa visual-perception só carrega diagnostics (e grounding,
# quando presente).
def test_visual_perception_debug_manifest_without_debug_images(tmp_path: Path) -> None:
    """Sem ``with_debug_images``, só diagnostics/grounding entram no manifest."""
    frame_source = tmp_path / "frames" / "corridor-02-00001"
    frame_source.mkdir(parents=True)
    (frame_source / "diagnostics.json").write_text("{}", encoding="utf-8")
    (frame_source / "raw.png").write_bytes(b"not-a-real-png")
    assets = tmp_path / "context-assets"
    assets.mkdir()

    manifest = _visual_perception_debug_manifest(
        frame_source, assets, "corridor-02-00001", with_debug_images=False,
    )

    assert manifest == {"diagnostics": "context-assets/corridor-02-00001-diagnostics.json"}
    assert "images" not in manifest
    assert "discovery_tiles" not in manifest
    assert "region_views" not in manifest


# Caso (b) do plano: com with_debug_images=True e os artifacts ricos
# presentes, images/discovery_tiles/region_views aparecem com seus caminhos.
def test_visual_perception_debug_manifest_with_debug_images(tmp_path: Path) -> None:
    """Com ``with_debug_images=True``, imagens, discovery e region views aparecem."""
    frame_source = tmp_path / "frames" / "corridor-02-00001"
    frame_source.mkdir(parents=True)
    (frame_source / "diagnostics.json").write_text("{}", encoding="utf-8")
    (frame_source / "raw.png").write_bytes(b"raw")
    (frame_source / "regions-overlay.png").write_bytes(b"overlay")
    discovery = frame_source / "discovery"
    discovery.mkdir()
    (discovery / "full-whole-input.png").write_bytes(b"tile-input")
    (discovery / "full-whole-proposals.png").write_bytes(b"tile-proposals")
    region_dir = frame_source / "regions" / "region-a"
    region_dir.mkdir(parents=True)
    (region_dir / "tight-crop.png").write_bytes(b"tight")
    assets = tmp_path / "context-assets"
    assets.mkdir()

    manifest = _visual_perception_debug_manifest(
        frame_source, assets, "corridor-02-00001", with_debug_images=True,
    )

    assert manifest["diagnostics"] == "context-assets/corridor-02-00001-diagnostics.json"
    assert manifest["images"]["raw"] == "context-assets/corridor-02-00001-debug/raw.png"
    assert manifest["images"]["regions-overlay"] == "context-assets/corridor-02-00001-debug/regions-overlay.png"
    assert (assets / "corridor-02-00001-debug" / "raw.png").is_file()
    assert manifest["discovery_tiles"] == [{
        "id": "full-whole",
        "input": "context-assets/corridor-02-00001-debug/discovery/full-whole-input.png",
        "proposals": "context-assets/corridor-02-00001-debug/discovery/full-whole-proposals.png",
    }]
    assert manifest["region_views"] == [{
        "region_id": "region-a",
        "tight_crop": "context-assets/corridor-02-00001-debug/regions/region-a/tight-crop.png",
    }]


# Resume os backends configurados na run de percepção, incluindo a
# alternativa declarada só quando ela existe — sem fallback, a chave nem
# aparece, para não sugerir uma opção que a run não considerou.
def test_pipeline_backends_reads_run_manifest_config(tmp_path: Path) -> None:
    """Lê backend/checkpoint por estágio e a alternativa quando declarada."""
    run_root = tmp_path / "samples" / "20260101T000000Z"
    frame_dir = run_root / "frames" / "corridor-02-00001"
    frame_dir.mkdir(parents=True)
    (run_root / "manifest.json").write_text(json.dumps({
        "config": {
            "region_discovery": {"backend": "sam3", "checkpoint": "facebook/sam3"},
            "feature_extraction": {
                "backend": "dinov2", "checkpoint": "facebook/dinov2-base",
                "fallback_backend": "resnet", "fallback_checkpoint": "microsoft/resnet-50",
            },
            "language_embedding": {"backend": "clip", "checkpoint": "openai/clip-vit-large-patch14"},
            "multimodal_reasoning": {"backend": "qwen_vl", "checkpoint": "Qwen/Qwen2.5-VL-3B-Instruct"},
        },
    }), encoding="utf-8")

    backends = _pipeline_backends(frame_dir / "observation.json")

    assert backends["region_discovery"] == {"backend": "sam3", "checkpoint": "facebook/sam3"}
    assert backends["feature_extraction"] == {
        "backend": "dinov2", "checkpoint": "facebook/dinov2-base",
        "fallback_backend": "resnet", "fallback_checkpoint": "microsoft/resnet-50",
    }
    assert "fallback_backend" not in backends["language_embedding"]


# Sem manifest.json na raiz da run (fixture de teste que não o escreve, ou
# run de trabalho movida), a explicabilidade simplesmente não tem essa
# informação — nunca é erro.
def test_pipeline_backends_is_best_effort_without_manifest(tmp_path: Path) -> None:
    """Sem ``manifest.json``, retorna vazio em vez de lançar."""
    frame_dir = tmp_path / "samples" / "20260101T000000Z" / "frames" / "corridor-02-00001"
    frame_dir.mkdir(parents=True)

    assert _pipeline_backends(frame_dir / "observation.json") == {}
