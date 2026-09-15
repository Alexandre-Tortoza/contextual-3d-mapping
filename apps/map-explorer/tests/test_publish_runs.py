"""Regressões de persistência e publicação de runs contextuais independentes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from publish_map_index import publish, save_run


# Materializa um mapa pequeno com previews reais para verificar a publicação
# sem inferência, rosbag ou fixtures volumosas de pesquisa.
def context_artifact(directory: Path) -> Path:
    """Cria uma entrada contextual mínima com imagens relativas."""
    assets = directory / "source-assets"
    assets.mkdir(parents=True)
    (assets / "raw.png").write_bytes(b"raw-image")
    (assets / "overlay.png").write_bytes(b"overlay-image")
    artifact = directory / "source.json"
    artifact.write_text(json.dumps({
        "schema_version": 2, "artifact_type": "contextual_rgb_lidar_slice",
        "map_id": "shared-map", "map_frame": "map", "source": {"sha256": "a" * 64, "point_count": 10},
        "points": [], "regions": [{"region_id": "region-1"}],
        "observations": [{"raw_image_uri": "source-assets/raw.png", "overlay_image_uri": "source-assets/overlay.png"}],
        "context_summary": {"visual_observation_count": 1, "contextual_point_count": 1},
    }))
    return artifact


# Um segundo resultado do mesmo mapa deve preservar o primeiro e manter os
# previews relativos válidos mesmo depois que a origem tiver sido alterada.
def test_runs_preserve_context_and_previews(tmp_path: Path) -> None:
    """Confere pastas independentes, fidelidade do conteúdo e índice contextual."""
    artifact = context_artifact(tmp_path / "input")
    public = tmp_path / "public"
    first = save_run(public, artifact, run_id="first", label="Antes")
    original = (first / "context.json").read_bytes()
    payload = json.loads(artifact.read_text())
    payload["context_summary"]["contextual_point_count"] = 12
    artifact.write_text(json.dumps(payload))
    second = save_run(public, artifact, run_id="second", label="Depois")
    assert first != second
    assert (first / "context.json").read_bytes() == original
    assert (second / "context.json").read_bytes() == artifact.read_bytes()
    assert (first / "source-assets/raw.png").read_bytes() == b"raw-image"
    maps = public / "maps"
    maps.mkdir()
    (maps / "geometry.json").write_text(json.dumps({"schema_version": 1, "map_id": "geometry"}))
    entries = json.loads(publish(public).read_text())
    runs = [entry for entry in entries if entry["artifact_type"] == "contextual_rgb_lidar_slice"]
    consolidated = [entry for entry in entries if entry["artifact_type"] == "consolidated_contextual_map"]
    assert [entry["run_id"] for entry in runs] == ["second", "first"]
    assert all(entry["url"].startswith("/runs/") for entry in runs)
    assert len(consolidated) == 1
    assert consolidated[0]["source_run_count"] == 2
    assert consolidated[0]["url"].startswith("/maps/consolidated/")


# O debug é evidência da mesma run e precisa sobreviver à cópia imutável do
# publisher para que o inspector não aponte de volta ao diretório de trabalho.
def test_publication_copies_declared_debug_assets(tmp_path: Path) -> None:
    """Publica imagem e diagnóstico de debug declarados por uma observação."""
    artifact = context_artifact(tmp_path / "input")
    debug = artifact.parent / "source-assets" / "DEBUG"
    debug.mkdir()
    (debug / "sam3.png").write_bytes(b"sam3")
    (debug / "diagnostics.json").write_text("{}")
    payload = json.loads(artifact.read_text())
    payload["observations"][0]["debug_assets"] = {
        "sam3": "source-assets/DEBUG/sam3.png",
        "diagnostics": "source-assets/DEBUG/diagnostics.json",
    }
    artifact.write_text(json.dumps(payload))
    saved = save_run(tmp_path / "public", artifact, run_id="debug")
    assert (saved / "source-assets/DEBUG/sam3.png").read_bytes() == b"sam3"
    assert (saved / "source-assets/DEBUG/diagnostics.json").read_text() == "{}"


# Reabrir uma execução não pode duplicar a run nem autorizar sobrescrita quando
# uma inferência posterior reaproveita o nome com conteúdo diferente.
def test_repeat_is_idempotent_and_changed_content_cannot_overwrite(tmp_path: Path) -> None:
    """Protege a identidade imutável inclusive quando apenas o preview muda."""
    artifact = context_artifact(tmp_path / "input")
    public = tmp_path / "public"
    first = save_run(public, artifact, run_id="stable")
    assert save_run(public, artifact) == first
    assert save_run(public, artifact, run_id="stable") == first
    (artifact.parent / "source-assets/raw.png").write_bytes(b"another-frame")
    with pytest.raises(FileExistsError):
        save_run(public, artifact, run_id="stable")
    assert (first / "source-assets/raw.png").read_bytes() == b"raw-image"
    assert save_run(public, artifact) != first


# Geometria pura e resultados incompletos não são candidatos válidos para a
# comparação contextual pedida pelo usuário.
def test_geometry_and_missing_previews_are_not_published(tmp_path: Path) -> None:
    """Rejeita entradas inválidas antes de publicar uma pasta de run."""
    artifact = context_artifact(tmp_path / "input")
    public = tmp_path / "public"
    (artifact.parent / "source-assets/raw.png").unlink()
    with pytest.raises(FileNotFoundError):
        save_run(public, artifact)
    artifact.write_text(json.dumps({"schema_version": 1, "map_id": "geometry"}))
    with pytest.raises(ValueError, match="contexto"):
        save_run(public, artifact)
    assert not public.exists()


# Regressão da campanha 10x4fps: o SAM3 não detectou nada, e o mapa sem regiões
# nem pontos contextuais foi publicado como se fosse uma run válida.
@pytest.mark.parametrize(
    ("regions", "contextual_point_count", "message"),
    [([], 1, "sem regiões"), ([{"region_id": "region-1"}], 0, "sem pontos contextuais")],
)
def test_runs_without_context_are_not_published(
    tmp_path: Path, regions: list, contextual_point_count: int, message: str,
) -> None:
    """Recusa mapas que passaram pelo pipeline sem produzir contexto semântico."""
    artifact = context_artifact(tmp_path / "input")
    payload = json.loads(artifact.read_text())
    payload["regions"] = regions
    payload["context_summary"]["contextual_point_count"] = contextual_point_count
    artifact.write_text(json.dumps(payload))
    public = tmp_path / "public"
    with pytest.raises(ValueError, match=message):
        save_run(public, artifact)
    assert not public.exists()


# Um path malformado não pode fazer a publicação copiar arquivos externos ou
# modificar os metadados reservados de outra run.
@pytest.mark.parametrize("uri", ["../outside.png", "/tmp/image.png", "https://example.org/image.png", "context.json", ".", "?image"])
def test_previews_stay_inside_the_run(tmp_path: Path, uri: str) -> None:
    """Confere a fronteira dos paths de preview."""
    artifact = context_artifact(tmp_path / "input")
    (artifact.parent / "context.json").write_text("reserved")
    payload = json.loads(artifact.read_text())
    payload["observations"][0]["raw_image_uri"] = uri
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        save_run(tmp_path / "public", artifact)


# Pastas temporárias ainda não representam uma run publicada e podem desaparecer
# durante a renomeação atômica; o catálogo não deve apontar para elas.
def test_index_ignores_staging_directories(tmp_path: Path) -> None:
    """Mantém apenas a pasta final da publicação no catálogo."""
    artifact = context_artifact(tmp_path / "input")
    public = tmp_path / "public"
    saved = save_run(public, artifact, run_id="ready")
    staging = saved.parent / ".pending"
    staging.mkdir()
    (staging / "manifest.json").write_bytes((saved / "manifest.json").read_bytes())
    (staging / "context.json").write_bytes((saved / "context.json").read_bytes())
    entries = json.loads(publish(public).read_text())
    assert [entry["run_id"] for entry in entries if entry["artifact_type"] == "contextual_rgb_lidar_slice"] == ["ready"]
    assert len([entry for entry in entries if entry["artifact_type"] == "consolidated_contextual_map"]) == 1


# Cada trecho carrega só a própria vizinhança; o fundo global declarado pela run é
# publicado uma vez, verificado pelo digest, e vira a geometria do mapa consolidado.
def test_declared_backdrop_is_published_once_and_used_by_the_consolidated_map(tmp_path: Path) -> None:
    """Publica o slice global da run e o usa como geometria do consolidado."""
    import hashlib

    backdrop = tmp_path / "global.json"
    backdrop.write_text(json.dumps({
        "artifact_type": "geometric_pcd_slice", "map_id": "shared-map", "map_frame": "map",
        "source": {"sha256": "a" * 64, "point_count": 10},
        "points": [{"geometry_id": "shared-map:pcd:0", "coordinates_m": [0.0, 0.0, 0.0]}],
    }))
    digest = hashlib.sha256(backdrop.read_bytes()).hexdigest()
    artifact = context_artifact(tmp_path / "input")
    payload = json.loads(artifact.read_text())
    payload["geometry_backdrop"] = {"artifact_uri": str(backdrop), "sha256": digest}
    artifact.write_text(json.dumps(payload))
    public = tmp_path / "public"

    saved = save_run(public, artifact, run_id="segment")
    entries = json.loads(publish(public).read_text())

    assert (public / "maps" / "geometry" / f"{digest}.json").read_bytes() == backdrop.read_bytes()
    assert json.loads((saved / "manifest.json").read_text())["geometry_backdrop_sha256"] == digest
    consolidated_url = next(entry["url"] for entry in entries if entry["artifact_type"] == "consolidated_contextual_map")
    consolidated = json.loads((public / consolidated_url.lstrip("/")).read_text())
    assert [point["geometry_id"] for point in consolidated["points"]] == ["shared-map:pcd:0"]
    backdrop.write_text("{}")
    payload["geometry_backdrop"]["sha256"] = "b" * 64
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        save_run(tmp_path / "other-public", artifact, run_id="tampered")
