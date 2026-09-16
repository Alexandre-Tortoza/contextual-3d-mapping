"""Contracts de publicação e comparação contextual pela CLI, sem modelos."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
from contextual_mapping_cli.main import app
from contextual_mapping_cli.models import DatasetProfile
from contextual_mapping_cli.processes import ProcessRunner
from contextual_mapping_cli.project import Project
from contextual_mapping_cli.workflows import Workflows
from typer.testing import CliRunner


# Isola os artifacts e executa o publisher real para proteger a fronteira entre
# as duas aplicações sem depender do catálogo ou de servidores do checkout.
def publication_project(tmp_path: Path) -> Project:
    """Cria um projeto temporário com o ponto de entrada real do publisher."""
    repository = Path(__file__).resolve().parents[3]
    source = repository / "apps/map-explorer/scripts/publish_map_index.py"
    target = tmp_path / "apps/map-explorer/scripts/publish_map_index.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    for module in ("semantic-map", "semantic-fusion"):
        shutil.copytree(repository / "modules" / module, tmp_path / "modules" / module,
                        ignore=shutil.ignore_patterns("__pycache__"))
    return Project(tmp_path)


# Fornece um mapa contextual mínimo com proveniência de imagens independente
# para testar persistência, sem carregar a geometria completa do dataset.
def write_context(artifact: Path, count: int = 1) -> None:
    """Materializa mapa e previews relativos no destino informado."""
    assets = artifact.parent / "context-assets"
    assets.mkdir(parents=True)
    (assets / "raw.png").write_bytes(b"raw")
    (assets / "overlay.png").write_bytes(b"overlay")
    artifact.write_text(json.dumps({
        "schema_version": 3, "artifact_type": "contextual_rgb_lidar_slice",
        "map_id": "shared-geometry", "map_frame": "map", "source": {"sha256": "a" * 64, "point_count": 10},
        "points": [], "regions": [{"region_id": "region-1"}],
        "observations": [{"raw_image_uri": "context-assets/raw.png", "overlay_image_uri": "context-assets/overlay.png"}],
        "context_summary": {"contextual_point_count": count},
    }), encoding="utf-8")


# Publicar uma cópia da mesma run precisa recuperar sua identidade original,
# mesmo quando o path de origem registrado no manifest já não é o argumento.
def test_cli_publishes_lists_and_reuses_moved_content(tmp_path: Path) -> None:
    """Exercita publish e runs com o mesmo catálogo público do viewer."""
    project = publication_project(tmp_path)
    artifact = tmp_path / "input/context.json"
    write_context(artifact)
    cli = CliRunner()
    result = cli.invoke(app, ["publish", "--root", str(project.root), "--artifact", str(artifact),
                              "--run-id", "first", "--label", "Primeira"])
    assert result.exit_code == 0, result.output
    moved = tmp_path / "copy"
    shutil.copytree(artifact.parent, moved)
    repeated = cli.invoke(app, ["publish", "--root", str(project.root), "--artifact", str(moved / "context.json")])
    assert repeated.exit_code == 0, repeated.output
    listing = cli.invoke(app, ["runs", "--json", "--root", str(project.root)])
    assert listing.exit_code == 0, listing.output
    entries = json.loads(listing.output)
    assert [(entry["run_id"], entry["label"]) for entry in entries] == [("first", "Primeira")]
    saved = Path(entries[0]["path"])
    assert (saved / "context.json").read_bytes() == artifact.read_bytes()
    assert (saved / "context-assets/raw.png").read_bytes() == b"raw"


# Um catálogo vazio ou uma identidade incorreta deve produzir erro acionável
# antes de iniciar qualquer servidor ou workflow de geometria.
def test_cli_serve_reports_missing_runs(tmp_path: Path) -> None:
    """Confere os erros públicos de seleção de run."""
    project = publication_project(tmp_path)
    cli = CliRunner()
    empty = cli.invoke(app, ["serve", "--root", str(project.root)])
    assert empty.exit_code == 1
    assert "publish --artifact" in empty.output
    artifact = tmp_path / "input/context.json"
    write_context(artifact)
    Workflows(project).publish_context(artifact, run_id="existing")
    missing = cli.invoke(app, ["serve", "--root", str(project.root), "--run-id", "missing"])
    assert missing.exit_code == 1
    assert "run desconhecida" in missing.output


# A comparação deve reutilizar o servidor que já serve o catálogo escolhido,
# sem lançar outro Vite nem exigir um segment-id geométrico.
def test_serve_reuses_matching_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confere URL estável e ausência de novo subprocesso."""
    project = publication_project(tmp_path)
    artifact = tmp_path / "input/context.json"
    write_context(artifact)
    workflows = Workflows(project)
    workflows.publish_context(artifact, run_id="saved")
    index = project.root / "apps/map-explorer/web/public/maps/index.json"
    monkeypatch.setattr("contextual_mapping_cli.workflows.urlopen", lambda *args, **kwargs: io.BytesIO(index.read_bytes()))
    calls = []
    monkeypatch.setattr(workflows.runner, "run", lambda *args, **kwargs: calls.append(args))
    assert workflows.serve(run_id="saved") == "http://localhost:5173/?artifact=/runs/saved/context.json"
    assert calls == []


# Substitui apenas a composição cara; a publicação continua usando o processo
# real para verificar que o resultado de cada execução chega ao catálogo.
class CompositionRunner(ProcessRunner):
    """Executor que fornece um mapa novo a cada chamada de composição."""

    # Registra o número de execuções no artifact para detectar sobrescritas.
    def __init__(self) -> None:
        """Inicializa o contador de composições simuladas."""
        self.compositions = 0
        self.visibility_modes = []

    # Materializa a saída solicitada pelo workflow ou delega ao publisher real.
    def run(self, command, *, cwd, environment=None) -> None:  # type: ignore[no-untyped-def]
        """Simula Make sem executar inferência ou mapeamento."""
        if command[0] == "make":
            self.compositions += 1
            self.visibility_modes.append(next(item.split("=", 1)[1] for item in command if item.startswith("M1_VISIBILITY_MODE=")))
            destination = next(item.split("=", 1)[1] for item in command if item.startswith("M1_CONTEXT_ARTIFACT="))
            write_context(Path(destination), self.compositions)
        else:
            super().run(command, cwd=cwd, environment=environment)


# Recompor a mesma janela e percepção é comum em ablações. As duas execuções
# precisam continuar comparáveis, com snapshots das entradas preservados.
def test_compose_keeps_separate_runs_for_the_same_inputs(tmp_path: Path) -> None:
    """Protege pastas distintas, manifests e publicação automática."""
    project = publication_project(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    window = artifacts / "segment-window.json"
    window.write_text('{"keyframes": []}')
    (artifacts / "segment.json").write_text("{}")
    visual_run = tmp_path / "perception/run"
    visual_run.mkdir(parents=True)
    (visual_run / "manifest.json").write_text('{"frame_count": 1}')
    profile = DatasetProfile("test", tmp_path / "run.bag", "/rgb", tmp_path / "intrinsics.json",
                             tmp_path / "extrinsics.json", tmp_path / "ground-truth.csv",
                             tmp_path / "masks.json", "map-target", "context-target")
    workflows = Workflows(project, CompositionRunner())
    first = workflows.compose(
        profile, segment_id="segment", window=window, visual_run=visual_run, run_name="segment-no-prior"
    )
    original = first.read_bytes()
    second = workflows.compose(
        profile, segment_id="segment", window=window, visual_run=visual_run,
        visibility_mode="dense_cells", run_name="segment-box-overlap"
    )
    assert workflows.runner.visibility_modes == ["measured_surfaces", "dense_cells"]
    assert json.loads((second.parent / "manifest.json").read_text())["visibility_mode"] == "dense_cells"
    assert first.parent != second.parent
    assert first.parent.name.endswith("-segment-no-prior")
    assert second.parent.name.endswith("-segment-box-overlap")
    assert first.read_bytes() == original
    assert first.parent.parent == artifacts / "runs"
    assert (first.parent / "window.json").read_bytes() == window.read_bytes()
    assert (second.parent / "perception-manifest.json").read_bytes() == (visual_run / "manifest.json").read_bytes()
    assert len(project.available_context_runs()) == 2
    assert {entry["contextual_point_count"] for entry in project.available_context_runs()} == {1, 2}


# Mantém a contagem absoluta após a limpeza dos artifacts de trabalho: o viewer
# publicado ainda é a fonte das identidades que o usuário pode comparar.
def test_next_context_run_id_considers_published_runs(tmp_path: Path) -> None:
    """Continua a numeração a partir das runs já publicadas."""
    project = publication_project(tmp_path)
    published = project.root / "apps/map-explorer/web/public/runs/2026-09-12-run-007-original"
    published.mkdir(parents=True)
    workflows = Workflows(project)
    assert workflows._next_context_run_id("candidate").endswith("-run-008-candidate")
