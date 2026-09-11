"""Testes da orquestração da CLI com processos substituídos."""

from __future__ import annotations

import json
from pathlib import Path

from contextual_mapping_cli.models import ExtractionRequest
from contextual_mapping_cli.processes import ProcessRunner
from contextual_mapping_cli.project import Project
from contextual_mapping_cli.workflows import Workflows

from contextual_mapping_adapters import RosbagImageTopic, RosbagRecording


# Cria a topologia mínima usada pelo workflow sem copiar algoritmos ou dados do
# repositório real.
def _project_root(tmp_path: Path) -> Path:
    """Materializa uma raiz mínima de projeto para testes."""
    (tmp_path / "AGENTS.md").write_text("# teste", encoding="utf-8")
    (tmp_path / "Makefile").write_text("test:\n", encoding="utf-8")
    (tmp_path / "apps" / "mapping-runtime" / "src").mkdir(parents=True)
    (tmp_path / "adapters" / "datasets").mkdir(parents=True)
    (tmp_path / "contracts").mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "artifacts").mkdir()
    return tmp_path


# A estimativa deve refletir a política selecionada antes de percorrer ou
# desserializar todas as imagens do arquivo.
def test_estimate_extraction_distingue_amostra_e_todos_os_frames(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Confere cardinalidade prevista para as duas políticas."""
    workflows = Workflows(Project(_project_root(tmp_path)))
    recording = RosbagRecording(
        Path("run.bag"),
        100.0,
        (RosbagImageTopic("/rgb", "sensor_msgs/msg/Image", 1000),),
    )
    monkeypatch.setattr(workflows, "inspect_bag", lambda _path: recording)
    sampled = workflows.estimate_extraction(
        ExtractionRequest(Path("run.bag"), "sample", duration_s=10.0, keyframe_interval_s=2.0)
    )
    complete = workflows.estimate_extraction(
        ExtractionRequest(Path("run.bag"), "complete", duration_s=10.0, all_frames=True)
    )
    assert sampled.frame_count == 6
    assert complete.frame_count == 100


# O runner fake publica um manifest depois de registrar argv e cwd, permitindo
# verificar a composição sem carregar nenhum modelo.
class _PerceptionRunner(ProcessRunner):
    """Executor fake que simula a publicação de visual-perception."""

    # Registra a chamada e materializa a saída mínima esperada pelo workflow.
    def run(self, command, *, cwd, environment=None) -> None:  # type: ignore[no-untyped-def]
        """Simula um processo bem-sucedido e seu manifest."""
        self.command = tuple(command)
        self.cwd = cwd
        destination = cwd / "benchmarks" / "results" / "samples" / "run-test"
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            json.dumps({"frame_count": 1, "frames": []}), encoding="utf-8"
        )


# Bags genéricos precisam passar a ausência de máscara de forma explícita, pois
# o harness tem uma máscara corridor-02 como default histórico.
def test_run_perception_desabilita_mascara_para_bag_generico(tmp_path: Path) -> None:
    """Confere argv, ambiente e descoberta do novo run."""
    root = _project_root(tmp_path)
    module = root / "modules" / "visual-perception"
    python = module / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    frames = root / "artifacts" / "generic-frames"
    frames.mkdir()
    (frames / "generic-00000.png").write_bytes(b"png")
    runner = _PerceptionRunner()
    workflows = Workflows(Project(root), runner)
    run = workflows.run_perception(frames, None)
    assert "--no-sequence-masks" in runner.command
    assert runner.cwd == module
    assert run.name == "run-test"
