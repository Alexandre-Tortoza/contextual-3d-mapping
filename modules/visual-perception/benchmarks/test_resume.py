"""Testes de resume/checkpoint de ``run_validation`` (processamento em lotes de um bag grande).

Cobrem o que ``test_validate_reference_pipeline.py``/``test_frame_artifacts.py`` não
exercitavam: reabrir um run existente via ``--run-dir``, pular frames já concluídos,
escrever o manifest incrementalmente (sobrevivendo a uma falha no meio do laço), e
encerrar de forma limpa sob SIGTERM sem perder o que já foi processado. Tudo com
fakes, sem GPU.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

import validate_reference_pipeline as vrp  # noqa: E402
from contextual_mapping_adapters import ExtractedFrameProvenance, write_frame_provenance  # noqa: E402
from PIL import Image  # noqa: E402

from fixtures_ports import default_ports  # noqa: E402
from validate_reference_pipeline import (  # noqa: E402
    ValidationOptions,
    _InterruptFlag,
    _read_manifest,
    _replace_frame_report,
    _resume_prior,
    run_validation,
)
from visual_perception.domain.region_reasoning import TemporalPriorMode  # noqa: E402


# Frame sintético com um blob deslocado por índice, para que cada um produza
# ao menos uma região sem depender de nenhum dado real do corridor-02.
def _write_frame(frames_dir: Path, frame_id: str, sequence_index: int) -> None:
    pixels = np.full((480, 640, 3), 255, dtype=np.uint8)
    offset = 20 * sequence_index
    pixels[20 : 60 + offset, 20:70] = (200, 30, 30)
    frames_dir.mkdir(parents=True, exist_ok=True)
    image_path = frames_dir / f"{frame_id}.png"
    Image.fromarray(pixels).save(image_path)
    write_frame_provenance(
        image_path,
        ExtractedFrameProvenance(
            recording_id="synthetic",
            topic="/camera/image_raw",
            frame_id="camera_optical_frame",
            timestamp_ns=1_000_000_000 + sequence_index * 100_000_000,
            sequence_index=sequence_index,
        ),
    )


def _fake_ports_factory(config: object, lifecycle: object) -> object:
    return default_ports()


# Envolve run_canonical_pipeline, chamado exatamente uma vez por frame
# efetivamente processado (nunca por frame pulado via skip de resume) — ao
# contrário de um port individual, que pode ser acionado várias vezes dentro
# do mesmo frame (tiling, refinamento). É o ponto certo para contar "quantos
# frames o laço realmente processou" e para injetar um efeito colateral
# (crash, sinal) num frame específico de forma determinística.
def _hook_pipeline_calls(monkeypatch: pytest.MonkeyPatch, on_call: object) -> None:
    original = vrp.run_canonical_pipeline

    def wrapper(*args: object, **kwargs: object) -> object:
        on_call()
        return original(*args, **kwargs)

    monkeypatch.setattr(vrp, "run_canonical_pipeline", wrapper)


# Um resume só é útil se realmente pular o trabalho de GPU/modelo, não só
# evitar duplicar a entrada no manifest — por isso a asserção central aqui é
# a contagem de chamadas ao pipeline, não a lista de frame_ids.
def test_run_dir_skips_frames_already_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confere que reabrir o run com --run-dir não reprocessa frames concluídos."""
    frames_dir = tmp_path / "frames-in"
    for index, frame_id in enumerate(("f-000", "f-001", "f-002")):
        _write_frame(frames_dir, frame_id, index)
    run_dir = tmp_path / "out" / "corridor-02-full"

    calls: list[str] = []
    _hook_pipeline_calls(monkeypatch, lambda: calls.append("call"))

    out_dir = run_validation(
        ValidationOptions(
            frames_dir=frames_dir,
            results_dir=tmp_path / "unused",
            frame_ids=("f-000", "f-001"),
            run_dir=run_dir,
            sequence_masks=None,
        ),
        ports_factory=_fake_ports_factory,
    )
    assert out_dir == run_dir
    assert len(calls) == 2

    out_dir_again = run_validation(
        ValidationOptions(
            frames_dir=frames_dir,
            results_dir=tmp_path / "unused",
            frame_ids=("f-000", "f-001", "f-002"),
            run_dir=run_dir,
            sequence_masks=None,
        ),
        ports_factory=_fake_ports_factory,
    )
    assert out_dir_again == run_dir
    # f-000 e f-001 não disparam uma segunda chamada do pipeline: só f-002,
    # o único frame novo, é processado.
    assert len(calls) == 3

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert manifest["ordered_frame_ids"] == ["f-000", "f-001", "f-002"]


# O requisito central do resume: uma falha no meio do laço não pode perder o
# manifest dos frames que já terminaram antes dela.
def test_manifest_survives_a_crash_mid_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confere que o manifest incremental sobrevive a uma exceção não tratada no meio do laço."""
    frames_dir = tmp_path / "frames-in"
    for index, frame_id in enumerate(("f-000", "f-001")):
        _write_frame(frames_dir, frame_id, index)
    run_dir = tmp_path / "out" / "corridor-02-full"

    seen: list[str] = []

    def on_call() -> None:
        seen.append("call")
        if len(seen) == 2:
            raise RuntimeError("falha simulada de GPU no segundo frame")

    _hook_pipeline_calls(monkeypatch, on_call)

    with pytest.raises(RuntimeError, match="falha simulada"):
        run_validation(
            ValidationOptions(
                frames_dir=frames_dir,
                results_dir=tmp_path / "unused",
                frame_ids=("f-000", "f-001"),
                run_dir=run_dir,
                sequence_masks=None,
            ),
            ports_factory=_fake_ports_factory,
        )

    manifest = _read_manifest(run_dir)
    assert manifest is not None
    assert manifest["frame_count"] == 1
    assert manifest["ordered_frame_ids"] == ["f-000"]
    assert (run_dir / "frames" / "f-000" / "diagnostics.json").is_file()
    assert not (run_dir / "frames" / "f-001" / "diagnostics.json").is_file()


# SIGTERM não deve abortar o frame em processamento nem propagar como
# exceção — só impedir que o próximo frame comece.
def test_sigterm_stops_before_the_next_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confere encerramento gracioso: o frame em curso termina, o seguinte nunca começa."""
    frames_dir = tmp_path / "frames-in"
    for index, frame_id in enumerate(("f-000", "f-001", "f-002")):
        _write_frame(frames_dir, frame_id, index)
    run_dir = tmp_path / "out" / "corridor-02-full"

    seen: list[str] = []

    def on_call() -> None:
        seen.append("call")
        if len(seen) == 1:
            os.kill(os.getpid(), signal.SIGTERM)

    _hook_pipeline_calls(monkeypatch, on_call)

    out_dir = run_validation(
        ValidationOptions(
            frames_dir=frames_dir,
            results_dir=tmp_path / "unused",
            frame_ids=("f-000", "f-001", "f-002"),
            run_dir=run_dir,
            sequence_masks=None,
        ),
        ports_factory=_fake_ports_factory,
    )
    assert out_dir == run_dir
    # f-000 recebeu o sinal no meio do seu próprio processamento e ainda
    # assim terminou (a checagem só acontece entre frames); f-001/f-002
    # nunca começaram.
    assert len(seen) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ordered_frame_ids"] == ["f-000"]


def test_interrupt_flag_sets_on_signal_and_restores_previous_handler() -> None:
    """Confere que o sinalizador captura o sinal e devolve o handler anterior ao restaurar."""
    previous_calls: list[int] = []
    previous_handler = signal.signal(signal.SIGTERM, lambda signum, frame: previous_calls.append(signum))
    try:
        flag = _InterruptFlag()
        flag.install()
        os.kill(os.getpid(), signal.SIGTERM)
        assert flag.requested is True
        assert previous_calls == []

        flag.restore()
        os.kill(os.getpid(), signal.SIGTERM)
        assert previous_calls == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def test_replace_frame_report_deduplicates_by_frame_id() -> None:
    """Confere que reprocessar um frame substitui, em vez de duplicar, seu relatório."""
    reports: list[dict[str, object]] = [{"frame_id": "f-000", "failed": True}]
    _replace_frame_report(reports, {"frame_id": "f-000", "failed": False})
    assert reports == [{"frame_id": "f-000", "failed": False}]


def test_read_manifest_returns_none_when_absent_or_corrupted(tmp_path: Path) -> None:
    """Confere o fallback seguro para run-dir novo ou com manifest truncado."""
    assert _read_manifest(tmp_path) is None
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert _read_manifest(tmp_path) is None


# prior_from só lê embedding_id/vector; validar isso end-to-end via
# write_frame_artifacts real garante que a reconstrução do resume usa
# exatamente o mesmo artifact que um run contínuo teria produzido.
def test_resume_prior_reconstructs_from_last_completed_frame(tmp_path: Path) -> None:
    """Confere que o prior do resume vem do frame concluído certo, não de qualquer um."""
    frames_dir = tmp_path / "frames-in"
    for index, frame_id in enumerate(("f-000", "f-001", "f-002")):
        _write_frame(frames_dir, frame_id, index)
    run_dir = tmp_path / "out" / "corridor-02-full"
    run_validation(
        ValidationOptions(
            frames_dir=frames_dir,
            results_dir=tmp_path / "unused",
            frame_ids=("f-000", "f-001"),
            run_dir=run_dir,
            sequence_masks=None,
            temporal_prior_mode=TemporalPriorMode.BOX_OVERLAP.value,
        ),
        ports_factory=_fake_ports_factory,
    )

    # O predecessor de f-002 (o próximo pendente) é f-001, o último dos dois
    # já concluídos — não f-000. Sem observation.json de f-000 no diretório
    # (removido abaixo), a reconstrução só pode ter vindo de f-001.
    (run_dir / "frames" / "f-000" / "observation.json").unlink()
    prior = _resume_prior(run_dir / "frames", (frames_dir / "f-002.png",), {"f-000", "f-001"})
    assert prior is not None
