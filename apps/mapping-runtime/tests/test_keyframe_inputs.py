"""Testes do casamento entre a janela de um trecho e a execução de percepção."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mapping_runtime.keyframe_inputs import frame_id_for, resolve_keyframe_inputs

SECOND_NS = 1_000_000_000


# Escreve uma janela mínima no formato publicado por `bag-window`.
def _write_window(directory: Path, indices: tuple[int, ...]) -> Path:
    """Persiste uma janela de teste com os keyframes informados."""
    payload = {
        "camera_topic": "/camera_1/image_raw",
        "start_header_ns": 100 * SECOND_NS,
        "end_header_ns": 130 * SECOND_NS,
        "play_offset_s": 97.0,
        "play_duration_s": 33.0,
        "lead_s": 3.0,
        "keyframes": [
            {
                "sequence_index": index,
                "header_timestamp_ns": 100 * SECOND_NS + position * 2 * SECOND_NS,
                "bag_timestamp_ns": 900 * SECOND_NS + position * 2 * SECOND_NS,
            }
            for position, index in enumerate(indices)
        ],
    }
    destination = directory / "window.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


# Materializa os quatro artifacts que uma execução de percepção produz por frame.
def _write_perception(run: Path, frame_id: str) -> None:
    """Cria os artifacts de percepção de um frame."""
    directory = run / "frames" / frame_id
    directory.mkdir(parents=True)
    for name in ("observation.json", "raw.png", "regions-overlay.png", "valid-area-mask.png"):
        (directory / name).write_bytes(b"")


# A identidade precisa ser o índice original no bag, e não a ordem da amostragem:
# é ela que permite reencontrar o frame, seu instante e sua pose.
def test_frame_id_usa_o_indice_original_do_bag() -> None:
    """Confere o formato estável da identidade de keyframe."""
    assert frame_id_for(4234) == "corridor-02-04234"
    assert frame_id_for(0) == "corridor-02-00000"


# A âncora de pose precisa considerar o prefixo reproduzido antes da janela,
# porque o mapa começa na pose do primeiro scan processado — e não no primeiro
# frame pedido.
def test_ancora_de_pose_recua_o_prefixo_da_janela(tmp_path: Path) -> None:
    """Confere que a âncora corresponde ao início real do mapa."""
    window = _write_window(tmp_path, (10,))
    run = tmp_path / "run"
    _write_perception(run, "corridor-02-00010")
    keyframes, missing, anchor_ns = resolve_keyframe_inputs(window, run)
    assert len(keyframes) == 1
    assert missing == ()
    assert anchor_ns == 97 * SECOND_NS


# Uma execução parcial de GPU precisa continuar utilizável, mas o que faltou não
# pode desaparecer em silêncio.
def test_keyframes_sem_percepcao_sao_reportados(tmp_path: Path) -> None:
    """Confere que frames sem artifacts são devolvidos separadamente."""
    window = _write_window(tmp_path, (10, 20, 30))
    run = tmp_path / "run"
    _write_perception(run, "corridor-02-00010")
    _write_perception(run, "corridor-02-00030")
    keyframes, missing, _ = resolve_keyframe_inputs(window, run)
    assert [item.camera_sequence_index for item in keyframes] == [10, 30]
    assert missing == ("corridor-02-00020",)


# Sem nenhum frame utilizável a composição não tem o que fazer, e falhar aqui é
# mais claro do que produzir um artifact sem contexto algum.
def test_falha_quando_nenhum_keyframe_tem_percepcao(tmp_path: Path) -> None:
    """Confere a falha explícita quando a execução de percepção não cobre a janela."""
    window = _write_window(tmp_path, (10,))
    run = tmp_path / "run"
    (run / "frames").mkdir(parents=True)
    with pytest.raises(ValueError, match="no keyframe"):
        resolve_keyframe_inputs(window, run)
