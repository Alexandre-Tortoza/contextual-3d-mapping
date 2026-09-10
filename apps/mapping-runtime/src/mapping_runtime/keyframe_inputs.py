"""Resolução dos artifacts de percepção que correspondem aos keyframes de um trecho."""

from __future__ import annotations

import json
from pathlib import Path

from .corridor02_context import Corridor02Keyframe

FRAME_ARTIFACTS = {
    "visual_observation": "observation.json",
    "raw_image": "raw.png",
    "overlay_image": "regions-overlay.png",
    "valid_area_mask": "valid-area-mask.png",
}


# Deriva a identidade do frame a partir da posição no stream RGB. Existe porque
# essa é a única identidade que sobrevive da extração de imagens até a execução
# de visual-perception, que não retém o índice original do bag.
def frame_id_for(camera_sequence_index: int) -> str:
    """Retorna a identidade canônica de um keyframe extraído do bag."""
    return f"corridor-02-{camera_sequence_index:05d}"


# Casa a janela resolvida com uma execução de visual-perception. Keyframes sem
# percepção são devolvidos separadamente, e não silenciosamente ignorados, para
# que uma execução parcial de GPU continue utilizável sem esconder o que falta.
def resolve_keyframe_inputs(
    window: Path, visual_run: Path
) -> tuple[tuple[Corridor02Keyframe, ...], tuple[str, ...], int]:
    """Resolve os keyframes de um trecho contra uma execução de percepção.

    Argumentos:
        window: janela gravada por ``mapping-runtime bag-window``.
        visual_run: diretório de execução com ``frames/<frame-id>/``.
    Retorna:
        keyframes resolvidos, identidades sem percepção e a âncora de pose.
    Levanta:
        ValueError: se nenhum keyframe da janela tiver percepção disponível.
    """
    payload = json.loads(window.read_text(encoding="utf-8"))
    frames_directory = visual_run / "frames"
    resolved: list[Corridor02Keyframe] = []
    missing: list[str] = []
    for entry in payload["keyframes"]:
        index = int(entry["sequence_index"])
        frame_id = frame_id_for(index)
        directory = frames_directory / frame_id
        artifacts = {name: directory / filename for name, filename in FRAME_ARTIFACTS.items()}
        if not all(path.is_file() for path in artifacts.values()):
            missing.append(frame_id)
            continue
        resolved.append(
            Corridor02Keyframe(
                frame_id=frame_id,
                camera_sequence_index=index,
                header_timestamp_ns=int(entry["header_timestamp_ns"]),
                bag_timestamp_ns=int(entry["bag_timestamp_ns"]),
                **artifacts,
            )
        )
    if not resolved:
        raise ValueError(
            f"no keyframe of {window} has visual perception artifacts under {frames_directory}."
        )
    anchor_ns = int(payload["start_header_ns"]) - int(float(payload["lead_s"]) * 1_000_000_000)
    return tuple(resolved), tuple(missing), anchor_ns
