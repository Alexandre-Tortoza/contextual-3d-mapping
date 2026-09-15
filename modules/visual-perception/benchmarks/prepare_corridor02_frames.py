"""Extrai um conjunto representativo de frames RGB do corridor-02.

Este script é um harness fino para benchmarks (#174) e validação end-to-end
(#190). A inspeção e decodificação da rosbag pertencem ao adapter de datasets e
são consumidas daqui por sua API pública; `visual-perception` continua sem
possuir sampling de dataset ou detalhes de encoding ROS.

Uso (a partir de `modules/visual-perception`, com o extra `bench` instalado):

    python benchmarks/prepare_corridor02_frames.py

Escreve frames PNG, amostrados uniformemente ao longo da sequência, em
`benchmarks/.local/corridor-02-frames/` (no gitignore — não versionado).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for _dependency in (
    _REPOSITORY_ROOT / "contracts",
    _REPOSITORY_ROOT / "datasets",
    _REPOSITORY_ROOT / "adapters" / "datasets",
):
    sys.path.insert(0, str(_dependency))

from contextual_mapping_adapters import (  # noqa: E402
    extract_rosbag_frames,
    extract_uniform_rosbag_frames,
)

DEFAULT_BAG = Path(__file__).resolve().parents[3] / "datasets" / "raw" / "corridor-02" / "corridor-02.bag"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / ".local" / "corridor-02-frames"


# Preserva a função pública histórica do harness enquanto delega parsing e
# decodificação ao adapter de datasets, que é o dono dessa integração.
def extract_frames(bag_path: Path, out_dir: Path, *, count: int, topic: str | None = None) -> list[Path]:
    """Extrai a amostra uniforme histórica do corridor-02.

    Argumentos:
        bag_path: rosbag de origem.
        out_dir: diretório de saída.
        count: quantidade de frames.
        topic: tópico RGB explícito ou detecção automática.
    Retorna:
        caminhos dos PNGs escritos.
    """
    return extract_uniform_rosbag_frames(
        bag_path,
        out_dir,
        count=count,
        topic=topic,
        frame_id_prefix="corridor-02",
    )


# Extrai exatamente os keyframes de uma janela resolvida por
# `mapping-runtime bag-window`, nomeando cada PNG pelo índice original do frame
# no stream RGB do bag. Existe porque a amostragem uniforme acima perde essa
# identidade (ela numera pela ordem da amostra), e é justamente ela que a
# composição contextual precisa para reencontrar o frame e sua pose.
def extract_window_frames(bag_path: Path, out_dir: Path, window_path: Path) -> list[Path]:
    """Extrai os keyframes declarados em um artifact de janela.

    Argumentos:
        bag_path: rosbag de origem.
        out_dir: diretório de saída.
        window_path: artifact JSON produzido por ``bag-window``.
    Retorna:
        caminhos dos PNGs escritos.
    """
    payload = json.loads(window_path.read_text(encoding="utf-8"))
    selected = {
        int(keyframe["bag_timestamp_ns"]): int(keyframe["sequence_index"])
        for keyframe in payload["keyframes"]
    }
    return extract_rosbag_frames(
        bag_path,
        out_dir,
        topic=str(payload["camera_topic"]),
        frame_indices_by_timestamp=selected,
        frame_id_prefix=str(payload.get("frame_id_prefix", "corridor-02")),
    )


# Ponto de entrada de CLI: parseia os argumentos (bag, out_dir, count, topic,
# window), valida que o bag existe, e extrai frames por amostragem uniforme ou
# pelos keyframes de uma janela. Executado via
# `python benchmarks/prepare_corridor02_frames.py`.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--topic", type=str, default=None, help="sobrescreve o tópico RGB detectado")
    parser.add_argument(
        "--window",
        type=Path,
        default=None,
        help="window JSON de `mapping-runtime bag-window`; extrai seus keyframes",
    )
    args = parser.parse_args()

    if not args.bag.exists():
        raise SystemExit(f"rosbag não encontrada: {args.bag}")

    if args.window is not None:
        written = extract_window_frames(args.bag, args.out_dir, args.window)
    else:
        written = extract_frames(args.bag, args.out_dir, count=args.count, topic=args.topic)
    print(f"Foram escritos {len(written)} frames em {args.out_dir}")


if __name__ == "__main__":
    main()
