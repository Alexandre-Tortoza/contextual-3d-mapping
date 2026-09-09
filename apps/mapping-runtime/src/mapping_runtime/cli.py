"""CLI do composition root para workflows operacionais do M1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .demo import export_demo_slice
from .pcd_slice import export_pcd_slice


# Declara os comandos suportados em um parser isolado para permitir teste sem
# iniciar subprocessos ou acessar argumentos globais implicitamente.
def _parser() -> argparse.ArgumentParser:
    """Cria o parser da CLI do mapping-runtime.

    Retorna:
        parser configurado para os workflows disponíveis.
    """
    parser = argparse.ArgumentParser(prog="mapping-runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="gera um slice RGB–LiDAR sintético")
    demo.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m1-demo.json"),
        help="arquivo JSON de destino",
    )
    pcd_slice = commands.add_parser(
        "pcd-slice",
        help="amostra um mapa PCD binário para inspeção no map-explorer",
    )
    pcd_slice.add_argument("source", type=Path, help="arquivo PCD de origem")
    pcd_slice.add_argument("--output", type=Path, required=True, help="arquivo JSON de destino")
    pcd_slice.add_argument("--map-id", required=True, help="identidade do mapa")
    pcd_slice.add_argument("--map-frame", default="map", help="frame global do mapa")
    pcd_slice.add_argument(
        "--max-points",
        type=int,
        default=25_000,
        help="quantidade máxima de pontos enviada ao viewer",
    )
    context = commands.add_parser(
        "corridor-02-context",
        help="associa uma observação visual real ao slice FAST-LIO do corridor-02",
    )
    context.add_argument("--geometric-slice", type=Path, required=True)
    context.add_argument("--bag", type=Path, required=True)
    context.add_argument("--intrinsics", type=Path, required=True)
    context.add_argument("--extrinsics", type=Path, required=True)
    context.add_argument("--ground-truth", type=Path, required=True)
    context.add_argument("--visual-observation", type=Path, required=True)
    context.add_argument("--raw-image", type=Path, required=True)
    context.add_argument("--overlay-image", type=Path, required=True)
    context.add_argument("--valid-area-mask", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    context.add_argument("--camera-sequence-index", type=int, default=0)
    return parser


# Executa o comando selecionado e devolve código POSIX. Existe como fronteira
# fina para console script, ``python -m`` e testes de aplicação.
def main(arguments: Sequence[str] | None = None) -> int:
    """Executa a CLI do mapping-runtime.

    Argumentos:
        arguments: argumentos sem o nome do executável; usa ``sys.argv`` quando ausente.
    Retorna:
        código de saída do processo.
    """
    options = _parser().parse_args(arguments)
    if options.command == "demo":
        destination = export_demo_slice(options.output)
        print(destination)
    elif options.command == "pcd-slice":
        destination = export_pcd_slice(
            options.source,
            options.output,
            map_id=options.map_id,
            map_frame=options.map_frame,
            max_points=options.max_points,
        )
        print(destination)
    elif options.command == "corridor-02-context":
        # Importa dependências opcionais apenas no workflow real, mantendo o
        # demo e os testes mínimos executáveis sem rosbags/Pillow/PyYAML.
        from .corridor02_context import Corridor02ContextRequest, export_corridor02_context

        destination = export_corridor02_context(
            Corridor02ContextRequest(
                geometric_slice=options.geometric_slice,
                bag=options.bag,
                intrinsics=options.intrinsics,
                extrinsics=options.extrinsics,
                ground_truth=options.ground_truth,
                visual_observation=options.visual_observation,
                raw_image=options.raw_image,
                overlay_image=options.overlay_image,
                valid_area_mask=options.valid_area_mask,
                destination=options.output,
                camera_sequence_index=options.camera_sequence_index,
            )
        )
        print(destination)
    return 0
