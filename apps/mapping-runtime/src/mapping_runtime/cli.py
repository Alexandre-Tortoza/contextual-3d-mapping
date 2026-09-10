"""CLI do composition root para workflows operacionais do M1."""

from __future__ import annotations

import argparse
import json
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
        help="contextualiza o slice FAST-LIO do corridor-02 com os keyframes de um trecho",
    )
    context.add_argument("--geometric-slice", type=Path, required=True)
    context.add_argument("--bag", type=Path, required=True)
    context.add_argument("--intrinsics", type=Path, required=True)
    context.add_argument("--extrinsics", type=Path, required=True)
    context.add_argument(
        "--window",
        type=Path,
        required=True,
        help="janela resolvida por `bag-window`, que identifica os keyframes",
    )
    context.add_argument(
        "--visual-run",
        type=Path,
        required=True,
        help="execução de visual-perception que contém frames/<frame-id>/",
    )
    context.add_argument(
        "--odometry",
        type=Path,
        default=None,
        help="trajetória do FAST-LIO; ignorada com aviso quando ausente",
    )
    context.add_argument("--ground-truth", type=Path, default=None)
    context.add_argument("--output", type=Path, required=True)
    window = commands.add_parser(
        "bag-window",
        help="resolve um trecho da rosbag e seus keyframes RGB nos dois relógios do arquivo",
    )
    window.add_argument("--bag", type=Path, required=True, help="rosbag de origem")
    window.add_argument(
        "--start-s",
        type=float,
        required=True,
        help="início da janela, em segundos após o primeiro frame RGB",
    )
    window.add_argument("--duration-s", type=float, required=True, help="duração da janela")
    window.add_argument(
        "--keyframe-interval-s",
        type=float,
        default=0.0,
        help="espaçamento entre keyframes; zero seleciona apenas o primeiro",
    )
    window.add_argument(
        "--lead-s",
        type=float,
        default=3.0,
        help="prefixo reproduzido antes da janela para o estimator convergir",
    )
    window.add_argument("--camera-topic", default="/camera_1/image_raw", help="tópico RGB")
    window.add_argument("--output", type=Path, default=None, help="arquivo JSON de destino")
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
        from .keyframe_inputs import resolve_keyframe_inputs

        keyframes, skipped, pose_anchor_ns = resolve_keyframe_inputs(
            options.window, options.visual_run
        )
        if skipped:
            print(f"AVISO: {len(skipped)} keyframes sem percepção visual: {', '.join(skipped)}")
        odometry = options.odometry
        if odometry is not None and not odometry.is_file():
            print(f"AVISO: odometria ausente em {odometry}; usando o ground-truth do dataset.")
            odometry = None
        destination = export_corridor02_context(
            Corridor02ContextRequest(
                geometric_slice=options.geometric_slice,
                bag=options.bag,
                intrinsics=options.intrinsics,
                extrinsics=options.extrinsics,
                keyframes=keyframes,
                destination=options.output,
                odometry=odometry,
                ground_truth=options.ground_truth,
                pose_anchor_ns=pose_anchor_ns,
            )
        )
        print(destination)
    elif options.command == "bag-window":
        # Importa a leitura de rosbag apenas no workflow real, mantendo demo e
        # testes mínimos executáveis sem a dependência opcional.
        from .bag_window import export_bag_window, resolve_bag_window

        window = resolve_bag_window(
            options.bag,
            start_s=options.start_s,
            duration_s=options.duration_s,
            keyframe_interval_s=options.keyframe_interval_s,
            lead_s=options.lead_s,
            camera_topic=options.camera_topic,
        )
        if options.output is not None:
            export_bag_window(window, options.output)
        print(json.dumps(window.to_payload(), indent=2))
    return 0
