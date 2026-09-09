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
    return 0
