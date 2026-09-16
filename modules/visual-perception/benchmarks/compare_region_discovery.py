"""Compara SAM3 e Florence-2 no mesmo frame com artifacts lado a lado.

Uso, a partir de ``modules/visual-perception``:

    python benchmarks/compare_region_discovery.py \
      --frame-id corridor-02-000

Cada braço executa a pipeline canônica completa. Somente
``region_discovery.backend`` e seu checkpoint variam; os demais stages,
prompts, frame de entrada e geometria de área permanecem iguais.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from compare_runs import compare, load_frame
from validate_reference_pipeline import (
    FRAMES_DIR,
    RESULTS_DIR,
    SEQUENCE_MASKS_DIR,
    ValidationOptions,
    load_dotenv,
    next_run_id,
    run_validation,
)

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parents[1]
_SEQUENCE_ID = "corridor-02"


# Produz uma única imagem de inspeção visual para que diferenças de cobertura,
# fragmentação e caixas possam ser vistas antes de interpretar os números do
# relatório. É chamada depois de ambos os runs terem persistido seus artifacts.
def write_side_by_side_overlay(
    baseline_overlay: Path,
    candidate_overlay: Path,
    *,
    baseline_name: str,
    candidate_name: str,
    output: Path,
) -> None:
    """Grava os overlays de dois backends lado a lado, identificados por nome.

    Argumentos:
        baseline_overlay: overlay produzido pelo primeiro backend.
        candidate_overlay: overlay produzido pelo segundo backend.
        baseline_name: rótulo do painel esquerdo.
        candidate_name: rótulo do painel direito.
        output: caminho PNG de saída.
    """
    baseline = Image.open(baseline_overlay).convert("RGB")
    candidate = Image.open(candidate_overlay).convert("RGB")
    width = baseline.width + candidate.width
    height = max(baseline.height, candidate.height) + 28
    comparison = Image.new("RGB", (width, height), "white")
    comparison.paste(baseline, (0, 28))
    comparison.paste(candidate, (baseline.width, 28))
    draw = ImageDraw.Draw(comparison)
    draw.text((6, 6), baseline_name, fill="black")
    draw.text((baseline.width + 6, 6), candidate_name, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output)


# Garante que os dois artifacts vieram dos mesmos pixels antes de produzir um
# relatório comparativo. Sem essa verificação, diferença de input pareceria
# diferença de modelo e invalidaria a análise.
def require_same_input(baseline_run: Path, candidate_run: Path, frame_id: str) -> None:
    """Valida que os dois runs registram o mesmo SHA-256 para o frame.

    Levanta:
        ValueError: quando algum frame não foi produzido ou os inputs divergem.
    """
    _, baseline_manifest = load_frame(baseline_run, frame_id)
    _, candidate_manifest = load_frame(candidate_run, frame_id)
    baseline_sha = (baseline_manifest or {}).get("input", {}).get("sha256")
    candidate_sha = (candidate_manifest or {}).get("input", {}).get("sha256")
    if not baseline_sha or baseline_sha != candidate_sha:
        raise ValueError("Os runs não registram o mesmo input SHA-256 para o frame comparado.")


# Executa os dois braços de discovery e reúne os artifacts sob uma mesma raiz
# datada. Existe para que o protocolo não dependa de duas invocações manuais
# que poderiam divergir por frame, máscara de área ou opções da pipeline.
def compare_region_discovery(options: ValidationOptions, *, output_root: Path) -> Path:
    """Executa SAM3 e Florence-2 no mesmo frame e retorna o relatório Markdown.

    Argumentos:
        options: opções comuns e reprodutíveis dos dois braços.
        output_root: diretório que receberá runs e comparação gerados.
    Retorna:
        caminho do relatório Markdown produzido.
    """
    if len(options.frame_ids) != 1:
        raise ValueError("compare_region_discovery requires exactly one frame_id.")
    frame_id = options.frame_ids[0]
    comparisons_dir = output_root / "comparisons"
    existing_run_ids = (path.name for path in comparisons_dir.glob("*")) if comparisons_dir.is_dir() else ()
    run_stamp = str(next_run_id(existing_run_ids, today=datetime.now(UTC).date(), name="region-discovery"))
    baseline = run_validation(
        dataclasses.replace(
            options,
            results_dir=output_root / "sam3",
            region_discovery_backend="sam3",
        )
    )
    candidate = run_validation(
        dataclasses.replace(
            options,
            results_dir=output_root / "florence2",
            region_discovery_backend="florence2",
        )
    )
    require_same_input(baseline, candidate, frame_id)

    report_dir = output_root / "comparisons" / run_stamp
    overlay_path = report_dir / f"{frame_id}-sam3-vs-florence2.png"
    write_side_by_side_overlay(
        baseline / "frames" / frame_id / "regions-overlay.png",
        candidate / "frames" / frame_id / "regions-overlay.png",
        baseline_name="SAM3",
        candidate_name="Florence-2",
        output=overlay_path,
    )
    report = compare(baseline, candidate, (frame_id,))
    report += (
        "\n## Inspeção visual de discovery\n\n"
        f"Overlay lado a lado: [{overlay_path.name}]({overlay_path.name}).\n\n"
        "As propostas e regiões são comparadas estruturalmente acima; a imagem "
        "permite inspecionar cobertura, fragmentação e geometria retangular do Florence-2.\n"
    )
    report_path = report_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# Constrói a CLI mínima do protocolo, expondo apenas variáveis que continuam
# iguais nos dois braços. Backends não são argumentos porque a comparação é
# definida especificamente como SAM3 versus Florence-2.
def _argument_parser() -> argparse.ArgumentParser:
    """Retorna o parser da comparação reproduzível de region discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "region-discovery-comparisons")
    parser.add_argument(
        "--sequence-masks",
        type=Path,
        default=SEQUENCE_MASKS_DIR / f"{_SEQUENCE_ID}.json",
    )
    parser.add_argument("--no-sequence-masks", action="store_true")
    parser.add_argument("--context-profile", choices=("baseline", "full"), default="full")
    return parser


# Traduz a CLI em uma única opção compartilhada e inicia os dois runs.
def main(argv: list[str] | None = None) -> None:
    """Executa o protocolo de comparação SAM3 versus Florence-2."""
    arguments = _argument_parser().parse_args(argv)
    load_dotenv(_REPOSITORY_ROOT / ".env")
    report = compare_region_discovery(
        ValidationOptions(
            frames_dir=arguments.frames_dir,
            results_dir=arguments.results_dir,
            frame_ids=(arguments.frame_id,),
            context_profile=arguments.context_profile,
            sequence_masks=None if arguments.no_sequence_masks else arguments.sequence_masks,
        ),
        output_root=arguments.results_dir,
    )
    print(f"Relatório: {report}")


if __name__ == "__main__":  # pragma: no cover - entrypoint de linha de comando
    main()
