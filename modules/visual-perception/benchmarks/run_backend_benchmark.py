"""Executor de CLI para o benchmark de seleção de backend da #174.

Carrega o conjunto representativo de frames corridor-02 (veja
``prepare_corridor02_frames.py``), roda cada candidato de um estágio através
de ``backend_benchmark.benchmark_candidate``, imprime uma tabela de
resultados, aplica ``execution_profile.select_research_quality_backend`` sob
o orçamento de referência de 8GB de VRAM, e escreve os resultados brutos em
``benchmarks/results/``.

Uso (a partir de ``modules/visual-perception``, com os extras ``ml``+``bench``
instalados e os frames já extraídos):

    python benchmarks/run_backend_benchmark.py --stage region_discovery
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Mirrors pyproject.toml's [tool.pytest.ini_options] pythonpath: these sibling
# packages aren't independently installable yet (#99/#100/#101, #103/#104).
for relative in ("../../contracts", "../../adapters/datasets", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / relative).resolve()))

from backend_benchmark import benchmark_candidate  # noqa: E402
from visual_perception.application.execution_profile import (  # noqa: E402
    select_research_quality_backend,
)

FRAMES_DIR = Path(__file__).resolve().parent / ".local" / "corridor-02-frames"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
GPU_MEMORY_BUDGET_GB = 8.0


# Lista os frames PNG já extraídos em FRAMES_DIR (ordenados), opcionalmente
# limitados a `limit` frames; falha explicitamente se
# prepare_corridor02_frames.py ainda não foi rodado. Chamada por main() para
# montar o conjunto de entrada do benchmark.
def _load_frames(limit: int | None) -> list[Path]:
    frames = sorted(FRAMES_DIR.glob("*.png"))
    if not frames:
        raise SystemExit(f"No frames found in {FRAMES_DIR}. Run prepare_corridor02_frames.py first.")
    return frames[:limit] if limit else frames


# Resolve, por nome de estágio, qual módulo de benchmarks/candidates/
# fornece a lista de candidatos daquele estágio (region_discovery,
# feature_extraction, language_embedding, multimodal_reasoning). Chamada por
# main() para despachar para o módulo de candidatos correto conforme o
# --stage pedido.
def _stage_module(stage: str):
    if stage == "region_discovery":
        from candidates import region_discovery as module
    elif stage == "feature_extraction":
        from candidates import feature_extraction as module
    elif stage == "language_embedding":
        from candidates import language_embedding as module
    elif stage == "multimodal_reasoning":
        from candidates import multimodal_reasoning as module
    else:
        raise SystemExit(f"Unknown stage: {stage!r}")
    return module


# Ponto de entrada de CLI: parseia os argumentos (--stage, --frames,
# --warmup-runs), carrega os frames e os candidatos do estágio escolhido,
# faz benchmark de cada candidato via benchmark_candidate, seleciona o
# vencedor via select_research_quality_backend sob o orçamento de VRAM de
# referência, e grava os resultados em JSON em benchmarks/results/.
# Executado via `python benchmarks/run_backend_benchmark.py --stage <nome>`.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["region_discovery", "feature_extraction", "language_embedding", "multimodal_reasoning"],
    )
    parser.add_argument("--frames", type=int, default=None, help="Cap the number of frames used.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()

    frames = _load_frames(args.frames)
    module = _stage_module(args.stage)
    candidate_specs = module.candidates(frames)

    results = []
    failures: list[dict[str, str]] = []
    for name, factory, run_once in candidate_specs:
        print(f"\n=== Benchmarking {name!r} on {len(frames)} frames ===")
        import torch

        try:
            candidate = benchmark_candidate(
                name, factory, run_once, warmup_runs=args.warmup_runs, measured_runs=len(frames)
            )
        except Exception as error:  # noqa: BLE001 - a broken candidate must not abort the others
            print(f"  FAILED: {error!r}")
            failures.append({"name": name, "reason": repr(error)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        print(
            f"  quality={candidate.quality_score:.4f}  "
            f"peak_vram_gb={candidate.peak_vram_gb:.3f}  "
            f"latency_s={candidate.latency_s:.3f}"
        )
        results.append(candidate)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit(f"Every candidate for {args.stage!r} failed: {failures}")

    winner = select_research_quality_backend(tuple(results), GPU_MEMORY_BUDGET_GB)
    print(f"\nSelected backend for {args.stage!r}: {winner.name}")
    if failures:
        print(f"Excluded (failed, not a memory-budget rejection): {[f['name'] for f in failures]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"benchmark-174-{args.stage}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "gpu_memory_budget_gb": GPU_MEMORY_BUDGET_GB,
                "frame_count": len(frames),
                "frames_dir": str(FRAMES_DIR),
                "candidates": [asdict(r) for r in results],
                "failed_candidates": failures,
                "selected": winner.name,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
