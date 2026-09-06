"""Valida o pipeline canônico real de ponta a ponta em frames selecionados (#212).

Roda ``run_canonical_pipeline`` com os 4 backends reais benchmark-selecionados
(#174) sobre o conjunto representativo de frames do corridor-02 (ver
``prepare_corridor02_frames.py``), verificando ausência de OOM e VRAM dentro
do budget de referência, e gera amostras (JSON + overlay + resumo) para
revisão humana.

Uso (a partir de ``modules/visual-perception``, com os extras ``ml``
instalados e os frames já extraídos):

    python benchmarks/validate_reference_pipeline.py \
      --frame-id corridor-02-000 \
      --frame-id corridor-02-008 \
      --frame-id corridor-02-017
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_MODULE_ROOT / "tests"))
for relative in ("../../contracts", "../../adapters/datasets", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / relative).resolve()))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fixtures import image_observation  # noqa: E402
from render_overlay import render_overlay  # noqa: E402
from visual_perception.application.execution_profile import research_quality_config  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.application.pipeline import PipelineResult, run_canonical_pipeline  # noqa: E402
from visual_perception.application.relation_generation import generate_relations  # noqa: E402
from visual_perception.application.semantic_merge import merge_same_label_regions  # noqa: E402
from visual_perception.application.tiling import build_tiles  # noqa: E402
from visual_perception.config import (  # noqa: E402
    ModuleConfig,
    MultiContextConfig,
)
from visual_perception.domain.errors import VisualPerceptionError  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.region_evidence import EvidenceSlot, EvidenceState  # noqa: E402
from visual_perception.domain.visual_observation import VisualObservation  # noqa: E402
from visual_perception.infrastructure.adapters.factory import create_perception_ports  # noqa: E402
from visual_perception.infrastructure.serialization import serialize_observation  # noqa: E402

FRAMES_DIR = Path(__file__).resolve().parent / ".local" / "corridor-02-frames"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Faixa inferior da imagem ocupada pelo chassi/rodas do robô que carrega a
#: câmera fisheye do corridor-02 — fixa em todo frame porque a câmera está
#: montada no próprio robô (não é um objeto da cena, é o "ego-veículo").
#: Específico deste dataset/rig, por isso vive aqui e não em
#: visual_perception (que não deve conhecer qual robô gerou os dados; ver
#: AGENTS.md "Configuration ownership"). Confirmado visualmente em 3 frames
#: (000, 012, 017): chassi+rodas sempre abaixo de y=340.
_EGO_VEHICLE_ROW_START = 340


# Representa a seleção determinística e as opções de persistência de uma
# validação. Existe para separar a composição de CLI da execução auditável.
@dataclasses.dataclass(frozen=True)
class ValidationOptions:
    """Opções reproduzíveis de uma execução do validador real.

    Argumentos:
        frames_dir: diretório que contém os PNGs de entrada.
        results_dir: raiz em que o run versionado será persistido.
        frame_ids: IDs explícitos, na ordem pedida, ou tupla vazia para todos.
        limit: limite aplicado depois da seleção ordenada.
        semantic_merge: habilita o pós-processamento externo ao pipeline canônico.
    """

    frames_dir: Path = FRAMES_DIR
    results_dir: Path = RESULTS_DIR
    frame_ids: tuple[str, ...] = ()
    limit: int | None = None
    semantic_merge: bool = False
    context_profile: str = "full"

    # Rejeita perfis livres para que um typo não produza uma ablation diferente.
    def __post_init__(self) -> None:
        """Valida o perfil de evidência multi-contexto selecionado."""
        if self.context_profile not in {"baseline", "full"}:
            raise ValueError("context_profile must be 'baseline' or 'full'.")


# Preenche a faixa do chassi/rodas do robô com preto (mesma cor do vinheta
# do fisheye) para que SAM/VLM não a tratem como conteúdo de cena — sem
# isso, o carrinho aparece rotulado em todo frame, sempre a mesma
# distração não relacionada ao ambiente sendo mapeado.
def _mask_ego_vehicle(pixels: np.ndarray) -> np.ndarray:
    """Retorna os pixels com a área fixa do ego-veículo mascarada."""
    masked = pixels.copy()
    masked[_EGO_VEHICLE_ROW_START:, :, :] = 0
    return masked


# Retorna o hash curto do commit atual, ou "unknown" fora de um git worktree;
# usado no manifest para amarrar as amostras à revisão de código que as gerou.
def _git_revision() -> str:
    """Retorna a revisão Git curta usada na proveniência do run."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_MODULE_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# Calcula o digest do conteúdo de um frame sem depender de nome ou mtime.
def _sha256(path: Path) -> str:
    """Retorna o SHA-256 hexadecimal do arquivo informado."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Resolve IDs explícitos contra o diretório de frames e preserva sua ordem;
# quando não há IDs, usa ordenação lexical estável. Erros de seleção falham
# antes de carregar qualquer modelo caro.
def select_frame_paths(
    frames_dir: Path, frame_ids: tuple[str, ...] = (), limit: int | None = None
) -> tuple[Path, ...]:
    """Seleciona frames deterministicamente e rejeita IDs inválidos ou repetidos.

    Argumentos:
        frames_dir: diretório com frames PNG.
        frame_ids: IDs sem extensão, na ordem desejada.
        limit: quantidade máxima após a seleção; ``None`` mantém todos.
    Retorna:
        caminhos selecionados na ordem reproduzível.
    Levanta:
        ValueError: quando a seleção é vazia, repetida ou contém ID desconhecido.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame_ids must not contain duplicates.")

    available = {path.stem: path for path in sorted(frames_dir.glob("*.png"))}
    unknown = tuple(frame_id for frame_id in frame_ids if frame_id not in available)
    if unknown:
        raise ValueError(f"Unknown frame ids in {frames_dir}: {list(unknown)}.")
    selected = tuple(available[frame_id] for frame_id in frame_ids) if frame_ids else tuple(available.values())
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No frames selected from {frames_dir}.")
    return selected


# Resume o estado observado de cada slot sem confundir slot desabilitado
# (missing) com falha operacional (failed). Consumido pelo manifest por frame.
def evidence_state_counts(observation: VisualObservation) -> dict[str, dict[str, int]]:
    """Conta estados available/missing/failed para cada slot de evidência."""
    counts = {
        slot.value: {state.value: 0 for state in EvidenceState}
        for slot in EvidenceSlot
    }
    for region in observation.regions:
        for evidence in region.evidence:
            counts[evidence.slot.value][evidence.state.value] += 1
    return counts


# Reconstrói as chamadas de modelo a partir das fronteiras explícitas do
# pipeline. Existe porque eventos de load do lifecycle não equivalem a calls.
def model_call_counts(
    payload: ImagePayload, config: ModuleConfig, result: PipelineResult
) -> dict[str, int]:
    """Conta chamadas por capacidade e devolve também o total do frame."""
    region_count = len(result.observation.regions)
    evidence_calls = sum(metric.model_calls for metric in result.evidence_metrics)
    counts = {
        "region_discovery": len(build_tiles(payload, config.tiling)),
        "feature_extraction": 1 if region_count else 0,
        "language_aligned_evidence": evidence_calls,
        "scene_reasoning": 1,
        "region_reasoning": region_count,
    }
    return {**counts, "total": sum(counts.values())}


# Aplica o merge semântico apenas à cópia pós-processada, preservando os
# IDs e a geometria emitidos pelo pipeline canônico em seu próprio artifact.
def _semantic_merge_observation(
    observation: VisualObservation, config: ModuleConfig
) -> VisualObservation:
    """Retorna uma observação pós-processada sem alterar a saída canônica."""
    merged_regions = merge_same_label_regions(observation.regions)
    return dataclasses.replace(
        observation,
        regions=merged_regions,
        relations=generate_relations(merged_regions, config.merge),
    )


# Executa a validação real e persiste artifacts canônicos e opcionais em
# diretórios distintos. Esta função existe para tornar seleção e provenance
# testáveis sem acoplar o contract ao argparse.
def run_validation(options: ValidationOptions) -> Path:
    """Executa o pipeline real nos frames selecionados e retorna o diretório do run."""
    try:
        frames = select_frame_paths(options.frames_dir, options.frame_ids, options.limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not frames:
        raise SystemExit(
            f"No frames found in {options.frames_dir}. Run prepare_corridor02_frames.py first."
        )

    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    if options.context_profile == "baseline":
        config = dataclasses.replace(
            config,
            multi_context=MultiContextConfig(
                foreground_enabled=True,
                tight_crop_enabled=True,
                contextual_crop_enabled=False,
                scene_conditioned_enabled=False,
            ),
        )
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = options.results_dir / "samples" / run_id
    canonical_dir = out_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    postprocessed_dir = out_dir / "postprocessed" / "semantic-merge"
    if options.semantic_merge:
        postprocessed_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[str] = []
    frame_reports: list[dict[str, object]] = []

    for frame_path in frames:
        name = frame_path.stem
        print(f"\n=== {name} ===")
        frame_start = time.monotonic()
        metric_start = len(lifecycle.metrics)
        image = Image.open(frame_path).convert("RGB")
        pixels = _mask_ego_vehicle(np.array(image))
        image = Image.fromarray(pixels)  # keep the overlay/render in sync with what the pipeline saw
        payload = ImagePayload(pixels, width=pixels.shape[1], height=pixels.shape[0])
        observation_input = image_observation(
            observation_id=name, width=pixels.shape[1], height=pixels.shape[0]
        )

        try:
            result = run_canonical_pipeline(observation_input, payload, config, ports)
        except VisualPerceptionError as error:
            print(f"  FAILED: {error!r}")
            frame_reports.append(
                {
                    "frame_id": name,
                    "input": {
                        "path": str(frame_path.resolve()),
                        "sha256": _sha256(frame_path),
                        "width": pixels.shape[1],
                        "height": pixels.shape[0],
                    },
                    "failed": True,
                    "reason": repr(error),
                    "latency_s": time.monotonic() - frame_start,
                }
            )
            summary_rows.append(f"## {name}\n\n**FALHOU:** `{error!r}`\n")
            continue

        canonical_observation = result.observation
        json_path = canonical_dir / f"{name}.json"
        json_path.write_text(json.dumps(serialize_observation(result.observation), indent=2))

        overlay_path = canonical_dir / f"{name}.overlay.png"
        render_overlay(image, result.observation).save(overlay_path)

        postprocessed_count: int | None = None
        if options.semantic_merge:
            postprocessed = _semantic_merge_observation(canonical_observation, config)
            postprocessed_count = len(postprocessed.regions)
            (postprocessed_dir / f"{name}.json").write_text(
                json.dumps(serialize_observation(postprocessed), indent=2)
            )
            render_overlay(image, postprocessed).save(postprocessed_dir / f"{name}.overlay.png")

        scene_type = next(
            (c.value for c in result.observation.scene_context.claims if c.kind.value == "scene_type"),
            "?",
        )
        audit_status = "pass" if result.audit.passed else "FAIL"
        print(
            f"  canonical_regions={len(canonical_observation.regions)} "
            f"relations={len(result.observation.relations)} "
            f"interpretation_failures={len(result.region_interpretation_failures)} "
            f"audit={audit_status} warnings={len(result.audit.warnings)}"
        )
        frame_metrics = lifecycle.metrics[metric_start:]
        frame_latency_s = time.monotonic() - frame_start
        frame_reports.append(
            {
                "frame_id": name,
                "input": {
                    "path": str(frame_path.resolve()),
                    "sha256": _sha256(frame_path),
                    "width": pixels.shape[1],
                    "height": pixels.shape[0],
                },
                "failed": False,
                "canonical_region_count": len(canonical_observation.regions),
                "postprocessed_region_count": postprocessed_count,
                "relation_count": len(result.observation.relations),
                "interpretation_failure_count": len(result.region_interpretation_failures),
                "evidence_failure_count": len(result.evidence_failures),
                "calibration_failure_count": len(result.calibration_failures),
                "evidence_slots": evidence_state_counts(canonical_observation),
                "evidence_slot_metrics": [asdict(metric) for metric in result.evidence_metrics],
                "model_calls": model_call_counts(payload, config, result),
                "audit_passed": result.audit.passed,
                "audit_error_count": len(result.audit.errors),
                "audit_warning_count": len(result.audit.warnings),
                "latency_s": frame_latency_s,
                "model_lifecycle_events": len(frame_metrics),
                "peak_vram_bytes": max(
                    (metric.peak_vram_bytes or 0 for metric in frame_metrics), default=0
                ),
                "artifacts": {
                    "canonical_json": str(json_path.relative_to(out_dir)),
                    "canonical_overlay": str(overlay_path.relative_to(out_dir)),
                    "semantic_merge": (
                        str((postprocessed_dir / f"{name}.json").relative_to(out_dir))
                        if options.semantic_merge
                        else None
                    ),
                },
            }
        )
        summary_rows.append(
            f"## {name}\n\n"
            f"![{name}]({overlay_path.relative_to(out_dir)})\n\n"
            f"**scene_type:** {scene_type} · "
            f"**regiões canônicas:** {len(canonical_observation.regions)} · "
            f"**regiões pós-processadas:** {postprocessed_count if postprocessed_count is not None else 'desabilitado'} · "
            f"**relações:** {len(result.observation.relations)} · "
            f"**falhas de interpretação:** {len(result.region_interpretation_failures)} · "
            f"**audit:** {'✅ pass' if result.audit.passed else '❌ FAIL'} "
            f"({len(result.audit.warnings)} warnings)\n"
        )

    lifecycle.release_active()

    manifest = {
        "run_id": run_id,
        "git_revision": _git_revision(),
        "gpu_memory_budget_gb": config.gpu_memory_budget_gb,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "frame_count": len(frames),
        "ordered_frame_ids": [frame.stem for frame in frames],
        "frames_dir": str(options.frames_dir.resolve()),
        "selection": {"frame_ids": list(options.frame_ids), "limit": options.limit},
        "canonical_output": "canonical",
        "postprocessing": ["semantic_merge"] if options.semantic_merge else [],
        "context_profile": options.context_profile,
        "lifecycle_metrics": [asdict(m) for m in lifecycle.metrics],
        "frames": frame_reports,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    peak_vram_gb = max((m.peak_memory_bytes / (1024**3) for m in lifecycle.metrics), default=0.0)
    summary_header = (
        f"# Validação do pipeline real — {run_id}\n\n"
        f"Revisão: `{manifest['git_revision']}` · Frames: {len(frames)} · "
        f"Pico de VRAM observado: {peak_vram_gb:.2f} GB "
        f"(budget: {config.gpu_memory_budget_gb} GB)\n\n"
        "Ver `manifest.json` para configuração completa e log de estágios.\n\n"
    )
    (out_dir / "summary.md").write_text(summary_header + "\n".join(summary_rows))

    print(f"\nWrote samples to {out_dir}")
    print(f"Peak VRAM across the run: {peak_vram_gb:.2f} GB (budget {config.gpu_memory_budget_gb} GB)")
    return out_dir


# Constrói o parser do CLI em um helper testável e documenta que o merge
# semântico é opt-in, nunca parte silenciosa da saída canônica.
def _argument_parser() -> argparse.ArgumentParser:
    """Retorna o parser do validador reproduzível."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--frame-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--context-profile",
        choices=("baseline", "full"),
        default="full",
        help="baseline desliga contexto/cena; full exercita os quatro slots",
    )
    parser.add_argument(
        "--semantic-merge",
        action="store_true",
        help="persiste uma variante pós-processada separada da saída canônica",
    )
    return parser


# Ponto de entrada do script: traduz argumentos em ValidationOptions sem
# esconder defaults ou seleção de frames em estado global.
def main(argv: list[str] | None = None) -> None:
    """Executa o CLI de validação real."""
    arguments = _argument_parser().parse_args(argv)
    run_validation(
        ValidationOptions(
            frames_dir=arguments.frames_dir,
            results_dir=arguments.results_dir,
            frame_ids=tuple(arguments.frame_id),
            limit=arguments.limit,
            semantic_merge=arguments.semantic_merge,
            context_profile=arguments.context_profile,
        )
    )


if __name__ == "__main__":
    main()
