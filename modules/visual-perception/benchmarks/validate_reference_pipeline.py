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
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_MODULE_ROOT / "tests"))
for relative in ("src", "../../contracts", "../../adapters/datasets", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / relative).resolve()))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fixtures import image_observation  # noqa: E402
from frame_artifacts import (  # noqa: E402
    FRAME_ARTIFACT_LAYOUT_VERSION,
    FrameInputs,
    write_frame_artifacts,
)
from visual_perception.application.execution_profile import research_quality_config  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.application.observation_diagnostics import diagnose_observation  # noqa: E402
from visual_perception.application.pipeline import (  # noqa: E402
    PerceptionPorts,
    PipelineResult,
    run_canonical_pipeline,
)
from visual_perception.application.tiling import build_tiles  # noqa: E402
from visual_perception.config import (  # noqa: E402
    ImageAreaConfig,
    ModuleConfig,
    MultiContextConfig,
)
from visual_perception.domain.errors import VisualPerceptionError  # noqa: E402
from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.region_evidence import EvidenceSlot, EvidenceState  # noqa: E402
from visual_perception.domain.region_reasoning import SceneContextMode  # noqa: E402
from visual_perception.domain.visual_observation import VisualObservation  # noqa: E402
from visual_perception.infrastructure.adapters.factory import create_perception_ports  # noqa: E402

FRAMES_DIR = Path(__file__).resolve().parent / ".local" / "corridor-02-frames"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Como run_validation obtém seus backends. Existe para que um teste injete
#: fakes e exercite o layout de artifacts de ponta a ponta sem GPU. É uma
#: factory, e não um PerceptionPorts pronto, porque run_validation constrói a
#: config internamente e é dona do ModelLifecycleManager: a factory preserva
#: essa posse e tem exatamente a assinatura de create_perception_ports, então o
#: default é a própria função, sem wrapper.
PortsFactory = Callable[[ModuleConfig, ModelLifecycleManager], PerceptionPorts]

#: Onde vive a geometria de área declarada por sequência. É específica de
#: dataset e rig, por isso o *arquivo* mora aqui e não em ``visual_perception``,
#: que não deve conhecer qual robô gerou os dados (ver AGENTS.md,
#: "Configuração"). O módulo define o contract (``ImageAreaConfig``) e a aplica;
#: o harness escolhe qual geometria carregar.
SEQUENCE_MASKS_DIR = Path(__file__).resolve().parent / "sequence-masks"

#: Sequência cuja geometria é carregada para os frames de referência.
_SEQUENCE_ID = "corridor-02"


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
        sequence_masks: arquivo de geometria de área da sequência.
        context_profile: quais slots de evidência multi-contexto são extraídos.
        region_views: quais dessas views o reasoner recebe; ``None`` usa o default.
    """

    frames_dir: Path = FRAMES_DIR
    results_dir: Path = RESULTS_DIR
    frame_ids: tuple[str, ...] = ()
    limit: int | None = None
    context_profile: str = "full"
    #: Sobrescreve quais views o reasoner recebe (#203). ``None`` mantém o
    #: default da configuração. Existe para que a ablation de views seja
    #: reproduzível pela linha de comando, e não por edição de código.
    region_views: tuple[str, ...] | None = None
    #: Se as claims de cena acompanham cada região no prompt. ``None`` mantém o
    #: default da configuração (local-first). É o par textual de
    #: ``region_views``: os dois canais de contexto são ablatáveis
    #: separadamente, com o resto do prompt inalterado.
    scene_context_mode: str | None = None
    #: Sobrescreve o checkpoint do reasoner multimodal, mantendo **todo o
    #: resto** constante: mesmos frames, mesmas views, mesmo prompt, mesma
    #: temperatura, mesmos tetos dos estágios contextuais. Existe porque a
    #: comparação de backends só é interpretável quando a arquitetura não se
    #: mexe entre os braços, e esse era exatamente o problema da seleção
    #: anterior — ela foi feita antes de os estágios contextuais existirem.
    #: ``None`` mantém o checkpoint da configuração de referência.
    reasoning_checkpoint: str | None = None
    #: Geometria de área da sequência (círculo útil da lente e silhueta do
    #: rig). A exclusão acontece dentro do pipeline, na filtragem de proposals,
    #: e nunca pintando pixels: até a #202 este harness tinha um
    #: ``--mask-ego-vehicle`` que zerava a faixa inferior antes do SAM, e a #212
    #: mediu que aquilo corrompia a análise de cena e colapsava 45 de 45 regiões
    #: em ``curved wall``. Passar ``None`` roda sem geometria declarada.
    sequence_masks: Path | None = SEQUENCE_MASKS_DIR / f"{_SEQUENCE_ID}.json"

    # Rejeita perfis livres para que um typo não produza uma ablation diferente.
    def __post_init__(self) -> None:
        """Valida o perfil de evidência multi-contexto selecionado."""
        if self.context_profile not in {"baseline", "full"}:
            raise ValueError("context_profile must be 'baseline' or 'full'.")
        if self.scene_context_mode is not None and self.scene_context_mode not in {
            mode.value for mode in SceneContextMode
        }:
            raise ValueError(
                "scene_context_mode must be "
                f"{sorted(mode.value for mode in SceneContextMode)} or None."
            )


# Lê a geometria de área versionada de uma sequência e a converte na
# configuração que o módulo consome. Existe no harness porque a geometria é
# específica de rig e dataset: o módulo define o contract e aplica a exclusão,
# a composição escolhe qual geometria entra.
def load_image_area_config(path: Path | None) -> ImageAreaConfig:
    """Carrega a geometria declarada da sequência, ou a configuração vazia.

    Argumentos:
        path: arquivo de geometria da sequência, ou ``None`` para rodar sem
            nenhuma exclusão declarada.
    Retorna:
        a configuração de área correspondente.
    """
    if path is None:
        return ImageAreaConfig()
    payload = json.loads(path.read_text())
    return ImageAreaConfig(
        valid_area=_geometry_from_dict(payload.get("valid_area")),
        ego_vehicle=_geometry_from_dict(payload.get("ego_vehicle")),
    )


# Lê a resolução para a qual a geometria foi medida. Existe porque aplicar uma
# geometria de 640x480 a um frame de outro tamanho rejeitaria o frame inteiro
# sem que nada no artifact explicasse a causa — exatamente o modo de falha
# silenciosa que esta rodada existe para eliminar.
def sequence_masks_resolution(path: Path | None) -> tuple[int, int] | None:
    """Retorna a resolução declarada no artifact de geometria, ou ``None``."""
    if path is None:
        return None
    payload = json.loads(path.read_text())
    width, height = payload.get("image_width"), payload.get("image_height")
    if width is None or height is None:
        return None
    return int(width), int(height)


# Converte uma entrada do artifact de sequência em ImageAreaGeometry. Isolada
# porque as duas áreas usam o mesmo formato e uma segunda leitura divergiria.
def _geometry_from_dict(payload: dict[str, Any] | None) -> ImageAreaGeometry | None:
    """Converte um bloco de geometria do artifact, ou ``None`` se vazio."""
    if not payload:
        return None
    circle = payload.get("circle")
    polygons = tuple(
        tuple((float(x), float(y)) for x, y in polygon) for polygon in payload.get("polygons", [])
    )
    if circle is None and not polygons:
        return None
    return ImageAreaGeometry(
        circle=None
        if circle is None
        else CircleArea(float(circle["cx"]), float(circle["cy"]), float(circle["r"])),
        polygons=polygons,
    )


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
    stage_calls = dict(result.stage_model_calls)
    counts = {
        "region_discovery": len(build_tiles(payload, config.tiling)),
        "feature_extraction": 1 if region_count else 0,
        "language_aligned_evidence": evidence_calls,
        "scene_reasoning": 1,
        "region_reasoning": region_count,
        # Os três estágios da #214/#204/#206 aparecem nomeados: um custo que só
        # existisse dentro de ``total`` não permitiria decidir se vale o preço.
        "hypothesis_support_text": stage_calls.get("hypothesis_support_text", 0),
        "region_refinement": stage_calls.get("region_refinement", 0),
        "semantic_relations": stage_calls.get("semantic_relations", 0),
    }
    return {**counts, "total": sum(counts.values())}


# Resolve a configuração de um run a partir das opções, sem executar nada.
# Existe separada de ``run_validation`` porque a decisão de configuração é o que
# uma comparação de backends precisa poder inspecionar: a #218 exige que apenas
# o checkpoint difira entre dois braços, e isso é verificável sem GPU.
def resolve_config(options: ValidationOptions) -> ModuleConfig:
    """Retorna a ``ModuleConfig`` que ``options`` seleciona.

    Argumentos:
        options: as opções reproduzíveis do run.
    Retorna:
        a configuração de referência com os overrides declarados aplicados.
    """
    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    # A geometria de área da sequência entra na configuração do módulo, e não
    # num passo do harness: assim ela participa do fingerprint e do manifest, e
    # a exclusão acontece dentro do pipeline em vez de sobre os pixels.
    config = dataclasses.replace(
        config, image_area=load_image_area_config(options.sequence_masks)
    )
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
    if options.region_views is not None:
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning, region_views=options.region_views
            ),
        )
    if options.reasoning_checkpoint is not None:
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning, checkpoint=options.reasoning_checkpoint
            ),
        )
    if options.scene_context_mode is not None:
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning, scene_context_mode=options.scene_context_mode
            ),
        )
    return config


# Executa a validação real e persiste artifacts canônicos e opcionais em
# diretórios distintos. Esta função existe para tornar seleção e provenance
# testáveis sem acoplar o contract ao argparse.
def run_validation(
    options: ValidationOptions, ports_factory: PortsFactory | None = None
) -> Path:
    """Executa o pipeline real nos frames selecionados e retorna o diretório do run.

    Argumentos:
        options: seleção de frames e opções de persistência do run.
        ports_factory: como obter os backends. ``None`` usa os reais, que é o
            que produção faz; um teste injeta fakes por aqui para exercitar o
            layout de artifacts sem GPU.
    Retorna:
        o diretório versionado em que o run foi persistido.
    """
    try:
        frames = select_frame_paths(options.frames_dir, options.frame_ids, options.limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not frames:
        raise SystemExit(
            f"No frames found in {options.frames_dir}. Run prepare_corridor02_frames.py first."
        )

    config = resolve_config(options)
    masks_resolution = sequence_masks_resolution(options.sequence_masks)

    lifecycle = ModelLifecycleManager()
    ports = (ports_factory or create_perception_ports)(config, lifecycle)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = options.results_dir / "samples" / run_id
    frames_out_dir = out_dir / "frames"
    frames_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[str] = []
    frame_reports: list[dict[str, object]] = []

    for frame_path in frames:
        name = frame_path.stem
        print(f"\n=== {name} ===")
        frame_start = time.monotonic()
        metric_start = len(lifecycle.metrics)
        raw_pixels = np.array(Image.open(frame_path).convert("RGB"))
        height, width = raw_pixels.shape[0], raw_pixels.shape[1]
        # Os pixels de origem são imutáveis, sem exceção. A exclusão do rig e
        # da área fora da lente acontece dentro do pipeline, sobre máscaras
        # (ver application/proposal_filtering.py); este harness não tem mais
        # como pintar a entrada.
        if masks_resolution is not None and masks_resolution != (width, height):
            raise SystemExit(
                f"{name}: frame is {width}x{height} but {options.sequence_masks} declares "
                f"{masks_resolution[0]}x{masks_resolution[1]}. Applying the geometry anyway would "
                "silently reject the whole frame."
            )
        area_masks = config.image_area.rasterize(width, height)
        ego_mask = None if area_masks.ego_vehicle is None else area_masks.ego_vehicle.data
        valid_mask = None if area_masks.valid_area is None else area_masks.valid_area.data
        payload = ImagePayload(raw_pixels, width=width, height=height)
        raw_payload = ImagePayload(raw_pixels, width=width, height=height)
        observation_input = image_observation(observation_id=name, width=width, height=height)

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
                        "width": width,
                        "height": height,
                    },
                    "failed": True,
                    "reason": repr(error),
                    "latency_s": time.monotonic() - frame_start,
                }
            )
            summary_rows.append(f"## {name}\n\n**FALHOU:** `{error!r}`\n")
            continue

        canonical_observation = result.observation
        frame_latency_s = time.monotonic() - frame_start
        diagnostics = diagnose_observation(
            canonical_observation,
            discovered_proposals=len(result.proposals) + len(result.rejected_proposals),
            kept_proposals=result.proposals,
            proposal_rejections=result.rejected_proposals,
            area_masks=result.area_masks,
            ego_overlap_threshold=config.proposal_filter.max_ego_overlap,
            valid_area_threshold=config.proposal_filter.min_valid_overlap,
        )
        frame_dir = frames_out_dir / name
        artifacts = write_frame_artifacts(
            frame_dir,
            inputs=FrameInputs(
                raw=raw_payload,
                pipeline_input=payload,
                ego_mask=ego_mask,
                valid_area_mask=valid_mask,
            ),
            result=result,
            diagnostics=diagnostics,
            extra_diagnostics={
                "frame_id": name,
                "input_sha256": _sha256(frame_path),
                "latency_s": frame_latency_s,
                "scene_context_mode": config.multimodal_reasoning.scene_context_mode,
                "region_views": list(config.multimodal_reasoning.region_views),
            },
        )
        overlay_path = frame_dir / artifacts["regions_overlay"]


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
        print(
            f"  proposals={diagnostics.proposal_count} "
            f"dominant_label={diagnostics.mode_collapse.dominant_label!r} "
            f"dominant_fraction={diagnostics.mode_collapse.dominant_fraction:.2f} "
            f"scene_echo={diagnostics.scene_echo_label_count}"
        )
        frame_metrics = lifecycle.metrics[metric_start:]
        frame_reports.append(
            {
                "frame_id": name,
                "input": {
                    "path": str(frame_path.resolve()),
                    "sha256": _sha256(frame_path),
                    "width": width,
                    "height": height,
                },
                "failed": False,
                "canonical_region_count": len(canonical_observation.regions),
                "relation_count": len(result.observation.relations),
                "interpretation_failure_count": len(result.region_interpretation_failures),
                "evidence_failure_count": len(result.evidence_failures),
                "calibration_failure_count": len(result.calibration_failures),
                # ``None`` significa que o backend denso configurado executou.
                # Um valor aqui é a única forma de o manifest distinguir um run
                # que usou o backend pedido de um que caiu para o fallback —
                # sem isso, o checklist "ausência de fallback silencioso" do
                # handoff não teria como ser verificado.
                "feature_fallback_reason": result.feature_fallback_reason,
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
                "proposal_count": diagnostics.proposal_count,
                "dominant_label": diagnostics.mode_collapse.dominant_label,
                "dominant_label_fraction": diagnostics.mode_collapse.dominant_fraction,
                "distinct_labels": diagnostics.mode_collapse.distinct_labels,
                "scene_echo_label_count": diagnostics.scene_echo_label_count,
                # O que os estágios de contexto produziram e o que custaram.
                # Sem estes campos, ligar ou desligar qualquer um deles não
                # mudaria nada de comparável entre dois manifests.
                "contextual": asdict(diagnostics.contextual),
                "signal_failure_count": len(result.signal_failures),
                "relation_failure_count": len(result.relation_failures),
                "refinement": [
                    {
                        "iteration": step.iteration,
                        "previous_evidence": list(step.previous_evidence),
                        "new_evidence": list(step.new_evidence),
                        "producer": step.producer,
                        "config_fingerprint": step.config_fingerprint,
                        "targets": [
                            {
                                "region_id": target.region_id,
                                "reasons": [reason.value for reason in target.reasons],
                            }
                            for target in step.targets
                        ],
                        "refined_region_ids": list(step.refined_region_ids),
                        "failures": [asdict(failure) for failure in step.failures],
                    }
                    for step in result.refinement_history
                ],
                "artifacts": {
                    artifact: str((frame_dir / relative).relative_to(out_dir))
                    for artifact, relative in artifacts.items()
                }
                | {
                },
            }
        )
        summary_rows.append(
            f"## {name}\n\n"
            f"![{name}]({overlay_path.relative_to(out_dir)})\n\n"
            f"**scene_type:** {scene_type} · "
            f"**regiões canônicas:** {len(canonical_observation.regions)} · "
            f"**relações:** {len(result.observation.relations)} · "
            f"**falhas de interpretação:** {len(result.region_interpretation_failures)} · "
            f"**audit:** {'✅ pass' if result.audit.passed else '❌ FAIL'} "
            f"({len(result.audit.warnings)} warnings)\n"
        )

    lifecycle.release_active()

    manifest = {
        "run_id": run_id,
        "region_views_override": list(options.region_views) if options.region_views else None,
        "git_revision": _git_revision(),
        "gpu_memory_budget_gb": config.gpu_memory_budget_gb,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "frame_count": len(frames),
        "ordered_frame_ids": [frame.stem for frame in frames],
        "frames_dir": str(options.frames_dir.resolve()),
        "selection": {"frame_ids": list(options.frame_ids), "limit": options.limit},
        "canonical_output": "frames",
        "frame_artifact_layout": FRAME_ARTIFACT_LAYOUT_VERSION,
        "context_profile": options.context_profile,
        "scene_context_mode": config.multimodal_reasoning.scene_context_mode,
        "sequence_masks": None if options.sequence_masks is None else str(options.sequence_masks),
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
        "--reasoning-checkpoint",
        default=None,
        help=(
            "Compara um checkpoint de raciocínio multimodal diferente mantendo todo o "
            "resto constante (#218). Trocar prompt e modelo na mesma comparação torna "
            "as duas mudanças ininterpretáveis."
        ),
    )
    parser.add_argument(
        "--context-profile",
        choices=("baseline", "full"),
        default="full",
        help="baseline desliga contexto/cena; full exercita os quatro slots",
    )
    parser.add_argument(
        "--region-view",
        action="append",
        default=[],
        dest="region_views",
        help="view de região enviada ao reasoner; repita para compor a ablation (#203)",
    )
    parser.add_argument(
        "--scene-context-mode",
        choices=tuple(mode.value for mode in SceneContextMode),
        default=None,
        help=(
            "local_first não envia claims de cena ao reasoner; context_assisted envia. "
            "O resto do prompt é idêntico nos dois, então a ablation isola uma variável"
        ),
    )
    parser.add_argument(
        "--sequence-masks",
        type=Path,
        default=SEQUENCE_MASKS_DIR / f"{_SEQUENCE_ID}.json",
        help=(
            "geometria de área da sequência (círculo útil da lente e silhueta do rig); "
            "passe um caminho inexistente para rodar sem exclusão declarada"
        ),
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
            context_profile=arguments.context_profile,
            reasoning_checkpoint=arguments.reasoning_checkpoint,
            region_views=tuple(arguments.region_views) or None,
            scene_context_mode=arguments.scene_context_mode,
            sequence_masks=arguments.sequence_masks if arguments.sequence_masks.is_file() else None,
        )
    )


if __name__ == "__main__":
    main()
