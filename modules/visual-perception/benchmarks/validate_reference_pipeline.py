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
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _MODULE_ROOT.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for relative in ("src", "../../contracts", "../../adapters/datasets", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / relative).resolve()))

import numpy as np  # noqa: E402
from contextual_mapping_adapters import (  # noqa: E402
    ROSBAG_CLOCK_ID,
    ExtractedFrameProvenance,
    read_frame_provenance,
)
from contextual_mapping_contracts import (  # noqa: E402
    FrameId,
    ObservationReference,
    SourceArtifactReference,
    Timestamp,
)
from PIL import Image  # noqa: E402

from frame_artifacts import (  # noqa: E402
    FRAME_ARTIFACT_LAYOUT_VERSION,
    FrameInputs,
    write_frame_artifacts,
)
from visual_perception.application.execution_profile import research_quality_config  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager, StageMetrics  # noqa: E402
from visual_perception.application.observation_diagnostics import diagnose_observation  # noqa: E402
from visual_perception.application.pipeline import (  # noqa: E402
    PerceptionPorts,
    PipelineResult,
    run_canonical_pipeline,
)
from visual_perception.application.temporal_prior import prior_from  # noqa: E402
from visual_perception.application.tiling import build_tiles  # noqa: E402
from visual_perception.config import (  # noqa: E402
    ConceptGroundingConfig,
    ImageAreaConfig,
    ModuleConfig,
    MultiContextConfig,
    SceneConceptDiscoveryConfig,
    TilingConfig,
)
from visual_perception.domain.errors import VisualPerceptionError  # noqa: E402
from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry  # noqa: E402
from visual_perception.domain.image_observation import ImageObservation  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.region_evidence import EvidenceSlot, EvidenceState  # noqa: E402
from visual_perception.domain.region_reasoning import (  # noqa: E402
    SceneContextMode,
    ScenePrior,
    TemporalPriorMode,
)
from visual_perception.domain.visual_observation import VisualObservation  # noqa: E402
from visual_perception.infrastructure.adapters.factory import create_perception_ports  # noqa: E402
from visual_perception.infrastructure.adapters.gemini_reasoning_backend import (  # noqa: E402
    DEFAULT_GEMINI_ROBOTICS_ER_MODEL,
    RemoteReasoningCall,
)

FRAMES_DIR = Path(__file__).resolve().parent / ".local" / "corridor-02-frames"

#: Variantes de tiling comparadas na #277. Vivem aqui para que o harness completo e o
#: benchmark só de discovery (``tiling_benchmark.py``) usem exatamente as mesmas.
TILING_VARIANTS: dict[str, TilingConfig] = {
    "1x1": TilingConfig(multi_scale_enabled=False),
    "2x2": TilingConfig(multi_scale_enabled=True, tile_grid="2x2"),
    "2x2-discard": TilingConfig(multi_scale_enabled=True, tile_grid="2x2", discard_tile_border_truncations=True),
    "3x3": TilingConfig(multi_scale_enabled=True, tile_grid="3x3"),
    "3x3-discard": TilingConfig(multi_scale_enabled=True, tile_grid="3x3", discard_tile_border_truncations=True),
}
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
    #: Troca o backend do reasoner mantendo prompts, views e estágios (#276).
    #: ``gemini_robotics_er`` envia frames e crops para a API do Gemini; ``None``
    #: mantém o Qwen local da configuração de referência.
    reasoning_backend: str | None = None
    #: Variante de tiling do benchmark da #277 (``1x1``, ``2x2``, ``2x2-discard``,
    #: ``3x3``, ``3x3-discard``); ``None`` mantém a da configuração de referência.
    tiling_variant: str | None = None
    #: Liga descoberta de conceitos na cena e grounding por SAM3 PCS (#277) como
    #: fonte adicional de propostas.
    concept_discovery: bool = False
    #: Encadeia o que cada frame afirmou no frame seguinte (prior temporal).
    #: A ordem dos frames é uma decisão de composição, e por isso vive aqui e
    #: não no módulo: ``visual_perception`` recebe apenas "isto foi afirmado
    #: antes", sem timestamp nem pose. ``None`` mantém o default da
    #: configuração, que é desligado.
    temporal_prior_mode: str | None = None
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


# Recusa uma geometria de área pedida que não existe. Existe porque a CLI
# convertia um caminho inexistente em "sem geometria", e um erro de digitação
# rodava o frame inteiro sem exclusão do rig e da vinheta, sem nada no
# manifest que denunciasse. Rodar sem geometria é pedido explícito
# (``--no-sequence-masks``), nunca consequência de um caminho errado.
def require_sequence_masks(path: Path | None) -> None:
    """Valida que a geometria de área pedida existe.

    Argumentos:
        path: arquivo de geometria, ou ``None`` quando o run não declara nenhuma.
    Levanta:
        ValueError: se um caminho foi pedido e não é um arquivo.
    """
    if path is not None and not path.is_file():
        raise ValueError(
            f"sequence masks file not found: {path}. Pass --no-sequence-masks to run without "
            "declared area geometry."
        )


# Pico de VRAM de um conjunto de estágios, ou ``None`` quando nenhum mediu GPU.
# Existe para que o manifest não registre zero onde não houve medida.
def _peak_vram_bytes(metrics: tuple[StageMetrics, ...]) -> int | None:
    """Retorna o maior pico de VRAM medido, ou ``None`` sem medida de GPU."""
    measured = [metric.peak_vram_bytes for metric in metrics if metric.peak_vram_bytes is not None]
    return max(measured, default=None)


# Resume a memória do run para o summary e o terminal, com GPU e host sob
# nomes distintos. Antes, o "pico de VRAM" do summary vinha de
# ``peak_memory_bytes``, que é RSS do host quando não há CUDA, e discordava do
# manifest do mesmo run.
def _memory_summary(metrics: tuple[StageMetrics, ...], budget_gb: float) -> str:
    """Retorna a linha de memória do run, sem apresentar RSS do host como VRAM."""
    vram = _peak_vram_bytes(metrics)
    vram_text = (
        "Pico de VRAM: não medido (nenhum estágio rodou em CUDA)"
        if vram is None
        else f"Pico de VRAM: {vram / 1024**3:.2f} GB (budget: {budget_gb} GB)"
    )
    host = max((metric.peak_cpu_rss_bytes for metric in metrics), default=None)
    host_text = "" if host is None else f" · Pico de RSS do host: {host / 1024**3:.2f} GB"
    return vram_text + host_text


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


# Resume no manifest a proveniência com que a observação do frame foi montada,
# para que o run registre de onde vieram timestamp, sensor e artifact.
def _provenance_record(observation: ImageObservation) -> dict[str, object]:
    """Retorna a proveniência da observação de entrada em forma serializável."""
    source = observation.source
    return {
        "observation_id": source.observation_id,
        "dataset_id": source.dataset_id,
        "sequence_id": source.sequence_id,
        "sensor_id": source.sensor_id,
        "sequence_index": source.sequence_index,
        "timestamp_ns": source.timestamp.nanoseconds,
        "clock_id": source.timestamp.clock_id,
        "frame_id": source.frame_id.value,
        "artifact_uri": observation.image.uri,
    }


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


# Lê a proveniência registrada na extração de cada frame selecionado, antes de
# qualquer modelo ser carregado. Existe porque o PNG não carrega timestamp nem
# posição no stream: sem o registro, a validação antiga montava a observação
# com valores fixos de um fixture de teste, e todos os frames saíam com o
# mesmo instante (#239). Um frame sem registro interrompe o run.
def load_frame_provenance(frames: tuple[Path, ...]) -> dict[Path, ExtractedFrameProvenance]:
    """Lê e valida a proveniência de todos os frames selecionados.

    Argumentos:
        frames: PNGs selecionados para o run.
    Retorna:
        a proveniência de cada frame, indexada pelo caminho.
    Levanta:
        ValueError: se algum frame não tiver registro, tiver registro inválido
            ou não declarar frame de coordenadas.
    """
    provenance: dict[Path, ExtractedFrameProvenance] = {}
    for frame_path in frames:
        try:
            record = read_frame_provenance(frame_path)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"{frame_path.stem}: {error}") from error
        if record.frame_id is None:
            raise ValueError(
                f"{frame_path.stem}: the source message declares no header.frame_id, so the "
                "observation has no coordinate frame to carry."
            )
        provenance[frame_path] = record
    return provenance


# Monta a observação canônica de um frame a partir do que a extração
# registrou, sem nenhum valor inventado: identidade pelo nome do frame,
# gravação como dataset e sequência, tópico como sensor, e o artifact
# apontando para o próprio PNG processado.
def observation_from_frame(
    frame_path: Path, provenance: ExtractedFrameProvenance, *, width: int, height: int
) -> ImageObservation:
    """Constrói a ``ImageObservation`` de um frame com sua proveniência real.

    Argumentos:
        frame_path: PNG processado.
        provenance: registro lido por ``load_frame_provenance``.
        width: largura dos pixels carregados.
        height: altura dos pixels carregados.
    Retorna:
        a observação de entrada do pipeline.
    """
    assert provenance.frame_id is not None, "load_frame_provenance rejects frames without frame_id"
    source = ObservationReference(
        observation_id=frame_path.stem,
        dataset_id=provenance.recording_id,
        sequence_id=provenance.recording_id,
        sensor_id=provenance.topic,
        sequence_index=provenance.sequence_index,
        timestamp=Timestamp(nanoseconds=provenance.timestamp_ns, clock_id=ROSBAG_CLOCK_ID),
        frame_id=FrameId(provenance.frame_id),
    )
    image = SourceArtifactReference(uri=frame_path.resolve().as_uri(), media_type="image/png")
    return ImageObservation(width=width, height=height, encoding="rgb8", image=image, source=source)


# Resume o estado observado de cada slot sem confundir slot desabilitado
# (missing) com falha operacional (failed). Consumido pelo manifest por frame.
def evidence_state_counts(observation: VisualObservation) -> dict[str, dict[str, int]]:
    """Conta estados available/missing/failed para cada slot de evidência."""
    counts = {
        slot.value: {state.value: 0 for state in EvidenceState}
        for slot in EvidenceSlot
    }
    for region in observation.all_regions:
        for evidence in region.evidence:
            counts[evidence.slot.value][evidence.state.value] += 1
    return counts


# Reconstrói as chamadas de modelo a partir das fronteiras explícitas do
# pipeline. Existe porque eventos de load do lifecycle não equivalem a calls.
def model_call_counts(
    payload: ImagePayload, config: ModuleConfig, result: PipelineResult
) -> dict[str, int]:
    """Conta chamadas por capacidade e devolve também o total do frame."""
    region_count = len(result.observation.all_regions)
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
        "scene_concept_discovery": stage_calls.get("scene_concept_discovery", 0),
        "concept_grounding": stage_calls.get("concept_grounding", 0),
    }
    return {**counts, "total": sum(counts.values())}


# Resume a geometria produzida por discovery antes e depois da filtragem e do
# merge. Existe para comparar configurações de discovery sem tratar contagem bruta de masks
# como qualidade: tamanho, sobreposição e fragmentação explicam se propostas
# extras carregam cobertura nova ou apenas repetem a mesma evidência.
def discovery_telemetry(
    payload: ImagePayload, result: PipelineResult
) -> dict[str, int | float | None]:
    """Calcula telemetria geométrica de region discovery para um frame.

    Argumentos:
        payload: frame que define a área de normalização das masks.
        result: saída do pipeline com proposals cruas, filtradas e regiões finais.
    Retorna:
        contagens, estatísticas de área, sobreposição média e fragmentação.
    """
    frame_area = payload.width * payload.height
    raw = result.discovered_proposals
    kept = result.proposals
    areas = np.asarray([proposal.mask.area() / frame_area for proposal in kept], dtype=np.float64)
    overlaps = [
        left.mask.iou(right.mask)
        for index, left in enumerate(kept)
        for right in kept[index + 1 :]
    ]
    region_count = len(result.observation.all_regions)
    return {
        "raw_proposal_count": len(raw),
        "kept_proposal_count": len(kept),
        "merged_region_count": region_count,
        "area_fraction_min": None if not len(areas) else float(areas.min()),
        "area_fraction_median": None if not len(areas) else float(np.median(areas)),
        "area_fraction_max": None if not len(areas) else float(areas.max()),
        "mean_pairwise_iou": None if not overlaps else float(np.mean(overlaps)),
        "fragmentation_ratio": None if not region_count else len(kept) / region_count,
    }


# Resume custo e falhas das consultas a um reasoner remoto. Existe porque latência,
# tokens e rate limit da API não aparecem nas métricas de VRAM do lifecycle; ``None``
# quando o reasoner é local e não registra consultas.
def remote_reasoning_summary(calls: tuple[RemoteReasoningCall, ...] | None) -> dict[str, object] | None:
    """Agrega a telemetria de consultas remotas do run.

    Argumentos:
        calls: consultas registradas pelo adapter remoto, ou ``None``.
    Retorna:
        contagens, latência, tokens e falhas transitórias por motivo.
    """
    if calls is None:
        return None
    transient: dict[str, int] = {}
    for call in calls:
        for reason in call.transient_failures:
            transient[reason] = transient.get(reason, 0) + 1
    latencies = [call.latency_s for call in calls]
    return {
        "model": calls[0].model if calls else None,
        "calls": len(calls),
        "failed_calls": sum(not call.succeeded for call in calls),
        "calls_by_operation": {op: sum(call.operation == op for call in calls) for op in sorted({c.operation for c in calls})},
        "total_latency_s": sum(latencies),
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "prompt_tokens": sum(call.prompt_tokens or 0 for call in calls),
        "output_tokens": sum(call.output_tokens or 0 for call in calls),
        "thought_tokens": sum(call.thought_tokens or 0 for call in calls),
        "transient_failures": transient,
    }


# Resume o que a descoberta de conceitos (#277) produziu e quanto disso virou região.
# ``concepts_without_region`` é o gatilho natural do refinamento adaptativo: um
# conceito que o VLM viu mas o grounding não localizou.
def concept_discovery_report(result: PipelineResult) -> dict[str, object] | None:
    """Retorna conceitos, descartes e a contribuição deles para as regiões do frame.

    Argumentos:
        result: saída do pipeline para o frame.
    Retorna:
        o relatório, ou ``None`` quando a descoberta de conceitos estava desligada.
    """
    if result.scene_concepts is None:
        return None
    concept_by_proposal = {p.proposal_id: p.concept for p in result.discovered_proposals if p.concept}
    kept_concepts = {p.concept for p in result.proposals if p.concept}
    regions_with_concept = 0
    regions_only_from_concepts = 0
    grounded_concepts: set[str] = set()
    for region in result.observation.all_regions:
        contributing = [concept_by_proposal.get(pid) for pid in region.contributing_proposal_ids]
        concepts = {concept for concept in contributing if concept}
        grounded_concepts |= concepts
        regions_with_concept += int(bool(concepts))
        regions_only_from_concepts += int(bool(contributing) and all(contributing))
    texts = [concept.text for concept in result.scene_concepts.concepts]
    return {
        "prompt_version": result.scene_concepts.provenance.prompt_version,
        "concepts": [
            {"text": c.text, "kind": c.kind.value, "confidence": c.confidence} for c in result.scene_concepts.concepts
        ],
        "discarded": [list(item) for item in result.scene_concepts.discarded],
        "concept_proposals": len(concept_by_proposal),
        "concept_proposals_after_area_filter": sum(1 for p in result.proposals if p.concept),
        "regions_with_concept": regions_with_concept,
        "regions_only_from_concepts": regions_only_from_concepts,
        "concepts_without_region": [text for text in texts if text not in grounded_concepts],
        "concepts_filtered_out": [text for text in texts if text not in kept_concepts],
    }


# Carrega variáveis de um ``.env`` sem sobrescrever o ambiente. Existe só no ponto
# de entrada: a chave nunca passa pela config nem pelo manifest.
def load_dotenv(path: Path) -> None:
    """Exporta pares ``CHAVE=valor`` de ``path`` que ainda não estão no ambiente."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


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
    config = research_quality_config(real_backends=True)
    # A geometria de área da sequência entra na configuração do módulo, e não
    # num passo do harness: assim ela participa do fingerprint e do manifest, e
    # a exclusão acontece dentro do pipeline em vez de sobre os pixels.
    config = dataclasses.replace(
        config, image_area=load_image_area_config(options.sequence_masks)
    )
    if options.context_profile == "baseline":
        # Desligar o crop contextual exige tirá-lo também de quem o consome: a
        # config recusa pedir um slot que não é produzido.
        without_context = tuple(
            name
            for name in config.hypothesis_support.slots
            if name != EvidenceSlot.CONTEXTUAL_CROP.value
        )
        escalation_without_context = tuple(
            name
            for name in config.refinement.escalation_views
            if name != EvidenceSlot.CONTEXTUAL_CROP.value
        )
        config = dataclasses.replace(
            config,
            multi_context=MultiContextConfig(
                foreground_enabled=True,
                tight_crop_enabled=True,
                contextual_crop_enabled=False,
                scene_conditioned_enabled=False,
            ),
            hypothesis_support=dataclasses.replace(config.hypothesis_support, slots=without_context),
            refinement=dataclasses.replace(
                config.refinement, escalation_views=escalation_without_context
            ),
        )
    if options.region_views is not None:
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning, region_views=options.region_views
            ),
        )
    if options.reasoning_backend == "gemini_robotics_er":
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning,
                backend="gemini_robotics_er",
                checkpoint=DEFAULT_GEMINI_ROBOTICS_ER_MODEL,
                load_in_4bit=False,
            ),
        )
    if options.tiling_variant is not None:
        config = dataclasses.replace(config, tiling=TILING_VARIANTS[options.tiling_variant])
    if options.concept_discovery:
        config = dataclasses.replace(
            config,
            scene_concept_discovery=SceneConceptDiscoveryConfig(enabled=True),
            concept_grounding=ConceptGroundingConfig(backend="sam3"),
        )
    if options.reasoning_checkpoint is not None:
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning, checkpoint=options.reasoning_checkpoint
            ),
        )
    if options.temporal_prior_mode is not None:
        # A versão do prompt é bumpada **junto** com o modo, e não em separado,
        # porque o texto dos prompts é um literal no adapter e nada bumpa a
        # versão sozinho: dois braços com prompts diferentes declarando a mesma
        # versão seriam indistinguíveis na proveniência de cada claim.
        enabled = options.temporal_prior_mode != TemporalPriorMode.DISABLED.value
        config = dataclasses.replace(
            config,
            multimodal_reasoning=dataclasses.replace(
                config.multimodal_reasoning,
                temporal_prior_mode=options.temporal_prior_mode,
                prompt_version="v9" if enabled else config.multimodal_reasoning.prompt_version,
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
        frame_provenance = load_frame_provenance(frames)
        require_sequence_masks(options.sequence_masks)
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
    # O prior do próximo frame é derivado do resultado deste. Este laço é o
    # único lugar do sistema que sabe que um frame precede outro: o módulo
    # recebe um ScenePrior sem nenhum metadado de tempo ou pose.
    prior_enabled = (
        config.multimodal_reasoning.temporal_prior_mode != TemporalPriorMode.DISABLED.value
    )
    prior: ScenePrior | None = None

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
        # Cópia independente: a garantia "o pipeline recebeu os pixels de
        # origem" só é verificável contra um buffer que o pipeline não alcança.
        # Com o mesmo array nos dois lados a comparação era verdadeira por
        # identidade, e um pré-processamento destrutivo passava em silêncio.
        raw_payload = ImagePayload(raw_pixels.copy(), width=width, height=height)
        observation_input = observation_from_frame(
            frame_path, frame_provenance[frame_path], width=width, height=height
        )

        try:
            result = run_canonical_pipeline(observation_input, payload, config, ports, prior)
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
                        "provenance": _provenance_record(observation_input),
                    },
                    "failed": True,
                    "reason": repr(error),
                    "latency_s": time.monotonic() - frame_start,
                }
            )
            summary_rows.append(f"## {name}\n\n**FALHOU:** `{error!r}`\n")
            # Um frame que falhou não deixa herança: manter o prior do frame
            # anterior faria uma afirmação atravessar um buraco da sequência e
            # alcançar um viewpoint que ninguém observou.
            prior = None
            continue

        canonical_observation = result.observation
        prior = (
            prior_from(canonical_observation, result.visual_embeddings) if prior_enabled else None
        )
        frame_latency_s = time.monotonic() - frame_start
        diagnostics = diagnose_observation(
            canonical_observation,
            discovered_proposals=len(result.proposals) + len(result.rejected_proposals),
            kept_proposals=result.proposals,
            proposal_rejections=result.rejected_proposals,
            region_suppressions=result.suppressed_regions,
            prior_assignments=result.prior_assignments,
            prior_applied=prior_enabled,
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
                "temporal_prior_mode": config.multimodal_reasoning.temporal_prior_mode,
                "prompt_version": config.multimodal_reasoning.prompt_version,
                "region_views": list(config.multimodal_reasoning.region_views),
            },
            config=config,
        )
        overlay_path = frame_dir / artifacts["regions_overlay"]


        scene_type = next(
            (c.value for c in result.observation.scene_context.claims if c.kind.value == "scene_type"),
            "?",
        )
        audit_status = "pass" if result.audit.passed else "FAIL"
        print(
            f"  canonical_regions={len(canonical_observation.all_regions)} "
            f"published={len(canonical_observation.regions)} "
            f"structural_context={len(canonical_observation.structural_context)} "
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
                    "provenance": _provenance_record(observation_input),
                },
                "failed": False,
                # Continua contando a observação inteira, para que a série
                # histórica desta métrica siga comparável com os runs
                # anteriores à política de publicação contextual. Quanto disso
                # chegou ao output público está nos dois campos seguintes.
                "canonical_region_count": len(canonical_observation.all_regions),
                "published_region_count": len(canonical_observation.regions),
                "structural_context_count": len(canonical_observation.structural_context),
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
                # ``None`` quando nenhum estágio mediu GPU: zero seria uma medida
                # que não aconteceu. A memória do host fica num campo próprio.
                "peak_vram_bytes": _peak_vram_bytes(frame_metrics),
                "peak_host_rss_bytes": max(
                    (metric.peak_cpu_rss_bytes for metric in frame_metrics), default=None
                ),
                "proposal_count": diagnostics.proposal_count,
                "discovery_telemetry": discovery_telemetry(payload, result),
                "concept_discovery": concept_discovery_report(result),
                "dominant_label": diagnostics.mode_collapse.dominant_label,
                "dominant_label_fraction": diagnostics.mode_collapse.dominant_fraction,
                "distinct_labels": diagnostics.mode_collapse.distinct_labels,
                "scene_echo_label_count": diagnostics.scene_echo_label_count,
                # O que os estágios de contexto produziram e o que custaram.
                # Sem estes campos, ligar ou desligar qualquer um deles não
                # mudaria nada de comparável entre dois manifests.
                "contextual": asdict(diagnostics.contextual),
                "prior": asdict(diagnostics.prior),
                "signal_failure_count": len(result.signal_failures),
                "relation_failure_count": len(result.relation_failures),
                "refinement": [
                    {
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
                        # `RegionInterpretationFailure` é uma exception (#165), não um
                        # dataclass — `asdict()` levanta TypeError nela. Isso derrubava
                        # o run inteiro sempre que a etapa de refinement isolava uma
                        # falha por região (ex: OOM real do backend durante o retry),
                        # o oposto do propósito da isolação por região.
                        "failures": [
                            {"region_id": failure.region_id, "reason": failure.reason}
                            for failure in step.failures
                        ],
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
            f"**regiões canônicas:** {len(canonical_observation.all_regions)} "
            f"({len(canonical_observation.regions)} publicadas, "
            f"{len(canonical_observation.structural_context)} contexto estrutural) · "
            f"**relações:** {len(result.observation.relations)} · "
            f"**falhas de interpretação:** {len(result.region_interpretation_failures)} · "
            f"**audit:** {'✅ pass' if result.audit.passed else '❌ FAIL'} "
            f"({len(result.audit.warnings)} warnings)\n"
        )

    lifecycle.release_all()

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
        "temporal_prior_mode": config.multimodal_reasoning.temporal_prior_mode,
        "prompt_version": config.multimodal_reasoning.prompt_version,
        "sequence_masks": None if options.sequence_masks is None else str(options.sequence_masks),
        "lifecycle_metrics": [asdict(m) for m in lifecycle.metrics],
        "remote_reasoning": remote_reasoning_summary(getattr(ports.multimodal_reasoner, "calls", None)),
        "frames": frame_reports,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    memory_line = _memory_summary(lifecycle.metrics, config.gpu_memory_budget_gb)
    summary_header = (
        f"# Validação do pipeline real — {run_id}\n\n"
        f"Revisão: `{manifest['git_revision']}` · Frames: {len(frames)} · {memory_line}\n\n"
        "Ver `manifest.json` para configuração completa e log de estágios.\n\n"
    )
    (out_dir / "summary.md").write_text(summary_header + "\n".join(summary_rows), encoding="utf-8")

    print(f"\nWrote samples to {out_dir}")
    print(memory_line)
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
        "--reasoning-backend",
        choices=("qwen_vl", "gemini_robotics_er"),
        default=None,
        help=(
            "gemini_robotics_er consulta a API do Gemini com os mesmos prompts e views do "
            "Qwen (#276); frames e crops saem da máquina. Lê GEMINI_API_KEY do ambiente ou do .env"
        ),
    )
    parser.add_argument(
        "--tiling",
        dest="tiling_variant",
        choices=tuple(TILING_VARIANTS),
        default=None,
        help="variante de tiling do benchmark da #277; discard descarta propostas truncadas pela borda do tile",
    )
    parser.add_argument(
        "--concept-discovery",
        action="store_true",
        help="liga descoberta de conceitos na cena e grounding por SAM3 PCS como fonte adicional de propostas (#277)",
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
        "--temporal-prior-mode",
        choices=tuple(mode.value for mode in TemporalPriorMode),
        default=None,
        help=(
            "encadeia o que cada frame afirmou no frame seguinte; box_overlap casa "
            "por sobreposição de caixa e bumpa prompt_version para v9, porque o "
            "prompt deixa de ser byte-idêntico ao v8"
        ),
    )
    parser.add_argument(
        "--sequence-masks",
        type=Path,
        default=SEQUENCE_MASKS_DIR / f"{_SEQUENCE_ID}.json",
        help=(
            "geometria de área da sequência (círculo útil da lente e silhueta do rig); "
            "um caminho inexistente interrompe o run — use --no-sequence-masks para rodar sem"
        ),
    )
    parser.add_argument(
        "--no-sequence-masks",
        action="store_true",
        help="roda sem geometria de área declarada para uma sequência genérica",
    )
    return parser


# Ponto de entrada do script: traduz argumentos em ValidationOptions sem
# esconder defaults ou seleção de frames em estado global.
def main(argv: list[str] | None = None) -> None:
    """Executa o CLI de validação real."""
    arguments = _argument_parser().parse_args(argv)
    load_dotenv(_REPOSITORY_ROOT / ".env")
    run_validation(
        ValidationOptions(
            frames_dir=arguments.frames_dir,
            results_dir=arguments.results_dir,
            frame_ids=tuple(arguments.frame_id),
            limit=arguments.limit,
            context_profile=arguments.context_profile,
            reasoning_checkpoint=arguments.reasoning_checkpoint,
            reasoning_backend=arguments.reasoning_backend,
            tiling_variant=arguments.tiling_variant,
            concept_discovery=arguments.concept_discovery,
            region_views=tuple(arguments.region_views) or None,
            scene_context_mode=arguments.scene_context_mode,
            temporal_prior_mode=arguments.temporal_prior_mode,
            sequence_masks=None if arguments.no_sequence_masks else arguments.sequence_masks,
        )
    )


if __name__ == "__main__":
    main()
