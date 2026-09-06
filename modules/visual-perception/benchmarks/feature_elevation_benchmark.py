"""Compara baseline, resolução nativa maior e FeatUp/JBU nos mesmos frames (#208)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parents[1]
for _relative in ("src", "../../contracts", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / _relative).resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_upsampling_benchmark import (  # noqa: E402
    BASELINE,
    HIGH_RESOLUTION,
    PIXEL_BILINEAR,
    PathResult,
    _combined,
    _describe_hardware,
    _git_revision,
    measure_path,
    prepare_frames,
)
from validate_reference_pipeline import select_frame_paths  # noqa: E402
from visual_perception.application.execution_profile import research_quality_config  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.config import FeatureExtractionConfig  # noqa: E402
from visual_perception.domain.feature_map import FeatureMap, feature_map_spec_to_dict  # noqa: E402
from visual_perception.infrastructure.adapters.factory import create_perception_ports  # noqa: E402
from visual_perception.infrastructure.adapters.feature_extraction_backend import (  # noqa: E402
    RealDenseFeatureExtractionAdapter,
)
from visual_perception.infrastructure.adapters.feature_upsampling_backend import (  # noqa: E402
    FeatUpDenseFeatureExtractionAdapter,
)

RESULTS_DIR = _MODULE_ROOT / "benchmarks" / "results"
REFERENCE_FRAMES = _MODULE_ROOT / "benchmarks" / ".local" / "corridor-02-frames"
REFERENCE_FRAME_IDS = ("corridor-02-000", "corridor-02-008", "corridor-02-017")


# Extrai um mapa candidato para cada frame já segmentado, preservando as
# mesmas instâncias/IDs de região e medindo custo apenas da extração.
def extract_candidate_maps(
    frames: tuple[tuple[str, Any, FeatureMap, tuple[Any, ...]], ...],
    extractor: Any,
    config: FeatureExtractionConfig,
) -> tuple[tuple[tuple[str, Any, FeatureMap, tuple[Any, ...]], ...], float, int | None]:
    """Substitui somente os mapas de feature e retorna tempo e pico de VRAM.

    Argumentos:
        frames: frames, payloads e regiões fixados pelo baseline.
        extractor: implementação do port ``DenseFeatureExtractor``.
        config: configuração exclusiva do candidato.
    Retorna:
        frames com os novos mapas, latência total e pico de VRAM.
    """
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prepared = tuple(
        (frame_id, payload, extractor.extract(payload, config), regions)
        for frame_id, payload, _feature_map, regions in frames
    )
    elapsed = time.monotonic() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated())
        if torch is not None and torch.cuda.is_available()
        else None
    )
    return prepared, elapsed, peak_vram


# Remove dados regionais volumosos do PathResult e anexa a proveniência do
# mapa efetivamente produzido, formando uma linha auditável do relatório.
def serialize_candidate(
    result: PathResult,
    frames: tuple[tuple[str, Any, FeatureMap, tuple[Any, ...]], ...],
    *,
    extraction_latency_s: float,
    extraction_peak_vram_bytes: int | None,
    comparable_embedding_space: bool,
) -> dict[str, Any]:
    """Converte um candidato medido em um documento JSON compacto."""
    document = asdict(result)
    document["representability"] = {
        "regions": result.regions,
        "representable_regions": result.representable_regions,
        "strata": result.strata,
    }
    document["evidence_quality"] = {
        **_combined(list(result.measurements)),
        "inter_region_separation": result.inter_region_separation,
    }
    document["cost"] = {
        "pool_latency_s": result.pool_latency_s,
        "materialized_bytes": result.materialized_bytes,
        "materialize_latency_s": result.materialize_latency_s,
        "materialize_refused": result.materialize_refused,
        "extraction_latency_s": extraction_latency_s,
        "extraction_peak_vram_bytes": extraction_peak_vram_bytes,
    }
    document.pop("measurements")
    document.pop("vectors")
    document["extraction_latency_s"] = extraction_latency_s
    document["extraction_peak_vram_bytes"] = extraction_peak_vram_bytes
    document["feature_specs"] = [
        {"frame_id": frame_id, **feature_map_spec_to_dict(feature_map)}
        for frame_id, _payload, feature_map, _regions in frames
    ]
    document["comparable_embedding_space"] = comparable_embedding_space
    return document


# Renderiza as métricas centrais em Markdown sem comparar diretamente
# vetores DINOv2-base e DINOv2-small, que vivem em espaços diferentes.
def render_summary(document: dict[str, Any]) -> str:
    """Retorna o resumo humano do benchmark #208."""
    lines = [
        f"# Benchmark de elevação de features — {document['run_id']}",
        "",
        f"Frames: `{', '.join(document['ordered_frame_ids'])}`",
        f"Hardware: {document['hardware']}",
        "",
        "| caminho | geração | grade efetiva | representáveis | extração (s) | pico VRAM |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in document["candidates"]:
        spec = candidate["feature_specs"][0]
        peak = candidate["extraction_peak_vram_bytes"]
        lines.append(
            f"| {candidate['name']} | {spec['generation']} | "
            f"{spec['grid_width']}x{spec['grid_height']} | "
            f"{candidate['representable_regions']}/{candidate['regions']} | "
            f"{candidate['extraction_latency_s']:.3f} | "
            f"{'n/a' if peak is None else f'{peak / 2**30:.2f} GiB'} |"
        )
    lines.extend(
        [
            "",
            "## Qualidade da evidência",
            "",
            "| caminho | suporte médio | consistência intra | separação inter |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for candidate in document["candidates"]:
        quality = candidate["evidence_quality"]
        lines.append(
            f"| {candidate['name']} | "
            f"{_format_metric(quality.get('mean_support_ratio'))} | "
            f"{_format_metric(quality.get('mean_intra_region_consistency'))} | "
            f"{_format_metric(quality.get('inter_region_separation'))} |"
        )
    lines.extend(
        [
            "",
            "## Custo de pooling e materialização",
            "",
            "| caminho | pooling (s) | mapa materializado | materialização (s) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for candidate in document["candidates"]:
        cost = candidate["cost"]
        materialized = cost["materialized_bytes"]
        lines.append(
            f"| {candidate['name']} | {cost['pool_latency_s']:.3f} | "
            f"{'n/a' if materialized is None else f'{materialized / 2**20:.1f} MiB'} | "
            f"{_format_metric(cost['materialize_latency_s'])} |"
        )
    lines.extend(
        [
            "",
            "A concordância vetorial só é calculada dentro do espaço DINOv2-base. ",
            "FeatUp usa DINOv2-small (384 dimensões), portanto sua qualidade é comparada por ",
            "representabilidade, cobertura, consistência, separação e custo — nunca por produto ",
            "escalar direto contra o backbone de 768 dimensões.",
            "",
        ]
    )
    return "\n".join(lines)


# Formata métricas opcionais mantendo ausência distinta de zero medido.
def _format_metric(value: float | None) -> str:
    """Representa uma métrica numérica ou a marca explícita de ausência."""
    return "n/a" if value is None else f"{value:.4f}"


# Executa todos os candidatos sobre geometria fixa e grava JSON/Markdown.
def run_benchmark(frame_paths: tuple[Path, ...], results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Compara os cinco caminhos da #208 nos mesmos frames e regiões."""
    base_config = research_quality_config(multi_scale_justified=False, real_backends=True)
    baseline_config = dataclasses.replace(
        base_config,
        feature_extraction=dataclasses.replace(
            base_config.feature_extraction,
            input_resolution=None,
            upsampling="patch_grid",
        ),
    )
    lifecycle = ModelLifecycleManager()
    baseline_ports = create_perception_ports(baseline_config, lifecycle)
    baseline_frames = prepare_frames(frame_paths, baseline_config, baseline_ports)
    baseline_frames, baseline_latency, baseline_vram = extract_candidate_maps(
        baseline_frames,
        RealDenseFeatureExtractionAdapter(ModelLifecycleManager()),
        baseline_config.feature_extraction,
    )

    candidates: list[dict[str, Any]] = []
    patch = measure_path("patch_grid", BASELINE, baseline_frames)
    candidates.append(
        serialize_candidate(
            patch,
            baseline_frames,
            extraction_latency_s=baseline_latency,
            extraction_peak_vram_bytes=baseline_vram,
            comparable_embedding_space=True,
        )
    )
    for name, method in (("nearest", HIGH_RESOLUTION), ("bilinear", PIXEL_BILINEAR)):
        result = measure_path(name, method, baseline_frames, baseline_vectors=patch.vectors)
        candidates.append(
            serialize_candidate(
                result,
                baseline_frames,
                extraction_latency_s=0.0,
                extraction_peak_vram_bytes=0,
                comparable_embedding_space=True,
            )
        )

    high_config = dataclasses.replace(
        base_config.feature_extraction,
        input_resolution=448,
        upsampling="nearest",
    )
    high_frames, high_latency, high_vram = extract_candidate_maps(
        baseline_frames,
        RealDenseFeatureExtractionAdapter(ModelLifecycleManager()),
        high_config,
    )
    high_result = dataclasses.replace(
        measure_path(
            "nearest", HIGH_RESOLUTION, high_frames, baseline_vectors=patch.vectors
        ),
        name="dinov2_native_448",
    )
    candidates.append(
        serialize_candidate(
            high_result,
            high_frames,
            extraction_latency_s=high_latency,
            extraction_peak_vram_bytes=high_vram,
            comparable_embedding_space=True,
        )
    )

    featup_config = FeatureExtractionConfig(
        backend="featup",
        checkpoint="facebookresearch/dinov2:dinov2_vits14",
        input_resolution=448,
        upsampling="nearest",
        max_feature_map_mb=384,
    )
    featup_frames, featup_latency, featup_vram = extract_candidate_maps(
        baseline_frames,
        FeatUpDenseFeatureExtractionAdapter(ModelLifecycleManager()),
        featup_config,
    )
    featup_result = dataclasses.replace(
        measure_path("nearest", HIGH_RESOLUTION, featup_frames),
        name="featup_dinov2_small_jbu",
    )
    candidates.append(
        serialize_candidate(
            featup_result,
            featup_frames,
            extraction_latency_s=featup_latency,
            extraction_peak_vram_bytes=featup_vram,
            comparable_embedding_space=False,
        )
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    document = {
        "schema_version": "feature-elevation-benchmark/1",
        "issue": 208,
        "run_id": run_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_revision": _git_revision(),
        "hardware": _describe_hardware(),
        "ordered_frame_ids": [path.stem for path in frame_paths],
        "candidates": candidates,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"benchmark-208-feature-elevation-{run_id}.json"
    markdown_path = json_path.with_suffix(".md")
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_summary(document), encoding="utf-8")
    return json_path, markdown_path


# Expõe seleção explícita dos mesmos três frames usados pelo validador #212.
def main(argv: list[str] | None = None) -> None:
    """Executa o benchmark real e imprime os caminhos dos artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=REFERENCE_FRAMES)
    parser.add_argument("--frame-id", action="append", default=[])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    arguments = parser.parse_args(argv)
    frame_ids = tuple(arguments.frame_id) or REFERENCE_FRAME_IDS
    frame_paths = select_frame_paths(arguments.frames_dir, frame_ids)
    json_path, markdown_path = run_benchmark(frame_paths, arguments.results_dir)
    print(f"Wrote {json_path}\nWrote {markdown_path}")


if __name__ == "__main__":  # pragma: no cover - entrada de CLI
    main()
