"""Ablação espacial em trecho curto com reconhecimento previamente congelado.

Reutiliza o mapa, calibração e artifact visual existentes. Executa grounding
uma vez por frame e recompõe quatro braços: discovery, grounding, boundary e
pose. A janela tem no máximo dez segundos; não executa tracking nem altera claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from mapping_runtime.corridor02_context import (
    Corridor02ContextRequest,
    Corridor02Keyframe,
    export_corridor02_context,
)
from PIL import Image
from sensor_association import BoundaryPolicy, distance_to_mask_boundary

from visual_perception import (
    DebugRecorder,
    ImagePayload,
    Mask,
    ModuleConfig,
    SemanticGroundingConfig,
    create_perception_ports,
    deserialize_observation,
    ground_regions,
    serialize_observation,
)
from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.application.quality_audit import audit_observation
from visual_perception.domain.image_area import ImageAreaMasks


# Centraliza a gravação legível dos resultados e impede métricas numpy de
# escaparem para JSON como representações de objetos em strings.
def _write_json(path: Path, payload: Any) -> None:
    """Grava JSON UTF-8 no diretório de resultados declarado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# Usa figuras científicas com coordenadas de imagem explícitas. Os PNGs de
# máscaras não alteram o input e nunca são consumidos pelo algoritmo.
def _mask_artifacts(directory: Path, raw: np.ndarray, regions) -> None:
    """Persiste raw, discovery, saída SAM, máscara aceita, boundary e overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for region in regions:
        grounding = region.grounding
        if grounding is None:
            continue
        root = directory / region.region_id
        root.mkdir(parents=True, exist_ok=True)
        discovery = region.mask.data
        model = np.zeros_like(discovery) if grounding.prediction.model_mask is None else grounding.prediction.model_mask.data
        semantic = np.zeros_like(discovery) if grounding.semantic_mask is None else grounding.semantic_mask.data
        boundary = semantic & (distance_to_mask_boundary(semantic) <= BoundaryPolicy().margin_px)
        for name, data in (("discovery", discovery), ("model", model), ("semantic", semantic), ("boundary", boundary)):
            Image.fromarray(data.astype(np.uint8) * 255).save(root / f"{name}.png")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, title, mask in zip(axes.flat,
                ("RGB original", "Discovery", "Segmentação condicionada", "Semantic mask", "Overlay semântico", "Boundary / interior"),
                (None, discovery, model, semantic, semantic, boundary), strict=True):
            if title in ("Discovery", "Segmentação condicionada", "Semantic mask"):
                ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(raw)
                if mask is not None and mask.any():
                    overlay = np.zeros((*mask.shape, 4))
                    overlay[mask] = (0.1, 0.8, 1.0, 0.55)
                    ax.imshow(overlay)
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle(f"{grounding.prediction.concept} · {region.region_id}\n{grounding.status.value} · {discovery.sum()} → {semantic.sum()} pixels")
        fig.savefig(root / "comparison.png", dpi=130)
        plt.close(fig)


# Mede suporte sem usar confiança bruta como ground truth. Concordância só é
# definida onde ao menos duas views realmente contribuíram ao mesmo ponto.
def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Resume labels fortes, tentativos e suporte espacial por conceito."""
    labels: dict[str, dict[str, Any]] = {}
    for point in payload["points"]:
        context = point.get("context") or {}
        label = context.get("label")
        if not label:
            continue
        record = labels.setdefault(label, {"point_count": 0, "spatial_support": [], "weak_support_count": 0, "agreements": [], "uncorroborated_count": 0})
        record["point_count"] += 1
        if context.get("spatial_support") is not None:
            record["spatial_support"].append(context["spatial_support"])
        record["weak_support_count"] += context.get("support_state") == "weak"
        record["uncorroborated_count"] += context.get("support_state") == "uncorroborated"
        if context.get("agreement") is not None:
            record["agreements"].append(context["agreement"])
    for record in labels.values():
        support = record.pop("spatial_support")
        record["spatial_support_mean"] = float(np.mean(support)) if support else None
        agreements = record.pop("agreements")
        record["multi_view_agreement"] = float(np.mean(agreements)) if agreements else None
        record["multi_view_point_count"] = len(agreements)
    associations = Counter()
    by_label: dict[str, Counter] = {}
    for observation in payload["observations"]:
        associations.update(observation["semantic_association_counts"])
        for label, counts in observation["semantic_label_association_counts"].items():
            by_label.setdefault(label, Counter()).update(counts)
    return {
        "labels": labels,
        "tentative_label_counts": dict(Counter(
            (point.get("context") or {})["tentative_label"] for point in payload["points"]
            if (point.get("context") or {}).get("tentative_label")
        )),
        "semantic_association_counts": dict(associations),
        "semantic_label_association_counts": {label: dict(counts) for label, counts in by_label.items()},
        "safe_interior_association_count": associations["interior"],
        "rejected_strong_boundary_association_count": associations["boundary"],
        "regions": [{**region["spatial_diagnostics"], "region_id": region["region_id"],
                     "observation_frame_id": region["observation_frame_id"], "label": region["label"]}
                    for region in payload["regions"]],
        "context_summary": payload["context_summary"],
        "observations": payload["observations"],
    }


# Mostra pontos efetivamente rotulados no artifact, sem filtros de suporte do
# viewer. As quatro colunas usam a mesma câmera e escala 3D para comparação.
def _projection_artifacts(directory: Path, raw: np.ndarray, arms: dict[str, dict[str, Any]], frame_id: str) -> None:
    """Produz comparação RGB→3D por conceito usando todas as associações fortes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    concepts = sorted({region["label"] for payload in arms.values() for region in payload["regions"] if region["observation_frame_id"] == frame_id})
    for index, concept in enumerate(concepts):
        selected = {
            name: [(point, contribution) for point in payload["points"]
                for contribution in ((point.get("context") or {}).get("contributions") or [point.get("context") or {}])
                if contribution.get("label") == concept and str(contribution.get("region_id", "")).startswith(frame_id + ":")]
            for name, payload in arms.items()
        }
        coordinates = [point["coordinates_m"] for labelled in selected.values() for point, _ in labelled]
        background = np.asarray([point["coordinates_m"] for point in next(iter(arms.values()))["points"]])
        extent = np.asarray(coordinates) if coordinates else background
        low, high = extent.min(axis=0), extent.max(axis=0)
        padding = np.maximum((high - low) * 0.1, 0.25)
        low, high = low - padding, high + padding
        background = background[np.all((background >= low) & (background <= high), axis=1)]
        fig = plt.figure(figsize=(20, 9), layout="constrained")
        for column, name in enumerate(arms):
            ax = fig.add_subplot(2, len(arms), column + 1)
            ax.imshow(raw)
            # Uma contribuição não vencedora ainda precisa aparecer em sua
            # própria imagem. Nunca desenha o pixel de outra câmera/frame.
            labelled = selected[name]
            if labelled:
                pixels = np.asarray([contribution["pixel"] for _, contribution in labelled])
                ax.scatter(pixels[:, 0], pixels[:, 1], s=3, color="#00b7ee")
            ax.set_title(f"{name}\n{len(labelled)} pontos")
            ax.set_axis_off()
            space = fig.add_subplot(2, len(arms), len(arms) + column + 1, projection="3d")
            if len(background):
                space.scatter(*background.T, s=0.1, c="gray", alpha=0.15)
            space.set_xlim(low[0], high[0])
            space.set_ylim(low[1], high[1])
            space.set_zlim(low[2], high[2])
            space.set_box_aspect((1, 1, 0.8))
            if labelled:
                xyz = np.asarray([point["coordinates_m"] for point, _ in labelled])
                space.scatter(*xyz.T, s=1.5, c="#0099cc")
            space.view_init(elev=18, azim=-70)
            space.set_xlabel("x (m)")
            space.set_ylabel("y (m)")
            space.set_zlabel("z (m)")
        fig.suptitle(concept)
        fig.savefig(directory / f"projection-{index:02d}.png", dpi=140)
        plt.close(fig)
    _write_json(directory / "projection-index.json", {f"projection-{index:02d}.png": concept for index, concept in enumerate(concepts)})


# Reexecuta apenas o estágio espacial de um artifact congelado. Cada frame
# persiste seu resultado antes de iniciar o próximo, preservando falhas parciais.
def _ground_frame(source: Path, destination: Path, config: ModuleConfig) -> dict[str, Any]:
    """Refina um frame e mede custo, claims e auditoria antes/depois."""
    raw = np.asarray(Image.open(source / "raw.png").convert("RGB"))
    valid = np.asarray(Image.open(source / "valid-area-mask.png").convert("L")) > 0
    ego_path = source / "ego-mask.png"
    ego = np.asarray(Image.open(ego_path).convert("L")) > 0 if ego_path.is_file() else np.zeros_like(valid)
    observation = deserialize_observation(json.loads((source / "observation.json").read_text()))
    baseline_audit = audit_observation(observation)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)
    started = time.perf_counter()
    try:
        grounded = ground_regions(observation.regions, ImagePayload(raw, raw.shape[1], raw.shape[0]),
            ports.semantic_grounder, config.semantic_grounding,
            ImageAreaMasks(Mask(valid, raw.shape[1], raw.shape[0]), Mask(ego, raw.shape[1], raw.shape[0])))
    finally:
        lifecycle.release_all()
    latency_s = time.perf_counter() - started
    after = replace(observation, regions=grounded)
    after_audit = audit_observation(after)
    destination.mkdir(parents=True)
    _write_json(destination / "observation.json", serialize_observation(after))
    for filename in ("raw.png", "valid-area-mask.png", "ego-mask.png", "regions-overlay.png"):
        if (source / filename).is_file():
            shutil.copyfile(source / filename, destination / filename)
    DebugRecorder(destination / "DEBUG").record_grounding(source.name, [region.grounding.diagnostics for region in grounded if region.grounding])
    _mask_artifacts(destination / "masks", raw, grounded)
    return {
        "frame_id": source.name,
        "source_observation": str(source / "observation.json"),
        "source_observation_sha256": hashlib.sha256((source / "observation.json").read_bytes()).hexdigest(),
        "latency_added_s": latency_s,
        "model_calls": sum(region.grounding.prediction.model_calls for region in grounded if region.grounding),
        "peak_vram_bytes": max((item.peak_vram_bytes or 0 for item in lifecycle.metrics), default=0),
        "lifecycle": [asdict(item) for item in lifecycle.metrics],
        "audit_errors_before": len(baseline_audit.errors), "audit_errors_after": len(after_audit.errors),
        "audit_warnings_before": len(baseline_audit.warnings), "audit_warnings_after": len(after_audit.warnings),
        "structural_context_unchanged": after.structural_context == observation.structural_context,
        "claims_unchanged": all(a.claims == b.claims for a, b in zip(observation.regions, grounded, strict=True)),
        "grounding_status_counts": dict(Counter(region.grounding.status.value for region in grounded if region.grounding)),
    }


# Mantém uma única janela e exatamente as mesmas observações nos quatro braços.
# A seleção é temporal; não há seleção de regiões ou parâmetros por label.
def run(options: argparse.Namespace) -> Path:
    """Reexecuta grounding em até dez segundos e compara fatores separados."""
    window = json.loads(options.window.read_text())
    entries = window["keyframes"]
    if options.frame_dir is not None:
        index = int(options.frame_dir.name.rsplit("-", 1)[-1])
        entries = [entry for entry in entries if entry["sequence_index"] == index]
    if not entries:
        raise ValueError("Validation requires at least one frame present in the window.")
    duration = (entries[-1]["header_timestamp_ns"] - entries[0]["header_timestamp_ns"]) / 1e9
    if duration > 10.1:
        raise ValueError("Grounding validation is limited to ten seconds.")
    sources = [(entry, options.frame_dir.resolve() if options.frame_dir is not None else
        options.visual_run.resolve() / "frames" / f"{window.get('frame_id_prefix', 'corridor-02')}-{entry['sequence_index']:05d}") for entry in entries]
    for _, source in sources:
        for filename in ("raw.png", "observation.json", "valid-area-mask.png", "regions-overlay.png"):
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
    destination = options.output.resolve()
    if destination.exists():
        raise ValueError("Choose a new validation output directory; results must not be overwritten.")
    destination.mkdir(parents=True)
    config = ModuleConfig(semantic_grounding=SemanticGroundingConfig(backend="grounded_sam"))
    manifest: dict[str, Any] = {
        "frame_count": len(entries), "duration_s": duration, "recognition_replayed": True,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "grounding_config": asdict(config.semantic_grounding),
        "window": window, "frames": [], "completed": False,
        "inputs": {key: str(getattr(options, key).resolve()) for key in ("geometry", "bag", "intrinsics", "extrinsics", "odometry")},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()},
    }
    from importlib.metadata import version

    from huggingface_hub import try_to_load_from_cache
    manifest["dependencies"] = {name: version(name) for name in ("torch", "transformers", "scipy", "numpy")}
    manifest["checkpoint_configs"] = {checkpoint: str(try_to_load_from_cache(checkpoint, "config.json"))
        for checkpoint in (config.semantic_grounding.detector_checkpoint, config.semantic_grounding.segmenter_checkpoint)}
    keyframes = []
    for entry, source in sources:
        after_dir = destination / "frames" / source.name
        record = _ground_frame(source, after_dir, config)
        manifest["frames"].append(record)
        _write_json(destination / "manifest.json", manifest)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        keyframes.append(Corridor02Keyframe(source.name, entry["sequence_index"], entry["header_timestamp_ns"], entry["bag_timestamp_ns"],
            after_dir / "observation.json", after_dir / "raw.png", after_dir / "regions-overlay.png", after_dir / "valid-area-mask.png",
            after_dir / "ego-mask.png" if (after_dir / "ego-mask.png").is_file() else None))
    arms, metrics = {}, {}
    for name, legacy, boundary, pose in (
        ("before", True, False, "nearest"),
        ("grounding", False, False, "nearest"),
        ("boundary", False, True, "nearest"),
        ("pose", False, True, "interpolated"),
    ):
        print(f"Compondo {name}: {len(keyframes)} frames, {duration:.3f} segundos", flush=True)
        output = destination / f"{name}.json"
        policy = BoundaryPolicy(enabled=boundary, allow_legacy_discovery=legacy)
        request = Corridor02ContextRequest(options.geometry, options.bag, options.intrinsics, options.extrinsics,
            tuple(keyframes), output, odometry=options.odometry, boundary_policy=policy,
            footprint_mode="legacy_discovery" if legacy else "semantic_grounding", pose_sampling=pose,
            visibility_mode="legacy_cells",
            stuff_discovery_fallback=options.stuff_discovery_fallback)
        export_corridor02_context(request)
        payload = json.loads(output.read_text())
        arms[name], metrics[name] = payload, _metrics(payload)
        _write_json(destination / "metrics.json", metrics)
        subprocess.run(["node", "apps/map-explorer/scripts/dump_palette_debug.mjs", str(output), str(destination / f"{name}-DEBUG" / "viewer.json")], check=True, capture_output=True)
    review_frames = options.review_frame or [sources[0][1].name]
    for frame_id in review_frames:
        raw = np.asarray(Image.open(destination / "frames" / frame_id / "raw.png").convert("RGB"))
        _projection_artifacts(destination / "projections" / frame_id, raw, arms, frame_id)
    manifest["completed"] = True
    _write_json(destination / "manifest.json", manifest)
    return destination


# Expõe entradas rastreáveis sem escolher dataset, labels ou trecho no algoritmo.
def main() -> None:
    """Lê paths explícitos e executa a validação de um trecho curto."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frame-dir", type=Path)
    source.add_argument("--visual-run", type=Path)
    for name in ("window", "geometry", "bag", "intrinsics", "extrinsics", "odometry", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stuff-discovery-fallback", action="store_true")
    parser.add_argument("--review-frame", action="append", help="frame com comparação de projeções, repetível")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
