"""Valida o pipeline canônico real de ponta a ponta na GPU de referência (#190).

Roda ``run_canonical_pipeline`` com os 4 backends reais benchmark-selecionados
(#174) sobre o conjunto representativo de frames do corridor-02 (ver
``prepare_corridor02_frames.py``), verificando ausência de OOM e VRAM dentro
do budget de referência, e gera amostras (JSON + overlay + resumo) para
revisão humana.

Uso (a partir de ``modules/visual-perception``, com os extras ``ml``
instalados e os frames já extraídos):

    python benchmarks/validate_reference_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
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
from visual_perception.application.pipeline import run_canonical_pipeline  # noqa: E402
from visual_perception.domain.errors import VisualPerceptionError  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
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


# Preenche a faixa do chassi/rodas do robô com preto (mesma cor do vinheta
# do fisheye) para que SAM/VLM não a tratem como conteúdo de cena — sem
# isso, o carrinho aparece rotulado em todo frame, sempre a mesma
# distração não relacionada ao ambiente sendo mapeado.
def _mask_ego_vehicle(pixels: np.ndarray) -> np.ndarray:
    masked = pixels.copy()
    masked[_EGO_VEHICLE_ROW_START:, :, :] = 0
    return masked


# Retorna o hash curto do commit atual, ou "unknown" fora de um git worktree;
# usado no manifest para amarrar as amostras à revisão de código que as gerou.
def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_MODULE_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    frames = sorted(FRAMES_DIR.glob("*.png"))
    if not frames:
        raise SystemExit(f"No frames found in {FRAMES_DIR}. Run prepare_corridor02_frames.py first.")

    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_DIR / "samples" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[str] = []
    frame_reports: list[dict[str, object]] = []

    for frame_path in frames:
        name = frame_path.stem
        print(f"\n=== {name} ===")
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
            frame_reports.append({"frame": name, "failed": True, "reason": repr(error)})
            summary_rows.append(f"## {name}\n\n**FALHOU:** `{error!r}`\n")
            continue

        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(serialize_observation(result.observation), indent=2))

        overlay_path = out_dir / f"{name}.overlay.png"
        render_overlay(image, result.observation).save(overlay_path)

        scene_type = next(
            (c.value for c in result.observation.scene_context.claims if c.kind.value == "scene_type"),
            "?",
        )
        audit_status = "pass" if result.audit.passed else "FAIL"
        print(
            f"  regions={len(result.observation.regions)} "
            f"relations={len(result.observation.relations)} "
            f"interpretation_failures={len(result.region_interpretation_failures)} "
            f"audit={audit_status} warnings={len(result.audit.warnings)}"
        )
        frame_reports.append(
            {
                "frame": name,
                "failed": False,
                "region_count": len(result.observation.regions),
                "relation_count": len(result.observation.relations),
                "interpretation_failure_count": len(result.region_interpretation_failures),
                "audit_passed": result.audit.passed,
                "audit_error_count": len(result.audit.errors),
                "audit_warning_count": len(result.audit.warnings),
            }
        )
        summary_rows.append(
            f"## {name}\n\n"
            f"![{name}]({overlay_path.name})\n\n"
            f"**scene_type:** {scene_type} · **regiões:** {len(result.observation.regions)} · "
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
        "frame_count": len(frames),
        "frames_dir": str(FRAMES_DIR),
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


if __name__ == "__main__":
    main()
