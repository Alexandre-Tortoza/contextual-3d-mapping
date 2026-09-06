"""Execução do pipeline canônico sobre o conjunto de referência.

Issues: #199, #200 (os experimentos que consomem as predições).

Este runner é a ponte entre o módulo e a avaliação: ele executa cada
configuração da matriz de ablation sobre as mesmas amostras, mede custo
real (latência, pico de VRAM, chamadas de modelo) e grava as predições no
contract versionado que as métricas consomem.

Duas garantias tornam a comparação legítima:

- **mesmas amostras, mesmas identidades**: toda configuração roda sobre a
  mesma lista de amostras do manifest, na mesma ordem, e a predição carrega
  o ``sample_id`` do manifest — sem isso, duas configurações não seriam
  comparáveis nem que produzissem números idênticos;
- **falha é dado**: uma amostra que falha vira uma predição explicitamente
  falha, e não uma amostra ausente. Descartá-la inflaria a qualidade da
  configuração que falhou mais.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from contextual_mapping_datasets import ReferenceManifest, SampleAnnotation, load_reference_manifest
from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.application.pipeline import PerceptionPorts, run_canonical_pipeline
from visual_perception.config import ModuleConfig
from visual_perception.domain.errors import VisualPerceptionError
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.adapters.factory import create_perception_ports
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner
from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer
from visual_perception_evaluation.prediction import PredictionSet, SamplePrediction, save_predictions

from .ablations import Ablation, ablation_matrix, selected_ablations
from .paths import REFERENCE_FRAMES, REFERENCE_MANIFEST, REPOSITORY_ROOT, RESULTS_ROOT
from .prediction_export import failed_sample_prediction, to_sample_prediction


# Conta as chamadas feitas a cada backend durante uma execução. Existe
# porque "custo de modelo" só é comparável entre configurações se for
# medido, e não estimado a partir do número de regiões.
@dataclass
class PortCallCounter:
    """Contador de chamadas por backend de uma execução."""

    region_discovery: int = 0
    feature_extraction: int = 0
    language_embedding: int = 0
    multimodal_reasoning: int = 0

    # Soma todas as chamadas de modelo da execução, que é o número que o
    # relatório de custo compara entre configurações.
    @property
    def total(self) -> int:
        """Total de chamadas de backend feitas na execução."""
        return (
            self.region_discovery
            + self.feature_extraction
            + self.language_embedding
            + self.multimodal_reasoning
        )

    # Zera o contador entre amostras, para que cada predição carregue o
    # próprio custo.
    def reset(self) -> None:
        """Zera todas as contagens."""
        self.region_discovery = 0
        self.feature_extraction = 0
        self.language_embedding = 0
        self.multimodal_reasoning = 0


# Envolve os ports contando cada chamada, sem alterar o comportamento.
# Existe para medir custo sem instrumentar o pipeline, que não deve saber
# que está sendo medido.
def counting_ports(ports: PerceptionPorts, counter: PortCallCounter) -> PerceptionPorts:
    """Retorna ``ports`` com cada backend envolvido por um contador de chamadas."""

    class _Discoverer:
        def discover(self, image: Any, config: Any) -> Any:
            counter.region_discovery += 1
            return ports.region_discoverer.discover(image, config)

    class _Extractor:
        def extract(self, image: Any, config: Any) -> Any:
            counter.feature_extraction += 1
            return ports.feature_extractor.extract(image, config)

    class _Encoder:
        def encode_image(self, image: Any, config: Any) -> Any:
            counter.language_embedding += 1
            return ports.language_encoder.encode_image(image, config)

        def encode_text(self, text: str, config: Any) -> Any:
            counter.language_embedding += 1
            return ports.language_encoder.encode_text(text, config)

    class _Reasoner:
        def analyze_scene(self, image: Any, config: Any) -> Any:
            counter.multimodal_reasoning += 1
            return ports.multimodal_reasoner.analyze_scene(image, config)

        def analyze_region(self, image: Any, crop: Any, summary: Any, config: Any) -> Any:
            counter.multimodal_reasoning += 1
            return ports.multimodal_reasoner.analyze_region(image, crop, summary, config)

    return PerceptionPorts(_Discoverer(), _Extractor(), _Encoder(), _Reasoner())


# Constrói os ports de uma configuração, reais ou fake. Existe para que o
# runner sirva tanto à execução na GPU de referência quanto a um ensaio
# determinístico sem GPU.
def build_ports(config: ModuleConfig, *, real_backends: bool) -> tuple[PerceptionPorts, Any]:
    """Retorna os ports da execução e o lifecycle manager, quando houver."""
    if not real_backends:
        return (
            PerceptionPorts(
                FakeRegionDiscoverer(),
                FakeDenseFeatureExtractor(),
                FakeLanguageAlignedEncoder(),
                FakeMultimodalReasoner(),
            ),
            None,
        )
    lifecycle = ModelLifecycleManager()
    return create_perception_ports(config, lifecycle), lifecycle


# Roda uma configuração sobre todas as amostras do manifest e devolve o
# conjunto de predições. Ponto de entrada reutilizável do runner.
def run_configuration(
    manifest: ReferenceManifest,
    ablation: Ablation,
    *,
    frames_dir: Path = REFERENCE_FRAMES,
    real_backends: bool = False,
    run_id: str | None = None,
    limit: int | None = None,
    sample_ids: tuple[str, ...] = (),
) -> PredictionSet:
    """Executa ``ablation`` sobre as amostras do manifest e devolve as predições.

    Argumentos:
        manifest: o conjunto de referência anotado.
        ablation: a configuração a executar.
        frames_dir: onde os frames das amostras estão gravados.
        real_backends: usa os backends reais de GPU em vez dos fakes.
        run_id: identidade da execução; o default é o instante UTC.
        limit: processa no máximo este número de amostras (ensaios rápidos).
        sample_ids: IDs explícitos na ordem desejada; vazio usa todo o manifest.
    Retorna:
        o :class:`PredictionSet` da configuração.
    """
    ports, _ = build_ports(ablation.config, real_backends=real_backends)
    counter = PortCallCounter()
    instrumented = counting_ports(ports, counter)

    samples = select_samples(manifest, sample_ids=sample_ids, limit=limit)
    predictions = [
        _run_sample(sample, ablation, instrumented, counter, frames_dir=frames_dir) for sample in samples
    ]
    return PredictionSet(
        run_id=run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        configuration=ablation.name,
        config_fingerprint=ablation.config.fingerprint(),
        code_revision=git_revision(),
        samples=tuple(predictions),
        hardware=describe_hardware(real_backends=real_backends),
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# Seleciona amostras antes de carregar frames/modelos, preservando a ordem
# explícita e recusando execução parcial com IDs desconhecidos ou repetidos.
def select_samples(
    manifest: ReferenceManifest,
    *,
    sample_ids: tuple[str, ...] = (),
    limit: int | None = None,
) -> tuple[SampleAnnotation, ...]:
    """Retorna amostras selecionadas deterministicamente pelo chamador."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must not contain duplicates.")
    indexed = {sample.sample_id: sample for sample in manifest.samples}
    unknown = sorted(set(sample_ids) - set(indexed))
    if unknown:
        raise ValueError(f"unknown sample ids {unknown}.")
    selected = tuple(indexed[sample_id] for sample_id in sample_ids) if sample_ids else manifest.samples
    return selected[:limit] if limit is not None else selected


# Executa uma amostra, medindo custo e isolando a falha. Helper de
# run_configuration, separado para que o tratamento de falha fique visível.
def _run_sample(
    sample: SampleAnnotation,
    ablation: Ablation,
    ports: PerceptionPorts,
    counter: PortCallCounter,
    *,
    frames_dir: Path,
) -> SamplePrediction:
    """Executa o pipeline em uma amostra, devolvendo predição ou falha explícita."""
    counter.reset()
    frame_path = frames_dir / f"{sample.sample_id}.png"
    if not frame_path.exists():
        return failed_sample_prediction(
            sample.sample_id, sample.width, sample.height, f"frame not available at {frame_path}"
        )

    pixels = _load_frame(frame_path)
    payload = ImagePayload(pixels, width=int(pixels.shape[1]), height=int(pixels.shape[0]))
    observation_input = ImageObservation(
        width=payload.width,
        height=payload.height,
        encoding="rgb8",
        image=_artifact_reference(sample),
        source=_observation_reference(sample),
    )

    _reset_vram()
    started = time.monotonic()
    try:
        result = run_canonical_pipeline(observation_input, payload, ablation.config, ports)
    except (VisualPerceptionError, ValueError) as error:
        return failed_sample_prediction(
            sample.sample_id,
            sample.width,
            sample.height,
            f"{type(error).__name__}: {error}",
            latency_s=time.monotonic() - started,
        )
    return to_sample_prediction(
        result.observation,
        sample.sample_id,
        latency_s=time.monotonic() - started,
        peak_vram_bytes=_peak_vram(),
        model_calls=counter.total,
    )


# Carrega o frame de uma amostra como array RGB.
def _load_frame(path: Path) -> np.ndarray:
    """Lê um frame PNG como array RGB uint8."""
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


# Constrói a referência de artifact da amostra no formato de contract do
# repositório, preservando o digest do manifest.
def _artifact_reference(sample: SampleAnnotation) -> Any:
    """Constrói a ``SourceArtifactReference`` da amostra."""
    from contextual_mapping_contracts import SourceArtifactReference

    return SourceArtifactReference(
        uri=f"datasets/raw/corridor-02/{sample.artifact.uri}",
        media_type=sample.artifact.media_type,
        digest=sample.artifact.digest,
    )


# Constrói a referência de observação da amostra, preservando a identidade
# do conjunto de referência dentro da saída canônica.
def _observation_reference(sample: SampleAnnotation) -> Any:
    """Constrói a ``ObservationReference`` da amostra."""
    from contextual_mapping_contracts import FrameId, ObservationReference, Timestamp

    return ObservationReference(
        observation_id=sample.sample_id,
        dataset_id="corridor-02",
        sequence_id="visual-reference",
        sensor_id="camera_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=1, clock_id="rosbag"),
        frame_id=FrameId("camera_1_optical_frame"),
    )


# Zera o contador de pico de VRAM entre amostras, para que cada predição
# reporte o próprio pico e não o acumulado da execução.
def _reset_vram() -> None:
    """Zera o pico de VRAM acumulado, quando há CUDA disponível."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# Lê o pico de VRAM da amostra, ou None quando não há GPU.
def _peak_vram() -> int | None:
    """Retorna o pico de VRAM alocado, ou ``None`` sem CUDA."""
    try:
        import torch
    except ImportError:
        return None
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


# Descreve o hardware da execução, para que o custo medido seja
# interpretável fora da máquina que o produziu.
def describe_hardware(*, real_backends: bool) -> str:
    """Descreve a GPU usada, ou o caminho determinístico sem GPU."""
    if not real_backends:
        return "deterministic fakes (no GPU)"
    try:
        import torch
    except ImportError:
        return "unknown (torch unavailable)"
    if not torch.cuda.is_available():
        return "CPU (no CUDA device available)"
    properties = torch.cuda.get_device_properties(0)
    return f"{properties.name} ({properties.total_memory / 1e9:.1f} GB)"


# Retorna a revisão de código da execução, sem a qual o resultado não é
# reproduzível.
def git_revision() -> str:
    """Retorna o hash curto do commit atual, ou ``unknown`` fora de um worktree."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPOSITORY_ROOT)
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Interface de linha de comando do runner.
def main() -> None:
    """Executa a matriz de ablation sobre o conjunto de referência."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--frames-dir", type=Path, default=REFERENCE_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--real-backends", action="store_true")
    parser.add_argument("--calibration-artifact", type=Path, default=None)
    parser.add_argument("--calibration-domain", type=str, default="indoor_corridor")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    arguments = parser.parse_args()

    manifest = load_reference_manifest(arguments.manifest)
    matrix = ablation_matrix(
        real_backends=arguments.real_backends,
        calibration_artifact=arguments.calibration_artifact,
        calibration_domain=arguments.calibration_domain,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = arguments.output_dir or RESULTS_ROOT / "predictions" / run_id
    destination.mkdir(parents=True, exist_ok=True)

    for ablation in selected_ablations(matrix, tuple(arguments.only)):
        print(f"=== {ablation.name} — {ablation.mechanism}")
        predictions = run_configuration(
            manifest,
            ablation,
            frames_dir=arguments.frames_dir,
            real_backends=arguments.real_backends,
            run_id=run_id,
            limit=arguments.limit,
            sample_ids=tuple(arguments.sample_id),
        )
        path = destination / f"{ablation.name}.json"
        save_predictions(predictions, path)
        failures = sum(1 for sample in predictions.samples if sample.failed)
        print(f"    {len(predictions.samples)} samples, {failures} failed -> {path}")


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
