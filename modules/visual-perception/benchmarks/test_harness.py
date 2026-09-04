"""Testes do harness de ablation (#175). Roda o pipeline canônico só contra fakes."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

from fixtures import default_config, image_observation, payload_with_blobs  # noqa: E402
from fixtures_ports import default_ports  # noqa: E402
from harness import (  # noqa: E402
    AblationConfig,
    DatasetReference,
    GroundTruthRegion,
    run_ablation,
    score_observation,
)
from visual_perception.application.pipeline import run_canonical_pipeline  # noqa: E402


# Verifica que score_observation dá cobertura total (1.0) e zero
# hallucination quando a região de ground-truth é exatamente a máscara que o
# pipeline canônico produziu para a mesma imagem.
def test_score_observation_rewards_matching_regions() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 10, 10, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    ground_truth = (GroundTruthRegion(mask=result.observation.regions[0].mask),)
    metrics = score_observation(result.observation, ground_truth)

    assert metrics.region_coverage == 1.0
    assert metrics.hallucination_rate == 0.0


# Verifica que score_observation reporta hallucination_rate == 1.0 quando
# não há nenhuma região de ground-truth, mas o pipeline produziu regiões
# (todas as predições são "alucinadas").
def test_score_observation_penalizes_hallucinated_regions() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 10, 10, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    metrics = score_observation(result.observation, ground_truth=())

    assert metrics.hallucination_rate == 1.0


# Verifica que run_ablation devolve um BenchmarkReport que referencia
# exatamente o dataset e a ablation passados, com uma chamada de modelo por
# amostra e uma hallucination_rate dentro do intervalo válido — garantindo
# que o report é reproduzível a partir de seus próprios campos.
def test_run_ablation_reports_reproducible_configuration() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 10, 10, (200, 30, 30)),))
    image = image_observation()
    dataset = DatasetReference(name="synthetic-smoke", version="v1", annotation_uri="memory://smoke")
    ablation = AblationConfig(name="baseline", module_config=default_config())

    report = run_ablation(dataset, ablation, ((image, payload, ()),), default_ports())

    assert report.dataset == dataset
    assert report.ablation == ablation
    assert report.model_calls == 1
    assert 0.0 <= report.metrics.hallucination_rate <= 1.0
