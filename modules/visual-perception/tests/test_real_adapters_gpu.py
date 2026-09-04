"""Smoke tests reais dos 4 adapters (#186-#189) na configuração benchmark-selecionada (#174).

Pulados automaticamente quando não há GPU CUDA visível (ex: CI, dev sem GPU):
estes testes carregam checkpoints reais e rodam inferência de verdade, ao
contrário de ``test_real_adapters.py`` (funções puras, sem GPU).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from fixtures import image_observation  # noqa: E402
from visual_perception.application.execution_profile import research_quality_config  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.application.pipeline import run_canonical_pipeline  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.infrastructure.adapters.factory import create_perception_ports  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requer GPU CUDA (checkpoints reais benchmark #174)."
)


# Libera o cache CUDA depois de cada teste deste arquivo: cada teste carrega
# seu próprio modelo real e, sem isso, o cache fragmentado de um teste (ex:
# Qwen2.5-VL) pode fazer o próximo (ex: SAM) estourar VRAM mesmo cabendo
# individualmente no budget de 8GB — visto na prática rodando este arquivo.
@pytest.fixture(autouse=True)
def _empty_cuda_cache_between_tests():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Uma imagem RGB pequena, mas com estrutura suficiente (gradiente + bloco
# sólido) para o SAM real encontrar ao menos uma região; usada por todos os
# testes deste arquivo para manter a inferência real rápida.
def _real_test_payload(width: int = 64, height: int = 64) -> ImagePayload:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    pixels[16:48, 16:48] = (30, 160, 30)
    return ImagePayload(pixels, width=width, height=height)


@requires_gpu
def test_real_region_discovery_adapter_finds_regions_on_gpu() -> None:
    """O adapter real de region discovery roda SAM de verdade e retorna geometria válida."""
    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)
    payload = _real_test_payload()

    proposals = ports.region_discoverer.discover(payload, config.region_discovery)
    lifecycle.release_active()

    assert len(proposals) > 0
    for proposal in proposals:
        assert proposal.mask.data.shape == (payload.height, payload.width)
        assert proposal.box == proposal.mask.bounding_box()
        assert 0.0 <= proposal.geometric_confidence <= 1.0


@requires_gpu
def test_real_feature_extraction_adapter_returns_finite_grid_on_gpu() -> None:
    """O adapter real de feature extraction roda DINOv2 de verdade e retorna um grid finito."""
    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)
    payload = _real_test_payload()

    feature_map = ports.feature_extractor.extract(payload, config.feature_extraction)
    lifecycle.release_active()

    assert feature_map.dimension > 0
    assert feature_map.grid_height > 0 and feature_map.grid_width > 0
    assert np.isfinite(feature_map.data).all()


@requires_gpu
def test_real_language_embedding_adapter_returns_normalized_vector_on_gpu() -> None:
    """O adapter real de language embedding roda CLIP de verdade e normaliza a saída."""
    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)
    payload = _real_test_payload()

    image_vector = ports.language_encoder.encode_image(payload, config.language_embedding)
    text_vector = ports.language_encoder.encode_text("a green square", config.language_embedding)
    lifecycle.release_active()

    for vector in (image_vector, text_vector):
        assert len(vector) == config.language_embedding.dimension
        assert all(np.isfinite(value) for value in vector)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)


@requires_gpu
def test_real_multimodal_reasoning_adapter_returns_scene_json_on_gpu() -> None:
    """O adapter real de multimodal reasoning roda o VLM de verdade e retorna JSON parseável."""
    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)
    payload = _real_test_payload()

    response = ports.multimodal_reasoner.analyze_scene(payload, config.multimodal_reasoning)
    lifecycle.release_active()

    assert isinstance(response, dict)
    assert "scene_type" in response
    assert "description" in response


@requires_gpu
def test_real_backends_pipeline_runs_end_to_end_within_vram_budget() -> None:
    """O pipeline canônico completo roda com os 4 backends reais sem estourar o budget de VRAM.

    Usa um ``ModelLifecycleManager`` compartilhado (ver factory.py) para que
    os 4 modelos nunca fiquem residentes ao mesmo tempo; sem isso, a soma dos
    picos individuais (SAM+DINOv2+CLIP+Qwen) estoura os 8GB de referência.
    """
    config = research_quality_config(multi_scale_justified=False, real_backends=True)
    lifecycle = ModelLifecycleManager()
    ports = create_perception_ports(config, lifecycle)
    payload = _real_test_payload()
    observation = image_observation(width=payload.width, height=payload.height)

    result = run_canonical_pipeline(observation, payload, config, ports)

    assert result.audit.errors == ()
    for metrics in lifecycle.metrics:
        assert metrics.peak_memory_bytes / (1024**3) <= config.gpu_memory_budget_gb
