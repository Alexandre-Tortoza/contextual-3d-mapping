"""Testa resolução nativa maior e elevação aprendida de features (#208)."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import payload_with_blobs
from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.errors import BackendUnavailableError
from visual_perception.domain.feature_map import (
    FeatureGeneration,
    FeatureMap,
    FeatureRepresentation,
    feature_map_spec_from_dict,
    feature_map_spec_to_dict,
)
from visual_perception.infrastructure.adapters.feature_extraction_backend import _scaled_size
from visual_perception.infrastructure.adapters.feature_upsampling_backend import (
    FallbackDenseFeatureExtractionAdapter,
    featup_input_size,
)


# Constrói um mapa mínimo com proveniência configurável para os doubles dos
# adapters, sem depender de torch, CUDA ou downloads.
def _feature_map(*, generation: FeatureGeneration = FeatureGeneration.BACKBONE_NATIVE) -> FeatureMap:
    """Retorna um mapa 2x2 finito usado pelos testes do contract."""
    return FeatureMap(
        data=np.ones((2, 2, 3), dtype=np.float32),
        stride_x=4.0,
        stride_y=4.0,
        dimension=3,
        model_id="test-backbone",
        generation=generation,
    )


# Confirma que elevar a entrada nativa mantém proporção e alinhamento ao patch.
def test_native_high_resolution_preserves_aspect_ratio_and_patch_alignment() -> None:
    """Dimensiona um frame 640x480 para uma grade nativa DINOv2 32x24."""
    height, width = _scaled_size(480, 640, 448, multiple=14)

    assert (height, width) == (336, 448)
    assert (height // 14, width // 14) == (24, 32)


# Confirma que a decisão de memória ocorre antes de construir o mapa elevado.
def test_featup_resolution_is_reduced_before_inference_to_fit_memory_budget() -> None:
    """Reduz a entrada pedida quando a saída JBU float32 excederia o teto."""
    height, width = featup_input_size(
        480,
        640,
        448,
        dimension=384,
        max_bytes=64 * 1024 * 1024,
    )
    output_bytes = (height // 14 * 16) * (width // 14 * 16) * 384 * 4

    assert output_bytes <= 64 * 1024 * 1024
    assert width < 448


# Protege a semântica da proveniência: um mapa aprendido não pode parecer
# uma simples interpolação e precisa identificar origem e checkpoint.
def test_learned_feature_map_requires_distinct_provenance() -> None:
    """Rejeita um mapa learned_upsampler sem metadata do JBU e grade fonte."""
    with pytest.raises(ValueError, match="elevated_grid"):
        FeatureMap(
            data=np.ones((2, 2, 3), dtype=np.float32),
            stride_x=1.0,
            stride_y=1.0,
            dimension=3,
            model_id="featup",
            generation=FeatureGeneration.LEARNED_UPSAMPLER,
        )


# Confirma que persistência de metadata mantém método, checkpoint e resolução efetiva.
def test_learned_provenance_round_trips_through_feature_spec() -> None:
    """Serializa a identidade completa do mapa FeatUp sem serializar seus valores."""
    feature_map = FeatureMap(
        data=np.ones((16, 16, 3), dtype=np.float32),
        stride_x=2.0,
        stride_y=2.0,
        dimension=3,
        model_id="featup-dinov2-small-jbu",
        representation=FeatureRepresentation.ELEVATED_GRID,
        checkpoint="facebookresearch/dinov2:dinov2_vits14",
        upsampling_method="featup_jbu_stack_cocostuff",
        generation=FeatureGeneration.LEARNED_UPSAMPLER,
        source_grid_width=1,
        source_grid_height=1,
        upsampler_checkpoint="https://example.test/featup.ckpt",
        upsampler_checkpoint_digest="a" * 64,
    )

    restored = feature_map_spec_from_dict(feature_map_spec_to_dict(feature_map))

    assert restored["generation"] == FeatureGeneration.LEARNED_UPSAMPLER.value
    assert restored["grid_width"] == 16
    assert restored["source_grid_width"] == 1
    assert restored["upsampler_checkpoint"] == "https://example.test/featup.ckpt"
    assert restored["upsampler_checkpoint_digest"] == "a" * 64


# Confirma que falha tipada do backend aprendido só cai no caminho declarado
# e deixa o motivo no resultado retornado ao consumidor.
def test_explicit_fallback_records_why_the_primary_backend_was_not_used() -> None:
    """Executa o fallback DINOv2 e preserva a causa da indisponibilidade FeatUp."""

    # Simula um backend opcional ausente sem capturar exceções arbitrárias.
    class MissingPrimary:
        """Extrator primário que representa uma instalação FeatUp ausente."""

        # Emite o erro operacional que o decorator tem autorização para tratar.
        def extract(self, image, config):  # noqa: ANN001
            """Falha como um backend opcional não instalado."""
            raise BackendUnavailableError("featup não instalado")

    # Simula o DINOv2 alternativo e verifica a configuração recebida.
    class WorkingFallback:
        """Extrator alternativo que registra a seleção DINOv2."""

        # Retorna o mapa mínimo apenas se o decorator trocou backend/checkpoint.
        def extract(self, image, config):  # noqa: ANN001
            """Produz um mapa nativo pelo backend alternativo declarado."""
            assert config.backend == "dinov2"
            assert config.checkpoint == "facebook/dinov2-base"
            return _feature_map()

    config = FeatureExtractionConfig(
        backend="featup",
        checkpoint="facebookresearch/dinov2:dinov2_vits14",
        fallback_backend="dinov2",
        fallback_checkpoint="facebook/dinov2-base",
    )
    adapter = FallbackDenseFeatureExtractionAdapter(MissingPrimary(), WorkingFallback())

    result = adapter.extract(payload_with_blobs(width=8, height=8), config)

    assert result.generation is FeatureGeneration.BACKBONE_NATIVE
    assert result.fallback_reason is not None
    assert "BackendUnavailableError" in result.fallback_reason
