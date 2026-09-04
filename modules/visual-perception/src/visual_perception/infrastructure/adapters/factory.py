"""Composição dos ports de percepção a partir da configuração do módulo.

Este é o único ponto que escolhe fakes ou adapters reais. A pipeline recebe
apenas ``PerceptionPorts`` e não precisa conhecer bibliotecas, checkpoints ou
nomes de backend.
"""

from __future__ import annotations

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.application.pipeline import PerceptionPorts
from visual_perception.config import ModuleConfig
from visual_perception.infrastructure.adapters.feature_extraction_backend import (
    RealDenseFeatureExtractionAdapter,
)
from visual_perception.infrastructure.adapters.language_embedding_backend import (
    RealLanguageAlignedEncoderAdapter,
)
from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import (
    RealMultimodalReasoningAdapter,
)
from visual_perception.infrastructure.adapters.region_discovery_backend import RealRegionDiscoveryAdapter
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner
from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer


# Cria os ports que a aplicação entrega à pipeline, sem adicionar stages ou
# alterar o fluxo canônico. Cada seleção é independente para permitir
# benchmark. Quando ao menos um backend real é selecionado, os 4 adapters
# reais compartilham um único ``ModelLifecycleManager`` (``lifecycle``, opcional
# — criado automaticamente se omitido) para que no máximo um modelo pesado
# fique residente em VRAM por vez, mesmo com os 4 ports já construídos (#171).
def create_perception_ports(
    config: ModuleConfig, lifecycle: ModelLifecycleManager | None = None
) -> PerceptionPorts:
    """Compõe ports fake ou reais conforme os backends declarados na configuração."""
    lifecycle = lifecycle or ModelLifecycleManager()
    return PerceptionPorts(
        region_discoverer=_region_discoverer_for(config, lifecycle),
        feature_extractor=_feature_extractor_for(config, lifecycle),
        language_encoder=_language_encoder_for(config, lifecycle),
        multimodal_reasoner=_multimodal_reasoner_for(config, lifecycle),
    )


# Seleciona o backend de geometria sem atribuir semântica a propostas.
def _region_discoverer_for(
    config: ModuleConfig, lifecycle: ModelLifecycleManager
) -> FakeRegionDiscoverer | RealRegionDiscoveryAdapter:
    """Seleciona o port de descoberta de regiões declarado pela configuração."""
    if config.region_discovery.backend == "fake":
        return FakeRegionDiscoverer()
    if config.region_discovery.backend == "sam":
        return RealRegionDiscoveryAdapter(lifecycle)
    raise ValueError(f"Backend de region discovery não suportado: {config.region_discovery.backend!r}.")


# Seleciona o backend que produz o FeatureMap denso para o pooling existente.
def _feature_extractor_for(
    config: ModuleConfig, lifecycle: ModelLifecycleManager
) -> FakeDenseFeatureExtractor | RealDenseFeatureExtractionAdapter:
    """Seleciona o port de extração de features declarado pela configuração."""
    if config.feature_extraction.backend == "fake":
        return FakeDenseFeatureExtractor()
    if config.feature_extraction.backend == "dinov2":
        return RealDenseFeatureExtractionAdapter(lifecycle)
    raise ValueError(f"Backend de feature extraction não suportado: {config.feature_extraction.backend!r}.")


# Seleciona o encoder que mantém imagem e texto no mesmo espaço de embedding.
def _language_encoder_for(
    config: ModuleConfig, lifecycle: ModelLifecycleManager
) -> FakeLanguageAlignedEncoder | RealLanguageAlignedEncoderAdapter:
    """Seleciona o port de embedding alinhado à linguagem declarado pela configuração."""
    if config.language_embedding.backend == "fake":
        return FakeLanguageAlignedEncoder()
    if config.language_embedding.backend == "clip":
        return RealLanguageAlignedEncoderAdapter(lifecycle)
    raise ValueError(f"Backend de language embedding não suportado: {config.language_embedding.backend!r}.")


# Seleciona o VLM responsável pelos prompts estruturados de cena e região.
def _multimodal_reasoner_for(
    config: ModuleConfig, lifecycle: ModelLifecycleManager
) -> FakeMultimodalReasoner | RealMultimodalReasoningAdapter:
    """Seleciona o port de raciocínio multimodal declarado pela configuração."""
    if config.multimodal_reasoning.backend == "fake":
        return FakeMultimodalReasoner()
    if config.multimodal_reasoning.backend == "qwen_vl":
        return RealMultimodalReasoningAdapter(lifecycle)
    raise ValueError(
        "Backend de multimodal reasoning não suportado: "
        f"{config.multimodal_reasoning.backend!r}."
    )
