"""Execution profile quality-first.

Issue: #181.

Define a política de seleção de referência de pesquisa: qualidade é o
alvo primário de otimização *sujeito ao* budget de memória de GPU
configurado. Latência e throughput são medidos mas nunca usados para
rejeitar um candidato que cabe no budget de memória. Compute adicional
(passes multi-scale, modelos maiores, reprocessamento seletivo) só é
ligado na config de referência quando evidência de benchmark (#174,
#175) mostra que ele realmente melhora a qualidade.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.config import (
    FeatureExtractionConfig,
    LanguageEmbeddingConfig,
    ModuleConfig,
    MultimodalReasoningConfig,
    QualityProfile,
    RegionDiscoveryConfig,
    TilingConfig,
)


# Representa uma opção de backend já avaliada por benchmark para um dado
# stage (ver #174), guardando as métricas que a seleção de referência usa
# para comparar candidatos.
@dataclass(frozen=True)
class BackendCandidate:
    """Uma opção de backend avaliada por benchmark para um dado stage (ver #174)."""

    name: str
    quality_score: float
    peak_vram_gb: float
    latency_s: float

    # Valida que as métricas de recurso do candidato são fisicamente
    # plausíveis (não negativas) assim que o candidato é construído.
    def __post_init__(self) -> None:
        if self.peak_vram_gb < 0 or self.latency_s < 0:
            raise ValueError("peak_vram_gb and latency_s must be non-negative.")


# Implementa a política "qualidade primeiro, sujeito ao budget de memória":
# escolhe o candidato de maior qualidade entre os que cabem no budget de
# GPU, usada por research_quality_config para montar a config de referência.
def select_research_quality_backend(
    candidates: tuple[BackendCandidate, ...], memory_budget_gb: float
) -> BackendCandidate:
    """Escolhe o candidato de maior qualidade que cabe no budget de memória.

    Latência nunca exclui um candidato; apenas ``peak_vram_gb`` exclui.
    """
    affordable = tuple(c for c in candidates if c.peak_vram_gb <= memory_budget_gb)
    if not affordable:
        raise ValueError(
            f"No candidate fits the {memory_budget_gb} GB memory budget: "
            f"{[c.name for c in candidates]}."
        )
    return max(affordable, key=lambda c: (c.quality_score, -c.peak_vram_gb, c.name))


# Decide se compute adicional (ex: multi-scale, #159) deve ser habilitado,
# a partir de uma comparação de qualidade medida por benchmark, em vez de
# uma suposição a priori de que "mais compute é melhor".
def additional_compute_is_justified(
    baseline_quality: float, enhanced_quality: float, minimum_improvement: float = 0.0
) -> bool:
    """Indica se a evidência medida justifica habilitar compute extra (ex: multi-scale, #159).

    ``minimum_improvement`` protege contra habilitar compute custoso por
    ganhos no nível de ruído.
    """
    return enhanced_quality > baseline_quality + minimum_improvement


# Backends selecionados pelo benchmark #174 sob o orçamento de referência de
# 8GB na RTX 3060 (ver benchmarks/results/benchmark-174-*.json). Mantidos
# aqui, próximos de research_quality_config, para que a config de referência
# real-backend tenha uma única fonte de verdade.
_REAL_REGION_DISCOVERY = RegionDiscoveryConfig(backend="sam", checkpoint="facebook/sam-vit-huge")
_REAL_FEATURE_EXTRACTION = FeatureExtractionConfig(backend="dinov2", checkpoint="facebook/dinov2-base")
_REAL_LANGUAGE_EMBEDDING = LanguageEmbeddingConfig(
    backend="clip", checkpoint="openai/clip-vit-large-patch14", dimension=768
)
_REAL_MULTIMODAL_REASONING = MultimodalReasoningConfig(
    backend="qwen_vl", checkpoint="Qwen/Qwen2.5-VL-3B-Instruct", load_in_4bit=True
)


# Monta a config de referência research-quality do módulo, combinando o
# budget de memória com a decisão (já tomada externamente, por benchmark)
# de habilitar ou não multi-scale, e opcionalmente os backends reais
# selecionados pelo benchmark #174 em vez dos fakes.
def research_quality_config(
    *,
    multi_scale_justified: bool,
    gpu_memory_budget_gb: float = 8.0,
    real_backends: bool = False,
) -> ModuleConfig:
    """Constrói a configuração de referência research-quality.

    ``multi_scale_justified`` deve vir de uma comparação por benchmark (ver
    :func:`additional_compute_is_justified`); esta função não decide isso
    sozinha. ``real_backends=False`` (o default) mantém os quatro estágios
    em ``"fake"`` — o módulo continua GPU-free por padrão; passar
    ``real_backends=True`` opta pelos backends reais benchmark-selecionados
    (#186-#190).
    """
    return ModuleConfig(
        quality_profile=QualityProfile.RESEARCH_QUALITY,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        tiling=TilingConfig(multi_scale_enabled=multi_scale_justified, tile_grid="2x2"),
        region_discovery=_REAL_REGION_DISCOVERY if real_backends else RegionDiscoveryConfig(),
        feature_extraction=_REAL_FEATURE_EXTRACTION if real_backends else FeatureExtractionConfig(),
        language_embedding=_REAL_LANGUAGE_EMBEDDING if real_backends else LanguageEmbeddingConfig(),
        multimodal_reasoning=_REAL_MULTIMODAL_REASONING if real_backends else MultimodalReasoningConfig(),
    )
