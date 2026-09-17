"""Execution profile quality-first.

Issue: #181.

Define a política de seleção de referência de pesquisa: qualidade é o
alvo primário de otimização *sujeito ao* budget de memória de GPU
configurado. Latência e throughput são medidos mas nunca usados para
rejeitar um candidato que cabe no budget de memória. O perfil ativa discovery
multi-scale por default: uma passada global preserva contexto e tiles
sobrepostos ampliam a cobertura de detalhes finos. Ablações e benchmarks que
exigem geometria fixa desligam esse modo explicitamente.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.config import (
    FeatureExtractionConfig,
    HypothesisSupportConfig,
    LanguageEmbeddingConfig,
    ModuleConfig,
    MultiContextConfig,
    MultimodalReasoningConfig,
    QualityProfile,
    RefinementConfig,
    RegionDiscoveryConfig,
    SemanticGroundingConfig,
    TilingConfig,
)
from visual_perception.domain.region_evidence import EvidenceSlot


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


# Backends selecionados pelo benchmark #174 sob o orçamento de referência de
# 8GB na RTX 3060 (ver benchmarks/results/benchmark-174-*.json). Mantidos
# aqui, próximos de research_quality_config, para que a config de referência
# real-backend tenha uma única fonte de verdade.
# min_mask_area acima do default (64px², um patch de 8x8): o SAM automático
# propõe muitos slivers minúsculos em superfícies uniformes (teto, parede)
# que dão pouco sinal visual ao VLM e inflam o over-segmentation sem
# agregar conteúdo distinto (visto na prática em #190: ~82% de falha de
# interpretação antes de ajustar o prompt, muitas delas em crops <30x30px).
# SAM3 roda como segment everything pelo tracker (grade de pontos,
# class-agnostic). Prompts textuais genéricos do PCS ("all visible objects",
# "objects", "segment everything") devolveram zero máscaras nos frames do
# corridor-02. Os thresholds 0.80/0.90 ficam abaixo dos defaults do SAM
# (0.88/0.95), que descartavam piso e quase todo o outdoor: na comparação de
# 2026-09-15, a cobertura foi de 11–54% para ~65–74% por frame.
_REAL_REGION_DISCOVERY = RegionDiscoveryConfig(
    backend="sam3",
    checkpoint="facebook/sam3",
    min_mask_area=500,
    pred_iou_threshold=0.80,
    stability_score_threshold=0.90,
)
_REAL_FEATURE_EXTRACTION = FeatureExtractionConfig(
    backend="dinov2",
    checkpoint="facebook/dinov2-base",
    input_resolution=448,
)
_REAL_LANGUAGE_EMBEDDING = LanguageEmbeddingConfig(
    backend="clip", checkpoint="openai/clip-vit-large-patch14", dimension=768
)
_REAL_MULTIMODAL_REASONING = MultimodalReasoningConfig(
    backend="qwen_vl",
    checkpoint="Qwen/Qwen2.5-VL-3B-Instruct",
    load_in_4bit=True,
    # v10: schema de prompt de região reduzido (2 campos), só para este
    # backend — ver o histórico de prompt_version em
    # MultimodalReasoningConfig e reasoning_prompts._MINIMAL_SCHEMA_BODY.
    # Generalizar isso para outros backends (era o default v9) quebrou o
    # grounding textual do gemini_robotics_er.
    prompt_version="v10",
    region_views=(
        EvidenceSlot.MASKED_SUBJECT.value,
        EvidenceSlot.TIGHT_CROP.value,
        EvidenceSlot.CONTEXTUAL_CROP.value,
        EvidenceSlot.SCENE_CONDITIONED.value,
    ),
)
_REAL_MULTI_CONTEXT = MultiContextConfig(
    foreground_enabled=True,
    tight_crop_enabled=True,
    contextual_crop_enabled=True,
    scene_conditioned_enabled=True,
    context_expansion=0.25,
    masked_tight_crop=False,
)
#: Com o crop contextual habilitado no perfil real, o suporte de hipótese e o
#: escalonamento do refinamento também o consomem. Ficam declarados aqui, ao
#: lado do ``multi_context`` que os habilita, porque a config recusa pedir um
#: slot que não é produzido.
_REAL_HYPOTHESIS_SUPPORT = HypothesisSupportConfig(
    slots=(
        EvidenceSlot.MASKED_SUBJECT.value,
        EvidenceSlot.TIGHT_CROP.value,
        EvidenceSlot.CONTEXTUAL_CROP.value,
    )
)
_REAL_REFINEMENT = RefinementConfig(
    escalation_views=(
        EvidenceSlot.FOREGROUND_DENSE.value,
        EvidenceSlot.MASKED_SUBJECT.value,
        EvidenceSlot.TIGHT_CROP.value,
        EvidenceSlot.CONTEXTUAL_CROP.value,
    )
)


# Monta a config de referência research-quality do módulo, combinando o
# budget de memória com o modo híbrido global+tiles do perfil e, opcionalmente,
# os backends reais selecionados pelo benchmark #174 em vez dos fakes.
def research_quality_config(
    *,
    multi_scale_enabled: bool = True,
    gpu_memory_budget_gb: float = 8.0,
    real_backends: bool = False,
) -> ModuleConfig:
    """Constrói a configuração de referência research-quality.

    ``multi_scale_enabled=True`` executa discovery na imagem completa e em
    quatro tiles 2x2 sobrepostos; a imagem completa preserva contexto global
    e os tiles elevam a cobertura de detalhes finos. Passar ``False`` é uma
    ablação explícita que mantém somente a imagem completa.

    ``real_backends=False`` (o default) mantém os quatro estágios em
    ``"fake"`` — o módulo continua GPU-free por padrão; passar
    ``real_backends=True`` opta pelos backends reais benchmark-selecionados
    (#186-#190).
    """
    return ModuleConfig(
        quality_profile=QualityProfile.RESEARCH_QUALITY,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        tiling=TilingConfig(multi_scale_enabled=multi_scale_enabled, tile_grid="2x2"),
        region_discovery=_REAL_REGION_DISCOVERY if real_backends else RegionDiscoveryConfig(),
        feature_extraction=_REAL_FEATURE_EXTRACTION if real_backends else FeatureExtractionConfig(),
        language_embedding=_REAL_LANGUAGE_EMBEDDING if real_backends else LanguageEmbeddingConfig(),
        multimodal_reasoning=_REAL_MULTIMODAL_REASONING if real_backends else MultimodalReasoningConfig(),
        multi_context=_REAL_MULTI_CONTEXT if real_backends else MultiContextConfig(),
        hypothesis_support=_REAL_HYPOTHESIS_SUPPORT if real_backends else HypothesisSupportConfig(),
        refinement=_REAL_REFINEMENT if real_backends else RefinementConfig(),
        semantic_grounding=SemanticGroundingConfig(backend="grounded_sam" if real_backends else "unavailable"),
    )
