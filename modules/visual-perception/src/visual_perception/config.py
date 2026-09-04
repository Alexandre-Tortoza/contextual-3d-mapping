"""Schema de configuração validada do módulo.

Issue: #157.

Este módulo possui os parâmetros que controlam seus próprios algoritmos
(identificadores de backend, checkpoints, resoluções, thresholds, quality
profile). Origem de dataset, sincronização e composição de aplicação são de
posse da aplicação consumidora, não deste schema (ver
``docs/engineering-principles.md`` "Configuration ownership").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


# Enumera os perfis de execução que o módulo pode otimizar. Existe porque o
# módulo precisa escolher entre priorizar qualidade científica ou custo
# computacional/GPU, e essa escolha afeta várias configs abaixo ao mesmo
# tempo (ex: valida incompatibilidade entre REDUCED_COST e tiling
# multi-scale em ModuleConfig.__post_init__).
class QualityProfile(StrEnum):
    """Define para qual perfil de execução o módulo deve otimizar.

    Issue: #181 define o profile ``RESEARCH_QUALITY`` em detalhe.
    """

    RESEARCH_QUALITY = "research_quality"
    REDUCED_COST = "reduced_cost"


# Configuração do backend de descoberta de regiões (RegionDiscoverer).
# Existe para manter os parâmetros do algoritmo de region discovery
# isolados e validados, seguindo a regra de ownership de configuração de
# AGENTS.md.
@dataclass(frozen=True)
class RegionDiscoveryConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    score_threshold: float = 0.5
    device: str = "auto"
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    max_regions: int = 100
    min_mask_area: int = 64

    # Valida o invariante de fronteira desta config logo após a construção,
    # falhando cedo com um erro acionável em vez de deixar um threshold
    # inválido se propagar para o pipeline.
    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("region_discovery.score_threshold must be in [0, 1].")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("region_discovery.device must be 'auto', 'cpu', or 'cuda'.")
        if not self.model_config:
            raise ValueError("region_discovery.model_config must not be empty.")
        if self.max_regions <= 0:
            raise ValueError("region_discovery.max_regions must be positive.")
        if self.min_mask_area <= 0:
            raise ValueError("region_discovery.min_mask_area must be positive.")


# Configuração do estágio de tiling (divisão da imagem em tiles/multi-scale)
# do pipeline. Existe para permitir trade-off entre custo e cobertura de
# detecção em imagens grandes.
@dataclass(frozen=True)
class TilingConfig:
    multi_scale_enabled: bool = False
    tile_grid: str = "1x1"
    overlap_ratio: float = 0.2

    # Valida que ``tile_grid`` está no formato esperado ("<linhas>x<colunas>")
    # e que ``overlap_ratio`` é um invariante de fronteira válido, falhando
    # cedo com mensagens acionáveis.
    def __post_init__(self) -> None:
        if "x" not in self.tile_grid:
            raise ValueError("tiling.tile_grid must look like '<rows>x<cols>', e.g. '2x2'.")
        rows, _, cols = self.tile_grid.partition("x")
        if not (rows.isdigit() and cols.isdigit() and int(rows) > 0 and int(cols) > 0):
            raise ValueError(f"tiling.tile_grid is not a valid grid: {self.tile_grid!r}.")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("tiling.overlap_ratio must be in [0, 1).")


# Configuração do estágio de merge de regiões (deduplicação de propostas
# sobrepostas vindas de tiles/scales diferentes). Existe para controlar
# quão agressivamente regiões duplicadas são unificadas antes da fusão
# semântica.
@dataclass(frozen=True)
class RegionMergeConfig:
    iou_merge_threshold: float = 0.85
    containment_merge_threshold: float = 0.9

    # Valida que os dois thresholds de merge são frações válidas em [0, 1].
    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_merge_threshold <= 1.0:
            raise ValueError("merge.iou_merge_threshold must be in [0, 1].")
        if not 0.0 <= self.containment_merge_threshold <= 1.0:
            raise ValueError("merge.containment_merge_threshold must be in [0, 1].")


# Configuração do backend de extração de features densas (DenseFeatureExtractor).
@dataclass(frozen=True)
class FeatureExtractionConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    feature_resolution: int = 16
    device: str = "auto"

    # Garante que a resolução do feature map configurada é um valor
    # utilizável (positivo) antes de chegar ao backend de extração.
    def __post_init__(self) -> None:
        if self.feature_resolution <= 0:
            raise ValueError("feature_extraction.feature_resolution must be positive.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("feature_extraction.device must be 'auto', 'cpu', or 'cuda'.")


# Configuração do backend de embedding alinhado com linguagem (LanguageAlignedEncoder).
@dataclass(frozen=True)
class LanguageEmbeddingConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    dimension: int = 512
    device: str = "auto"
    model_name: str = "ViT-B-32"
    normalize: bool = True

    # Garante que a dimensão do embedding configurada é utilizável (positiva).
    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("language_embedding.dimension must be positive.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("language_embedding.device must be 'auto', 'cpu', or 'cuda'.")
        if not self.model_name:
            raise ValueError("language_embedding.model_name must not be empty.")


# Configuração do backend de raciocínio multimodal (MultimodalReasoner).
@dataclass(frozen=True)
class MultimodalReasoningConfig:
    backend: str = "fake"
    checkpoint: str = "none"
    prompt_version: str = "v1"
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    load_in_4bit: bool = False

    # Garante que a versão do prompt está definida, já que ela identifica
    # qual template estruturado o backend deve usar.
    def __post_init__(self) -> None:
        if not self.prompt_version:
            raise ValueError("multimodal_reasoning.prompt_version must not be empty.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("multimodal_reasoning.device must be 'auto', 'cpu', or 'cuda'.")
        if self.max_new_tokens <= 0:
            raise ValueError("multimodal_reasoning.max_new_tokens must be positive.")
        if self.temperature < 0.0:
            raise ValueError("multimodal_reasoning.temperature must not be negative.")


# Configuração raiz do módulo: agrega todas as sub-configs acima em um único
# objeto reproduzível. Existe porque o pipeline inteiro precisa de uma única
# fonte de verdade de configuração, serializável e com fingerprint estável
# para caching (#170).
@dataclass(frozen=True)
class ModuleConfig:
    """A configuração completa e reproduzível de posse de visual-perception."""

    quality_profile: QualityProfile = QualityProfile.RESEARCH_QUALITY
    gpu_memory_budget_gb: float = 8.0
    region_discovery: RegionDiscoveryConfig = field(default_factory=RegionDiscoveryConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    merge: RegionMergeConfig = field(default_factory=RegionMergeConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    language_embedding: LanguageEmbeddingConfig = field(default_factory=LanguageEmbeddingConfig)
    multimodal_reasoning: MultimodalReasoningConfig = field(
        default_factory=MultimodalReasoningConfig
    )

    # Valida invariantes que dependem de mais de um campo ao mesmo tempo
    # (o que os ``__post_init__`` das sub-configs não conseguem verificar
    # sozinhos), como a incompatibilidade entre o profile REDUCED_COST e
    # tiling multi-scale (#181).
    def __post_init__(self) -> None:
        if self.gpu_memory_budget_gb <= 0:
            raise ValueError("gpu_memory_budget_gb must be positive.")
        if (
            self.quality_profile is QualityProfile.REDUCED_COST
            and self.tiling.multi_scale_enabled
        ):
            raise ValueError(
                "Incompatible configuration: 'reduced_cost' quality profile does not support "
                "multi-scale tiling (see issue #181)."
            )

    # Serializa a configuração inteira (incluindo sub-configs aninhadas) em
    # um dict simples, usado tanto para persistência quanto para o cálculo
    # de ``fingerprint``.
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_profile"] = self.quality_profile.value
        return payload

    # Reconstrói um ``ModuleConfig`` a partir de um dict (o inverso de
    # ``to_dict``), usado ao carregar configuração persistida ou vinda de
    # um arquivo de config da aplicação.
    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ModuleConfig:
        payload = dict(payload)
        payload["quality_profile"] = QualityProfile(
            payload.get("quality_profile", QualityProfile.RESEARCH_QUALITY.value)
        )
        for key, config_type in (
            ("region_discovery", RegionDiscoveryConfig),
            ("tiling", TilingConfig),
            ("merge", RegionMergeConfig),
            ("feature_extraction", FeatureExtractionConfig),
            ("language_embedding", LanguageEmbeddingConfig),
            ("multimodal_reasoning", MultimodalReasoningConfig),
        ):
            if key in payload and isinstance(payload[key], dict):
                payload[key] = config_type(**payload[key])
        return ModuleConfig(**payload)

    # Calcula uma identidade estável da configuração inteira, usada como
    # chave de cache (#170) para evitar reprocessar a mesma imagem com a
    # mesma configuração.
    def fingerprint(self) -> str:
        """Um hash estável da configuração completa, usado para caching (#170)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
