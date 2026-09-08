"""Schema de configuração validada do módulo.

Issues: #157 (schema base), #191/#192 (amostragem de evidência densa),
#193/#194 (slots multi-contexto), #196 (fronteira de calibração).

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
from pathlib import Path
from typing import Any

from visual_perception.domain.image_area import ImageAreaGeometry, ImageAreaMasks
from visual_perception.domain.region_evidence import FOREGROUND_SLOTS, EvidenceSlot
from visual_perception.domain.region_reasoning import SceneContextMode


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
    #: Qualidade mínima que o SAM exige da máscara predita, e estabilidade
    #: mínima sob perturbação do limiar de binarização. Os dois já existiam no
    #: backend e nunca eram passados: o adapter chamava o pipeline só com
    #: ``points_per_batch``, de modo que "estabilidade" não era um controle
    #: disponível ao módulo. Os defaults são os do próprio backend, então
    #: declará-los não muda a execução — muda o fato de poderem ser ajustados
    #: e registrados no fingerprint.
    pred_iou_threshold: float = 0.88
    stability_score_threshold: float = 0.95

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
        for name in ("pred_iou_threshold", "stability_score_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"region_discovery.{name} must be in [0, 1], got {value}.")
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
# Declara as áreas de imagem que limitam a evidência utilizável de um frame.
# A *geometria* é específica de rig e dataset, então quem a preenche é a
# aplicação ou o experimento que compõe a execução — o módulo só define o
# contract e a aplica. Existe aqui, e não no harness, porque a exclusão
# acontece dentro do pipeline: até a #202 ela era uma flag de linha de comando
# que pintava pixels, e pintar pixels foi o que colapsou 45 de 45 regiões em
# ``curved wall`` (ver docs/known-limitations.md).
@dataclass(frozen=True)
class ImageAreaConfig:
    """As áreas declaradas do frame que limitam a evidência utilizável.

    Os limiares que decidem exclusão vivem em :class:`ProposalFilterConfig`:
    aqui fica só a geometria, que é a parte específica de rig e dataset.

    Argumentos:
        valid_area: área utilizável do sensor; ``None`` desliga a exclusão.
        ego_vehicle: área ocupada pelo rig; ``None`` desliga a exclusão.
    """

    valid_area: ImageAreaGeometry | None = None
    ego_vehicle: ImageAreaGeometry | None = None

    # Rasteriza as geometrias declaradas na resolução do frame. Chamada uma vez
    # por frame pelo pipeline, para que a filtragem compare máscaras já prontas
    # em vez de recalcular a geometria por proposal.
    def rasterize(self, width: int, height: int) -> ImageAreaMasks:
        """Converte as geometrias declaradas nas máscaras daquele frame.

        Argumentos:
            width: largura do frame em pixels.
            height: altura do frame em pixels.
        Retorna:
            as máscaras correspondentes; ausentes quando nada foi declarado.
        """
        return ImageAreaMasks(
            valid_area=None if self.valid_area is None else self.valid_area.rasterize(width, height),
            ego_vehicle=(
                None if self.ego_vehicle is None else self.ego_vehicle.rasterize(width, height)
            ),
        )


# Define os limiares geométricos que separam evidência redundante de evidência
# nova entre discovery e merge. Existe separada de RegionMergeConfig porque
# merge *une* propostas que descrevem a mesma região, enquanto esta filtragem
# *descarta* propostas que não acrescentam evidência — as duas evoluem por
# razões diferentes.
@dataclass(frozen=True)
class ProposalFilterConfig:
    """Limiares de área e redundância aplicados às proposals de discovery.

    Este estágio decide **validade**, não redundância. Duplicatas geométricas
    continuam sendo problema de ``merge_regions`` (:class:`RegionMergeConfig`),
    que as une preservando os dois ``contributing_proposal_ids`` em vez de
    descartar uma delas — descartar aqui perderia essa proveniência e tornaria
    o caminho de IoU do merge inalcançável.

    Nenhum destes campos é um limite de contagem: um ``top-N`` cego atingiria
    qualquer meta de número descartando evidência útil junto com a redundante.

    Argumentos:
        min_relative_area: área mínima de uma proposal, como fração do frame.
        max_relative_area: área máxima; acima disso a proposal descreve a cena.
        min_valid_overlap: fração mínima de uma proposal dentro da área válida
            do sensor para que ela seja aceita.
        max_ego_overlap: fração máxima de uma proposal sobre o rig antes de ela
            ser descartada.
    """

    min_relative_area: float = 0.002
    max_relative_area: float = 0.6
    min_valid_overlap: float = 0.5
    max_ego_overlap: float = 0.3

    # Valida as frações e a coerência entre área mínima e máxima, para que uma
    # configuração impossível falhe na construção e não como frame vazio.
    def __post_init__(self) -> None:
        """Valida as frações e a ordem entre área mínima e máxima."""
        for name in (
            "min_relative_area",
            "max_relative_area",
            "min_valid_overlap",
            "max_ego_overlap",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"proposal_filter.{name} must be in [0, 1], got {value}.")
        if self.min_relative_area >= self.max_relative_area:
            raise ValueError(
                "proposal_filter.min_relative_area must be smaller than max_relative_area, got "
                f"{self.min_relative_area} and {self.max_relative_area}."
            )


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Seleciona o backbone e a produção de evidência densa (#191/#192/#208)."""

    backend: str = "fake"
    checkpoint: str = "none"
    feature_resolution: int = 16
    device: str = "auto"
    upsampling: str = "nearest"
    input_resolution: int | None = None
    max_feature_map_mb: int = 384
    upsampler_repository: str = "mhamilton723/FeatUp:6b5a6c0e91f75e69194807128dcbc39c3084a30d"
    upsampler_checkpoint: str = (
        "https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/"
        "pretrained/dinov2_jbu_stack_cocostuff.ckpt"
    )
    upsampler_checkpoint_digest: str = (
        "438dad4dedb3c3ba4b7fa13d1c8eb465dafd323dadc039afd04b15178ea7b38c"
    )
    fallback_backend: str | None = None
    fallback_checkpoint: str | None = None

    # Garante que a resolução do feature map configurada é um valor
    # utilizável (positivo) e que a regra de amostragem densa é conhecida,
    # antes de chegar ao backend de extração. O default ``nearest`` preserva
    # o comportamento canônico do pipeline (pooling pixel-aligned por vizinho
    # mais próximo); ``patch_grid`` é o baseline explícito da ablation #192.
    def __post_init__(self) -> None:
        """Rejeita resoluções e métodos de amostragem incompatíveis."""
        if self.feature_resolution <= 0:
            raise ValueError("feature_extraction.feature_resolution must be positive.")
        if self.input_resolution is not None and self.input_resolution <= 0:
            raise ValueError("feature_extraction.input_resolution must be positive when provided.")
        if self.max_feature_map_mb <= 0:
            raise ValueError("feature_extraction.max_feature_map_mb must be positive.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("feature_extraction.device must be 'auto', 'cpu', or 'cuda'.")
        if self.upsampling not in {"patch_grid", "nearest", "bilinear"}:
            raise ValueError("feature_extraction.upsampling must be patch_grid, nearest or bilinear.")
        if self.fallback_backend not in {None, "dinov2"}:
            raise ValueError("feature_extraction.fallback_backend must be None or 'dinov2'.")
        if self.fallback_backend is not None and not self.fallback_checkpoint:
            raise ValueError("feature_extraction.fallback_backend requires fallback_checkpoint.")
        if len(self.upsampler_checkpoint_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.upsampler_checkpoint_digest
        ):
            raise ValueError("feature_extraction.upsampler_checkpoint_digest must be SHA-256 hex.")


# Configuração da extração de evidência multi-contexto por região (#193/#194).
# Existe aqui, e não junto do contract de domínio, porque é configuração de
# algoritmo: quais slots o módulo gasta compute para produzir, e com que
# margem de contexto — a mesma regra de ownership que vale para as demais
# sub-configs deste arquivo.
@dataclass(frozen=True)
class MultiContextConfig:
    """Quais slots de evidência de região o pipeline extrai, e com que geometria."""

    foreground_enabled: bool = True
    #: Se o reasoner recebe o sujeito isolado sobre fundo neutro. Ligado por
    #: default: até a #202 as únicas views textuais eram recortes de bounding
    #: box sem indicação de máscara, e numa região cuja máscara ocupa parte
    #: pequena da caixa o modelo interpretava a caixa.
    masked_subject_enabled: bool = True
    tight_crop_enabled: bool = True
    contextual_crop_enabled: bool = False
    scene_conditioned_enabled: bool = False
    context_expansion: float = 0.25
    masked_tight_crop: bool = False

    # Garante que a margem de contexto é uma fração utilizável e que pedir
    # evidência condicionada à cena não faz sentido sem um crop de contexto
    # para condicionar.
    def __post_init__(self) -> None:
        """Rejeita margens de contexto inválidas e combinações incoerentes de slot."""
        if not 0.0 <= self.context_expansion <= 4.0:
            raise ValueError("multi_context.context_expansion must be in [0, 4].")
        if self.contextual_crop_enabled and self.context_expansion <= 0.0:
            raise ValueError(
                "multi_context.contextual_crop_enabled requires a positive context_expansion."
            )


# Configuração da fronteira de calibração semântica (#196). Existe para que
# a regra de calibração seja selecionável e versionada por configuração, e
# para que o artifact de calibração participe do fingerprint de cache: uma
# tabela de calibração diferente produz claims diferentes.
@dataclass(frozen=True)
class CalibrationConfig:
    """Qual regra de calibração pontua claims, e quando ela deve se abster."""

    enabled: bool = False
    method: str = "reliability_table"
    version: str = "calibration/1"
    artifact_path: str | None = None
    min_visual_support: float = 0.2
    min_region_quality: float = 0.0
    domain: str = "unspecified"

    # Valida o método, a versão e os limiares de abstenção, e exige um
    # artifact quando o método é orientado a dados: calibrar sem artifact
    # seria inventar a curva de confiança.
    def __post_init__(self) -> None:
        """Rejeita métodos desconhecidos, limiares inválidos e calibração sem artifact."""
        if self.method not in {"reliability_table", "temperature"}:
            raise ValueError("calibration.method must be 'reliability_table' or 'temperature'.")
        if not self.version:
            raise ValueError("calibration.version must not be empty.")
        for name in ("min_visual_support", "min_region_quality"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"calibration.{name} must be in [0, 1].")
        if not self.domain:
            raise ValueError("calibration.domain must not be empty.")
        if self.enabled and self.artifact_path is None:
            raise ValueError(
                "calibration.enabled requires an artifact_path: a calibration rule without "
                "measured data would fabricate the confidence it claims to calibrate."
            )


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
    prompt_version: str = "v6"
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    load_in_4bit: bool = False
    #: Quais views de região o reasoner recebe, na ordem dos slots (#203).
    #: Uma view listada aqui mas cujo slot está desabilitado em
    #: ``multi_context`` simplesmente não é produzida, então este campo
    #: descreve o teto de evidência, não uma exigência.
    region_views: tuple[str, ...] = (
        EvidenceSlot.MASKED_SUBJECT.value,
        EvidenceSlot.TIGHT_CROP.value,
        EvidenceSlot.CONTEXTUAL_CROP.value,
    )
    #: Se as claims de cena acompanham cada região no prompt. Governa o canal
    #: *textual* de contexto; ``region_views`` governa o *visual*. Os dois são
    #: ablatáveis de forma independente, e foi essa separação que permitiu
    #: medir de qual deles vinha o vazamento: em ``corridor-02-002``, desligar
    #: só o canal visual derrubou o eco de cena de 45/45 regiões para 2/60,
    #: com as claims de cena ainda no prompt. O default segue
    #: ``context_assisted`` porque o modo local-first mediu levemente pior no
    #: colapso de labels, e o eco que ele elimina é residual.
    scene_context_mode: str = SceneContextMode.CONTEXT_ASSISTED.value

    # Garante que a versão do prompt está definida, já que ela identifica
    # qual template estruturado o backend deve usar. ``v4`` troca o exemplo de
    # formato por placeholders, depois de o exemplo concreto de ``v3`` ser
    # medido como a resposta padrão do modelo em 27/54 regiões de um frame;
    # ``v3`` introduziu o request multi-view com contexto de cena estruturado
    # (#202/#203); ``v2`` havia introduzido o contract
    # label/kind/category/confidence/alternatives e a confiança opcional. A
    # versão entra em ModelProvenance e no fingerprint() de cache, então o bump
    # é o que invalida resultados do prompt anterior.
    #
    # ``region_views`` é validado contra o vocabulário fechado de
    # EvidenceSlot e precisa conter ao menos uma view de foreground: sem
    # evidência local, a interpretação descreveria o entorno da região.
    def __post_init__(self) -> None:
        if not self.prompt_version:
            raise ValueError("multimodal_reasoning.prompt_version must not be empty.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("multimodal_reasoning.device must be 'auto', 'cpu', or 'cuda'.")
        if self.max_new_tokens <= 0:
            raise ValueError("multimodal_reasoning.max_new_tokens must be positive.")
        if self.temperature < 0.0:
            raise ValueError("multimodal_reasoning.temperature must not be negative.")
        known = {slot.value for slot in EvidenceSlot}
        unknown = [name for name in self.region_views if name not in known]
        if unknown:
            raise ValueError(
                f"multimodal_reasoning.region_views has unknown evidence slots: {unknown}."
            )
        if len(set(self.region_views)) != len(self.region_views):
            raise ValueError("multimodal_reasoning.region_views must not repeat an evidence slot.")
        if not any(EvidenceSlot(name) in FOREGROUND_SLOTS for name in self.region_views):
            raise ValueError(
                "multimodal_reasoning.region_views must include at least one foreground slot: "
                "interpreting a region from context alone would describe its surroundings."
            )
        if self.scene_context_mode not in {mode.value for mode in SceneContextMode}:
            raise ValueError(
                "multimodal_reasoning.scene_context_mode must be "
                f"{sorted(mode.value for mode in SceneContextMode)}."
            )


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
    multi_context: MultiContextConfig = field(default_factory=MultiContextConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    image_area: ImageAreaConfig = field(default_factory=ImageAreaConfig)
    proposal_filter: ProposalFilterConfig = field(default_factory=ProposalFilterConfig)

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
            ("multi_context", MultiContextConfig),
            ("calibration", CalibrationConfig),
            ("image_area", ImageAreaConfig),
            ("proposal_filter", ProposalFilterConfig),
        ):
            if key in payload and isinstance(payload[key], dict):
                payload[key] = config_type(**payload[key])
        return ModuleConfig(**payload)

    # Calcula uma identidade estável da configuração inteira, usada como
    # chave de cache (#170) para evitar reprocessar a mesma imagem com a
    # mesma configuração.
    def fingerprint(self) -> str:
        """Um hash estável da configuração completa, usado para caching (#170)."""
        payload = self.to_dict()
        if self.calibration.artifact_path is not None:
            payload["calibration_artifact_digest"] = hashlib.sha256(
                Path(self.calibration.artifact_path).read_bytes()
            ).hexdigest()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
