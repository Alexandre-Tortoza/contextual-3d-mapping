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
from math import isfinite
from pathlib import Path
from typing import Any

from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry, ImageAreaMasks
from visual_perception.domain.region_evidence import FOREGROUND_SLOTS, EvidenceSlot
from visual_perception.domain.region_reasoning import SceneContextMode, TemporalPriorMode


# Configura exclusivamente grounding espacial; não altera interpretação, merge,
# frequência temporal ou a política de incerteza de sensor-association.
@dataclass(frozen=True)
class SemanticGroundingConfig:
    """Localização condicionada ao conceito e segmentação com lifecycle sequencial.

    Os limiares do detector seguem o exemplo oficial do Transformers; não são
    confiança calibrada. O mínimo de um pixel apenas rejeita máscaras vazias.
    """

    backend: str = "unavailable"
    detector_checkpoint: str = "IDEA-Research/grounding-dino-base"
    # SAM2 (não SAM1): o checkpoint SAM1 satura numericamente sob a stack
    # atual de torch/transformers (ver region_discovery_backend.py e
    # semantic_grounding_backend.py._segment).
    segmenter_checkpoint: str = "facebook/sam2.1-hiera-large"
    device: str = "auto"
    detection_threshold: float = 0.4
    text_threshold: float = 0.3
    duplicate_box_iou: float = 0.85
    min_mask_area: int = 1

    # Impede configurações silenciosamente inválidas de um stage ablatável.
    def __post_init__(self) -> None:
        """Valida backend, unidades e limiares de fronteira."""
        if self.backend not in {"unavailable", "grounded_sam"}:
            raise ValueError("semantic_grounding.backend must be unavailable or grounded_sam.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("semantic_grounding.device must be auto, cpu or cuda.")
        if not self.detector_checkpoint or not self.segmenter_checkpoint:
            raise ValueError("Grounding checkpoints must not be empty.")
        for name in ("detection_threshold", "text_threshold", "duplicate_box_iou"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"semantic_grounding.{name} must be in (0, 1].")
        if type(self.min_mask_area) is not int or self.min_mask_area < 1:
            raise ValueError("semantic_grounding.min_mask_area must be a positive integer.")


# Configura a descoberta de conceitos concretos na cena (#277), que alimenta o
# grounding por conceito como fonte adicional de propostas. Desligada por default:
# o discovery genérico continua sendo a fonte obrigatória de regiões.
@dataclass(frozen=True)
class SceneConceptDiscoveryConfig:
    """Teto, exclusões estruturais e versão do prompt da descoberta de conceitos."""

    enabled: bool = False
    #: Máximo de conceitos por frame. Cada conceito vira uma consulta de grounding
    #: e pode gerar regiões que depois custam uma chamada de VLM cada.
    max_concepts: int = 12
    #: Conceitos estruturais genéricos descartados quando aparecem sozinhos. Frases
    #: com evidência contextual ("wall crack", "flooded floor") são mantidas.
    structural_exclusions: tuple[str, ...] = ("wall", "walls", "floor", "floors", "ceiling", "ceilings")
    prompt_version: str = "concepts/v1"

    # Recusa teto e versão inválidos na fronteira da config.
    def __post_init__(self) -> None:
        """Valida teto de conceitos, exclusões e versão do prompt."""
        if self.max_concepts <= 0:
            raise ValueError("scene_concept_discovery.max_concepts must be positive.")
        if any(not item or item != item.strip().lower() for item in self.structural_exclusions):
            raise ValueError("scene_concept_discovery.structural_exclusions must be stripped lowercase terms.")
        if not self.prompt_version:
            raise ValueError("scene_concept_discovery.prompt_version must not be empty.")


# Configura o grounding por conceito (#277): cada conceito da descoberta de cena
# vira um prompt textual do SAM3 PCS, e as máscaras entram como propostas
# adicionais ao discovery genérico, nunca no lugar dele.
@dataclass(frozen=True)
class ConceptGroundingConfig:
    """Backend, limiares e teto de regiões do grounding por conceito."""

    backend: str = "unavailable"
    checkpoint: str = "facebook/sam3"
    device: str = "auto"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    min_mask_area: int = 500
    #: Máximo de máscaras aceitas por conceito; um conceito genérico demais não pode
    #: inundar o merge e as chamadas de VLM.
    max_regions_per_concept: int = 8

    # Recusa backend, device e limiares inválidos na fronteira da config.
    def __post_init__(self) -> None:
        """Valida backend, device, limiares e tetos."""
        if self.backend not in {"unavailable", "sam3"}:
            raise ValueError("concept_grounding.backend must be unavailable or sam3.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("concept_grounding.device must be 'auto', 'cpu', or 'cuda'.")
        for name in ("score_threshold", "mask_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"concept_grounding.{name} must be in [0, 1].")
        if self.min_mask_area <= 0 or self.max_regions_per_concept <= 0:
            raise ValueError("concept_grounding.min_mask_area and max_regions_per_concept must be positive.")


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
        if self.backend not in {"fake", "sam", "sam3", "florence2"}:
            raise ValueError("region_discovery.backend must be fake, sam, sam3, or florence2.")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("region_discovery.score_threshold must be in [0, 1].")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("region_discovery.device must be 'auto', 'cpu', or 'cuda'.")
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
    #: Descarta propostas de tile cortadas por uma borda interna do tile. Sem isso,
    #: superfícies grandes (céu, asfalto) viram faixas retas que o merge por
    #: containment mútua não junta à passada global (visto em 2026-09-15 no
    #: outdoor do corridor-02). Objetos menores que a sobreposição continuam
    #: inteiros no tile vizinho; os maiores ficam com a passada global.
    discard_tile_border_truncations: bool = False

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

    # Recusa qualquer coisa que não seja geometria já construída. Existe porque
    # uma config relida de disco chegava aqui com dicts no lugar das geometrias,
    # passava calada e só quebrava quando o pipeline tentava rasterizar.
    def __post_init__(self) -> None:
        """Rejeita áreas que não sejam ``ImageAreaGeometry`` ou ``None``."""
        for name in ("valid_area", "ego_vehicle"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ImageAreaGeometry):
                raise TypeError(
                    f"image_area.{name} must be an ImageAreaGeometry or None, got {type(value).__name__}."
                )

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
    persist_pixel_aligned_features: bool = False
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

    # Lista os slots que o estágio de evidência efetivamente produz. Existe para
    # que estágios que consomem slots sejam validados contra o que existe, em
    # vez de pedirem um slot desligado e receberem sempre "indisponível".
    def enabled_slots(self) -> frozenset[EvidenceSlot]:
        """Retorna os slots de evidência habilitados nesta configuração."""
        flags = {
            EvidenceSlot.FOREGROUND_DENSE: self.foreground_enabled,
            EvidenceSlot.MASKED_SUBJECT: self.masked_subject_enabled,
            EvidenceSlot.TIGHT_CROP: self.tight_crop_enabled,
            EvidenceSlot.CONTEXTUAL_CROP: self.contextual_crop_enabled,
            EvidenceSlot.SCENE_CONDITIONED: self.scene_conditioned_enabled,
        }
        return frozenset(slot for slot, enabled in flags.items() if enabled)


# Configuração do estágio de suporte de hipótese por alinhamento (#214).
# Existe porque este estágio tem um input de modelo que é texto — o template
# que envolve cada conceito antes de codificá-lo — e a lição de método deste
# módulo é que **prompt é parte versionada do modelo**: uma frase diferente
# produz vetores diferentes e, portanto, um sinal diferente. ``template_version``
# entra no fingerprint junto com o template, de modo que uma mudança invalida o
# cache dependente em vez de se misturar silenciosamente a resultados antigos.
@dataclass(frozen=True)
class HypothesisSupportConfig:
    """Como o alinhamento language-aligned arbitra entre hipóteses de identidade.

    Argumentos:
        enabled: se o estágio roda. Desligado, nenhuma claim recebe sinal, e a
            calibração volta a não ter suporte visual independente.
        source: identidade do produtor do sinal, gravada em cada
            :class:`HypothesisSupportSignal` e nos reports.
        slots: quais slots de evidência são comparados, na ordem canônica. Todo
            slot listado precisa estar habilitado em ``multi_context`` — um slot
            que não é produzido geraria sempre um sinal indisponível. Os
            defaults são os slots de linguagem habilitados por default; o perfil
            real acrescenta o crop contextual, que ele habilita.
        indistinguishable_margin: piso abaixo do qual a diferença entre duas
            hipóteses não é reportada como vitória de nenhuma. Medido nos
            frames de referência, a margem mediana entre primária e melhor
            alternativa fica entre 0,019 e 0,027, de modo que um piso de 0,01
            separa "decidiu" de "empatou" sem inventar vencedor.
        prompt_template: como o conceito é escrito antes de ser codificado.
        template_version: versão do template, para provenance e fingerprint.
    """

    enabled: bool = True
    source: str = "clip_alignment"
    slots: tuple[str, ...] = (
        EvidenceSlot.MASKED_SUBJECT.value,
        EvidenceSlot.TIGHT_CROP.value,
    )
    indistinguishable_margin: float = 0.01
    prompt_template: str = "a photo of {concept}"
    template_version: str = "align/v1"

    # Valida o vocabulário de slots, o piso de margem e a presença do
    # placeholder no template: um template sem ``{concept}`` codificaria a
    # mesma frase para toda hipótese e produziria um sinal constante.
    def __post_init__(self) -> None:
        """Rejeita slots desconhecidos, margens inválidas e templates sem placeholder."""
        known = {slot.value for slot in EvidenceSlot}
        unknown = [name for name in self.slots if name not in known]
        if unknown:
            raise ValueError(f"hypothesis_support.slots has unknown evidence slots: {unknown}.")
        if len(set(self.slots)) != len(self.slots):
            raise ValueError("hypothesis_support.slots must not repeat an evidence slot.")
        if self.enabled and not self.slots:
            raise ValueError("hypothesis_support.enabled requires at least one evidence slot.")
        if not 0.0 <= self.indistinguishable_margin <= 1.0:
            raise ValueError("hypothesis_support.indistinguishable_margin must be in [0, 1].")
        if "{concept}" not in self.prompt_template:
            raise ValueError(
                "hypothesis_support.prompt_template must contain '{concept}': a template without the "
                "placeholder would encode the same sentence for every hypothesis."
            )
        if not self.source or not self.template_version:
            raise ValueError("hypothesis_support.source and template_version must not be empty.")


# Configuração do refinamento seletivo dirigido por suporte (#204). Existe
# separada de ``RefinementConfig`` histórica (que vivia em application/) porque
# o refinamento passou a ser um estágio canônico e a sua configuração precisa
# participar do fingerprint como qualquer outro estágio.
@dataclass(frozen=True)
class RefinementConfig:
    """Quando uma região é reinterpretada, e com qual evidência nova.

    Argumentos:
        enabled: se o estágio roda. O refinamento é um passe único: o
            escalonamento é um conjunto fixo de views, então um segundo passe
            repetiria a mesma evidência. Até a #247 existia um
            ``max_iterations`` que, acima de 1, não mudava nada.
        max_refined_regions: teto de regiões reprocessadas no passe. Existe
            como orçamento de latência explícito, e a seleção dentro do teto é
            determinística por prioridade de razão.
        escalation_views: as views usadas no passe de refinamento. Precisa ser
            diferente do conjunto do passe anterior: repetir a mesma chamada
            com a mesma evidência e temperatura zero é pedir de novo esperando
            outra resposta.

            O escalonamento é **region-local por decisão medida**. A primeira
            versão deste default acrescentava ``scene_conditioned`` — o frame
            inteiro — e o resultado foi o vazamento que a #202 e a #212 tinham
            eliminado: em ``corridor-02-008``, 6 das 16 regiões trocaram a sua
            hipótese por ``rows of crops``, que é literalmente a claim
            ``layout`` da cena. Regiões que antes diziam ``wall``,
            ``purple flower`` e ``plain surface`` passaram a repetir o texto da
            cena. Uma propriedade global não pode virar identidade local, e
            entregar o frame inteiro ao prompt de uma região é o caminho mais
            direto para isso acontecer.
        small_region_area_px: abaixo desta área, "região pequena" acompanha
            outra razão. Nunca dispara sozinha: tamanho não é incerteza
            semântica.
        min_mask_fill_ratio: abaixo desta fração do bounding box ocupada pela
            máscara, a evidência de foreground é considerada insuficiente e a
            interpretação pode estar descrevendo o fundo.
    """

    enabled: bool = True
    max_refined_regions: int = 24
    escalation_views: tuple[str, ...] = (
        EvidenceSlot.FOREGROUND_DENSE.value,
        EvidenceSlot.MASKED_SUBJECT.value,
        EvidenceSlot.TIGHT_CROP.value,
    )
    small_region_area_px: int = 1024
    min_mask_fill_ratio: float = 0.15

    # Valida tetos, fração e o vocabulário de views de escalonamento.
    def __post_init__(self) -> None:
        """Rejeita tetos negativos, frações inválidas e views desconhecidas."""
        if self.max_refined_regions <= 0:
            raise ValueError("refinement.max_refined_regions must be positive.")
        if self.small_region_area_px < 0:
            raise ValueError("refinement.small_region_area_px must not be negative.")
        if not 0.0 <= self.min_mask_fill_ratio <= 1.0:
            raise ValueError("refinement.min_mask_fill_ratio must be in [0, 1].")
        known = {slot.value for slot in EvidenceSlot}
        unknown = [name for name in self.escalation_views if name not in known]
        if unknown:
            raise ValueError(f"refinement.escalation_views has unknown evidence slots: {unknown}.")
        if len(set(self.escalation_views)) != len(self.escalation_views):
            raise ValueError("refinement.escalation_views must not repeat an evidence slot.")
        if self.enabled and not any(
            EvidenceSlot(name) in FOREGROUND_SLOTS for name in self.escalation_views
        ):
            raise ValueError(
                "refinement.escalation_views must include at least one foreground slot: refining a "
                "region from context alone would describe its surroundings."
            )


# Configuração da reconciliação contextual intra-frame (#205). Existe para que
# os dois limiares que decidem *proposta* e *corroboração* sejam explícitos e
# ablatáveis: a medição mostrou que a coerência densa sozinha não separa bem o
# caso, então ela entra como corroboração e o seu limiar precisa ser visível.
@dataclass(frozen=True)
class ReconciliationConfig:
    """Como regiões do mesmo frame são reconciliadas sem alterar geometria.

    Argumentos:
        enabled: se o estágio roda.
        adjacency_margin_px: distância máxima entre bounding boxes para que
            duas regiões sejam tratadas como em contato.
        min_group_coherence: coerência densa média a partir da qual um grupo é
            marcado ``supported``. Abaixo dela o grupo continua existindo, como
            ``unresolved``: o contato e o conceito ainda são evidência.
        canonicalize_labels: se a normalização lexical mínima produz um
            conceito canônico ao lado do label cru.
    """

    enabled: bool = True
    adjacency_margin_px: float = 5.0
    min_group_coherence: float = 0.6
    canonicalize_labels: bool = True

    # Valida margem e limiar de coerência.
    def __post_init__(self) -> None:
        """Rejeita margem negativa e limiar de coerência fora de ``[-1, 1]``."""
        if self.adjacency_margin_px < 0.0:
            raise ValueError("reconciliation.adjacency_margin_px must not be negative.")
        if not -1.0 <= self.min_group_coherence <= 1.0:
            raise ValueError("reconciliation.min_group_coherence must be a cosine in [-1, 1].")


# Configuração da inferência de relações semânticas (#206). O campo que mais
# importa aqui é ``max_pairs``: sem um orçamento, este estágio é O(n²) chamadas
# de VLM, e num frame de 40 regiões seriam 780.
@dataclass(frozen=True)
class SemanticRelationConfig:
    """Quantos e quais pares de regiões o reasoner julga por frame.

    Argumentos:
        enabled: se o estágio roda.
        max_pairs: teto de pares consultados por frame.
        max_pairs_per_region: quantas vezes uma mesma região pode aparecer
            entre os pares escolhidos. Medido em ``corridor-02-000``: sem esse
            teto, 13 dos 16 pares do orçamento tinham a **mesma** região de
            2 px² como sujeito, porque contenção satura em 1,00 para toda
            região pequena contida numa grande. O orçamento cobria a vizinhança
            de uma região em vez do frame.
        min_containment: containment mínimo para um par entrar como candidato
            por contenção.
        include_adjacent: se pares apenas encostados também são candidatos.
        require_distinct_concepts: se um par cujos dois lados compartilham o
            conceito reconciliado é descartado. Ligado por default porque
            "parede encosta em parede" é fragmentação, e a reconciliação já a
            descreve melhor do que uma relação descreveria.
    """

    enabled: bool = True
    max_pairs: int = 16
    max_pairs_per_region: int = 2
    min_containment: float = 0.6
    include_adjacent: bool = True
    require_distinct_concepts: bool = True

    # Valida o orçamento de pares e o limiar de contenção.
    def __post_init__(self) -> None:
        """Rejeita orçamento negativo e containment fora de ``[0, 1]``."""
        if self.max_pairs < 0:
            raise ValueError("semantic_relations.max_pairs must not be negative.")
        if self.max_pairs_per_region <= 0:
            raise ValueError("semantic_relations.max_pairs_per_region must be positive.")
        if not 0.0 <= self.min_containment <= 1.0:
            raise ValueError("semantic_relations.min_containment must be in [0, 1].")


# Configuração da política de publicação contextual. Existe para que a decisão
# "isto vale como evidência contextual?" seja versionada no fingerprint e
# ablatável por configuração, como todo estágio contextual deste módulo: uma
# comparação precisa conseguir atribuir o efeito da política a ela sozinha.
@dataclass(frozen=True)
class ContextualPublicationConfig:
    """O que o módulo publica como evidência contextual, e o que fica como contexto.

    Argumentos:
        enabled: se a partição roda. Desligada, toda região continua sendo
            publicada e ``structural_context`` fica vazio — que é exatamente o
            comportamento anterior à política, preservado para ablação.
        extra_structural_head_nouns: núcleos nominais estruturais adicionais,
            além dos que o domínio já conhece. Existe para que uma composição
            possa declarar o vocabulário do seu ambiente sem que o módulo
            precise conhecê-lo de antemão; vazio por default, porque uma lista
            longa aqui viraria a taxonomia fechada que o projeto recusa.
    """

    enabled: bool = True
    extra_structural_head_nouns: tuple[str, ...] = ()

    # Valida que cada núcleo extra é uma palavra utilizável, já que um token
    # vazio ou com espaço nunca casaria e ficaria como configuração morta.
    def __post_init__(self) -> None:
        """Rejeita núcleo estrutural vazio ou composto por mais de uma palavra."""
        for noun in self.extra_structural_head_nouns:
            if not noun.strip() or len(noun.split()) != 1:
                raise ValueError(
                    "contextual_publication.extra_structural_head_nouns must contain "
                    f"single non-empty words, got {noun!r}."
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
    #: SHA-256 do artifact de calibração, calculado uma vez na construção. Faz
    #: parte do fingerprint: uma tabela diferente produz claims diferentes.
    artifact_digest: str | None = field(default=None, init=False)

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
        # O artifact é lido aqui, uma única vez: antes o fingerprint relia o
        # arquivo a cada chamada, e uma config válida na construção levantava
        # FileNotFoundError só quando a chave de cache era calculada.
        if self.artifact_path is not None:
            artifact = Path(self.artifact_path)
            if not artifact.is_file():
                raise ValueError(f"calibration.artifact_path does not exist: {artifact}.")
            object.__setattr__(self, "artifact_digest", hashlib.sha256(artifact.read_bytes()).hexdigest())


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
    prompt_version: str = "v8"
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
    #: Se o conceito afirmado pela observação anterior para a mesma área da
    #: imagem acompanha a região no prompt. É o terceiro canal ablatável, ao
    #: lado de ``scene_context_mode`` (textual, dentro do frame) e
    #: ``region_views`` (visual): este é o canal **temporal**. Vive aqui, e não
    #: em uma sub-config própria, porque ``fingerprint_of`` carimba cada claim
    #: de região com o fingerprint **desta** sub-config — um switch em outro
    #: lugar deixaria dois braços de experimento indistinguíveis no artifact
    #: por claim. O default é ``disabled``: sem ele, o request é idêntico ao
    #: histórico.
    temporal_prior_mode: str = TemporalPriorMode.DISABLED.value
    #: Sobreposição mínima de caixa entre a região e a região anterior para o
    #: casamento valer. Não é um limiar de similaridade densa: a #203 mediu o
    #: cosseno DINOv2 em 0,660 de acurácia balanceada para separar "mesmo
    #: label", e por isso ele entra só como desempate entre candidatos que já se
    #: sobrepõem geometricamente.
    temporal_prior_min_overlap: float = 0.3
    #: Limite de espera por requisição e tentativas extras em falhas transitórias
    #: (timeout, 429, 5xx). Só os backends remotos usam; o Qwen local ignora.
    timeout_s: float = 60.0
    max_retries: int = 3
    #: Orçamento de raciocínio interno do Gemini Robotics ER. ``0`` desliga: medido
    #: em 2026-09-15 no prompt de cena, 2,4 s contra 22,6 s com o default do modelo,
    #: e a resposta segue no mesmo contract. Muda a saída, então entra no fingerprint.
    thinking_budget: int = 0
    #: Backend para o qual `FallbackMultimodalReasoningAdapter` troca, de forma
    #: permanente pelo resto do run, quando o backend primário esgota os
    #: próprios retries (#292: runs de dias contra um backend remoto de cota
    #: limitada não podem travar quando a cota acaba). Mesmo padrão de
    #: `FeatureExtractionConfig.fallback_backend`.
    fallback_backend: str | None = None
    #: Checkpoint do backend de fallback; obrigatório quando `fallback_backend`
    #: está setado, pelo mesmo motivo de `checkpoint` para o backend primário.
    fallback_checkpoint: str | None = None
    #: `region_views` usado quando o fallback está ativo. Quando omitido e o
    #: fallback é o Qwen local, cai para um teto de 2 views em vez do
    #: `region_views` do primário: o Qwen2.5-VL-3B, ao contrário de um modelo
    #: remoto maior, mede pior (rótulo errado, "colapso") com os até 4 views
    #: simultâneas usadas pela referência de qualidade — relatado
    #: empiricamente pelo usuário para este papel específico.
    fallback_region_views: tuple[str, ...] | None = None

    # Garante que a versão do prompt está definida, já que ela identifica
    # qual template estruturado o backend deve usar. ``v8`` des-arredonda os dois
    # ``confidence`` de exemplo que ainda ofereciam um valor plausível para o
    # modelo copiar: o do prompt de **cena** (``0.9`` → ``0.63``) e o da
    # ``alternatives`` no prompt de região (``0.2`` → ``0.18``). A mesma correção
    # já tinha sido medida no exemplo primário do prompt de região (``v4``,
    # 0,71) e aplicada ao de relação (``v7``, 0,42); estes dois escaparam — o de
    # cena por ser consultado uma vez por frame, e o de alternativa por não
    # aparecer em nenhuma distribuição que o diagnóstico resume. Fora esses dois
    # números, os três prompts são byte-idênticos aos do ``v7``.
    #
    # ``v10``/``v11`` (2026-09-17) reduzem o schema do prompt de região de 8
    # campos para 2, mas só para o backend Qwen local. Generalizado como
    # ``v9``, o schema reduzido fez o Gemini ecoar os exemplos do prompt
    # (``"pallet"`` em vez de ``"wooden pallet"``) e perder confidence e
    # alternatives. O Grounding-DINO não localiza ``"pallet"`` no palete do
    # corridor-02, então a região perdeu a geometria no mapa publicado. A
    # medição está em ``reasoning_prompts._MINIMAL_SCHEMA_BODY``. ``v9`` continua reservado ao
    # schema de 8 campos + prior temporal (ver ``temporal_prior_mode``); só
    # ``execution_profile._REAL_MULTIMODAL_REASONING`` (a config de referência
    # do Qwen) seleciona ``v10``/``v11`` por default. ``v7`` acrescentou o prompt
    # de **relação** (#206); os prompts de cena e de região são idênticos aos do
    # ``v6``, byte a byte, de modo que labels e claims continuam comparáveis
    # entre os dois — o que muda é que a versão passa a identificar três
    # prompts, e não dois. ``v6`` mudou o contract de cena para ambiental e as
    # views que acompanham a região. ``v4`` troca o exemplo de
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
        if self.fallback_backend not in {None, "qwen_vl", "gemini_robotics_er"}:
            raise ValueError(
                "multimodal_reasoning.fallback_backend must be None, 'qwen_vl', or 'gemini_robotics_er'."
            )
        if self.fallback_backend == self.backend:
            raise ValueError("multimodal_reasoning.fallback_backend must differ from backend.")
        if self.fallback_backend is not None and not self.fallback_checkpoint:
            raise ValueError("multimodal_reasoning.fallback_backend requires fallback_checkpoint.")
        if self.fallback_region_views is not None:
            if self.fallback_backend is None:
                raise ValueError("multimodal_reasoning.fallback_region_views requires fallback_backend.")
            unknown_fallback = [name for name in self.fallback_region_views if name not in known]
            if unknown_fallback:
                raise ValueError(
                    "multimodal_reasoning.fallback_region_views has unknown evidence slots: "
                    f"{unknown_fallback}."
                )
            if len(set(self.fallback_region_views)) != len(self.fallback_region_views):
                raise ValueError(
                    "multimodal_reasoning.fallback_region_views must not repeat an evidence slot."
                )
            if not any(EvidenceSlot(name) in FOREGROUND_SLOTS for name in self.fallback_region_views):
                raise ValueError(
                    "multimodal_reasoning.fallback_region_views must include at least one foreground slot."
                )
        if self.scene_context_mode not in {mode.value for mode in SceneContextMode}:
            raise ValueError(
                "multimodal_reasoning.scene_context_mode must be "
                f"{sorted(mode.value for mode in SceneContextMode)}."
            )
        if self.temporal_prior_mode not in {mode.value for mode in TemporalPriorMode}:
            raise ValueError(
                "multimodal_reasoning.temporal_prior_mode must be "
                f"{sorted(mode.value for mode in TemporalPriorMode)}."
            )
        if not 0.0 <= self.temporal_prior_min_overlap <= 1.0:
            raise ValueError(
                "multimodal_reasoning.temporal_prior_min_overlap must be within [0, 1]."
            )
        if self.timeout_s <= 0.0:
            raise ValueError("multimodal_reasoning.timeout_s must be positive.")
        if self.max_retries < 0:
            raise ValueError("multimodal_reasoning.max_retries must not be negative.")
        if self.thinking_budget < 0:
            raise ValueError("multimodal_reasoning.thinking_budget must not be negative.")


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
    hypothesis_support: HypothesisSupportConfig = field(default_factory=HypothesisSupportConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    semantic_relations: SemanticRelationConfig = field(default_factory=SemanticRelationConfig)
    contextual_publication: ContextualPublicationConfig = field(
        default_factory=ContextualPublicationConfig
    )
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    image_area: ImageAreaConfig = field(default_factory=ImageAreaConfig)
    proposal_filter: ProposalFilterConfig = field(default_factory=ProposalFilterConfig)
    semantic_grounding: SemanticGroundingConfig = field(default_factory=SemanticGroundingConfig)
    scene_concept_discovery: SceneConceptDiscoveryConfig = field(
        default_factory=SceneConceptDiscoveryConfig
    )
    concept_grounding: ConceptGroundingConfig = field(default_factory=ConceptGroundingConfig)

    # Valida invariantes que dependem de mais de um campo ao mesmo tempo
    # (o que os ``__post_init__`` das sub-configs não conseguem verificar
    # sozinhos), como a incompatibilidade entre o profile REDUCED_COST e
    # tiling multi-scale (#181).
    def __post_init__(self) -> None:
        """Valida invariantes entre sub-configs."""
        if self.gpu_memory_budget_gb <= 0:
            raise ValueError("gpu_memory_budget_gb must be positive.")
        enabled_slots = self.multi_context.enabled_slots()
        for field_name, requested, active in (
            ("hypothesis_support.slots", self.hypothesis_support.slots, self.hypothesis_support.enabled),
            ("refinement.escalation_views", self.refinement.escalation_views, self.refinement.enabled),
        ):
            disabled = [name for name in requested if EvidenceSlot(name) not in enabled_slots]
            if active and disabled:
                raise ValueError(
                    f"{field_name} requests evidence slots disabled in multi_context: {disabled}. "
                    "Enable them in multi_context or remove them from the request."
                )
        if (
            self.quality_profile is QualityProfile.REDUCED_COST
            and self.tiling.multi_scale_enabled
        ):
            raise ValueError(
                "Incompatible configuration: 'reduced_cost' quality profile does not support "
                "multi-scale tiling (see issue #181)."
            )
        if self.concept_grounding.backend != "unavailable" and not self.scene_concept_discovery.enabled:
            raise ValueError(
                "concept_grounding requires scene_concept_discovery.enabled: without concepts there is "
                "nothing to ground."
            )

    # Serializa a configuração inteira (incluindo sub-configs aninhadas) em
    # um dict simples, usado tanto para persistência quanto para o cálculo
    # de ``fingerprint``.
    def to_dict(self) -> dict[str, Any]:
        """Serializa a configuração inteira em tipos JSON simples."""
        payload = asdict(self)
        payload["quality_profile"] = self.quality_profile.value
        return payload

    # Reconstrói um ``ModuleConfig`` a partir de um dict (o inverso de
    # ``to_dict``), usado ao carregar configuração persistida ou vinda de
    # um arquivo de config da aplicação.
    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ModuleConfig:
        """Reconstrói a configuração, inclusive geometrias aninhadas e tuplas.

        Argumentos:
            payload: dict produzido por ``to_dict``, direto ou relido de JSON.
        Retorna:
            a configuração igual à que foi serializada.
        Levanta:
            ValueError: se o artifact de calibração mudou desde a serialização.
        """
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
            ("hypothesis_support", HypothesisSupportConfig),
            ("refinement", RefinementConfig),
            ("reconciliation", ReconciliationConfig),
            ("semantic_relations", SemanticRelationConfig),
            ("contextual_publication", ContextualPublicationConfig),
            ("calibration", CalibrationConfig),
            ("image_area", ImageAreaConfig),
            ("proposal_filter", ProposalFilterConfig),
            ("semantic_grounding", SemanticGroundingConfig),
            ("scene_concept_discovery", SceneConceptDiscoveryConfig),
            ("concept_grounding", ConceptGroundingConfig),
        ):
            if key in payload and isinstance(payload[key], dict):
                if key == "image_area":
                    payload[key] = _image_area_from_dict(payload[key])
                elif key == "calibration":
                    payload[key] = _calibration_from_dict(payload[key])
                else:
                    payload[key] = config_type(**_as_tuples(payload[key]))
        return ModuleConfig(**payload)

    # Calcula uma identidade estável da configuração inteira, usada como
    # chave de cache (#170) para evitar reprocessar a mesma imagem com a
    # mesma configuração.
    def fingerprint(self) -> str:
        """Um hash estável da configuração completa, usado para caching (#170)."""
        payload = self.to_dict()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# Converte listas em tuplas, recursivamente. Existe porque toda sequência das
# configs é tupla, e uma config relida de JSON chegava com listas: os campos
# eram aceitos, mas a config deixava de ser igual à que foi escrita.
def _as_tuples(value: Any) -> Any:
    """Retorna ``value`` com toda lista, em qualquer profundidade, trocada por tupla."""
    if isinstance(value, dict):
        return {key: _as_tuples(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_as_tuples(item) for item in value)
    return value


# Reconstrói uma geometria de área serializada por ``asdict``. Isolada porque
# as duas áreas usam o mesmo formato.
def _geometry_from_dict(payload: dict[str, Any] | None) -> ImageAreaGeometry | None:
    """Converte o dict de uma geometria de área, ou ``None``."""
    if payload is None:
        return None
    circle = payload.get("circle")
    return ImageAreaGeometry(
        circle=None if circle is None else CircleArea(**circle),
        polygons=_as_tuples(payload.get("polygons", ())),
    )


# Reconstrói a configuração de áreas com geometrias de verdade, e não dicts.
def _image_area_from_dict(payload: dict[str, Any]) -> ImageAreaConfig:
    """Converte o dict de ``ImageAreaConfig`` produzido por ``to_dict``."""
    return ImageAreaConfig(
        valid_area=_geometry_from_dict(payload.get("valid_area")),
        ego_vehicle=_geometry_from_dict(payload.get("ego_vehicle")),
    )


# Reconstrói a configuração de calibração conferindo o digest gravado. Existe
# porque o digest é derivado do arquivo: se o artifact mudou desde que a
# config foi escrita, reler a config reproduziria outra calibração em silêncio.
def _calibration_from_dict(payload: dict[str, Any]) -> CalibrationConfig:
    """Converte o dict de ``CalibrationConfig`` e valida o digest do artifact."""
    fields = dict(payload)
    recorded_digest = fields.pop("artifact_digest", None)
    config = CalibrationConfig(**fields)
    if recorded_digest is not None and recorded_digest != config.artifact_digest:
        raise ValueError(
            f"calibration artifact {config.artifact_path} changed since this configuration was written."
        )
    return config
