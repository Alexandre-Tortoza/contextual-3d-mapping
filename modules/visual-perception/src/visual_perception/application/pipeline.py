"""Pipeline canônico de percepção visual.

Issues: #169 (pipeline canônico), #194 (evidência multi-contexto),
#196 (calibração e abstenção), #202/#203 (raciocínio de região mask-aware
com contexto de cena estruturado), #214 (suporte independente), #204 (refinamento
dirigido por razão), #217 (embeddings expostos), #205 (reconciliação intra-frame), #206 (relações
semânticas), #207 (integração canônica).

Este é o único ponto de entrada de aplicação primário do módulo. A ordem
canônica é determinística e tem uma linha divisória explícita:

```text
GEOMETRIA (decide o que existe)          SEMÂNTICA (decide o que significa)
---------------------------------        ----------------------------------
tiling                                   scene context
region discovery                         region semantics
filtragem de proposals                   hypothesis support signals
cross-scale merge  <-- geometria         calibração
dense features         congelada         refinamento seletivo
evidência multi-contexto                 relações geométricas
                                         reconciliação intra-frame
                                         relações semânticas
                                         audit final
```

Depois do merge, **nenhum estágio altera mask, box ou identidade de região**.
Os estágios semânticos só acrescentam claims, sinais, grupos e relações — e a
auditoria final roda depois de todos eles, porque um audit tirado no meio
descreveria um estado que ninguém consome.

Falhas isoladas não abortam a execução, e cada tipo é reportado
separadamente: interpretação de região (#165), extração de um slot de
evidência (#194), calibração de uma claim (#196), sinal de suporte e
inferência de relação (#214/#206). As regiões afetadas são mantidas com sua
geometria e com tudo que os outros estágios já anexaram.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

from visual_perception.application.hypothesis_support import (
    HypothesisSupportResult,
    SignalExtractionFailure,
    attach_hypothesis_signals,
)
from visual_perception.application.multi_context import (
    EvidenceExtractionFailure,
    EvidenceSlotMetrics,
    MultiContextResult,
    extract_region_evidence,
)
from visual_perception.application.proposal_filtering import filter_proposals
from visual_perception.application.quality_audit import audit_observation
from visual_perception.application.reconciliation import ReconciliationRecord, reconcile_observation
from visual_perception.application.refinement import RefinementStep, refine_observation
from visual_perception.application.region_merge import merge_regions
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.application.relation_generation import generate_relations
from visual_perception.application.scene_context import analyze_scene
from visual_perception.application.semantic_calibration import (
    CalibrationFailure,
    build_calibrator,
    calibrate_observation_claims,
)
from visual_perception.application.semantic_relations import (
    RelationInferenceFailure,
    infer_semantic_relations,
)
from visual_perception.application.tiling import build_tiles, remap_to_global
from visual_perception.config import ModuleConfig
from visual_perception.domain.audit import AuditResult
from visual_perception.domain.contextual_entities import ContextualEntityHypothesis
from visual_perception.domain.embeddings import (
    EmbeddingModality,
    EmbeddingSpace,
    LanguageEmbedding,
    VisualEmbedding,
)
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_reasoning import RegionView
from visual_perception.domain.regions import ObservedRegion, RegionProposal, RejectedProposal
from visual_perception.domain.visual_observation import SceneContext, VisualObservation
from visual_perception.ports.feature_extraction import DenseFeatureExtractor
from visual_perception.ports.language_embedding import LanguageAlignedEncoder
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner
from visual_perception.ports.region_discovery import RegionDiscoverer


# Agrupa os backends substituíveis contra os quais o pipeline canônico é
# composto, permitindo trocar cada backend (ex: por um fake em teste, ou
# por outro modelo em benchmark) sem alterar run_canonical_pipeline.
@dataclass(frozen=True)
class PerceptionPorts:
    """Os backends substituíveis contra os quais o pipeline canônico é composto."""

    region_discoverer: RegionDiscoverer
    feature_extractor: DenseFeatureExtractor
    language_encoder: LanguageAlignedEncoder
    multimodal_reasoner: MultimodalReasoner


# Agrupa a saída canônica do pipeline com tudo que é necessário para
# auditar a execução, retornada por run_canonical_pipeline aos
# consumidores (ex: integração com mapping-runtime, benchmarks).
@dataclass(frozen=True)
class PipelineResult:
    """A saída canônica mais tudo que é necessário para auditar a execução."""

    observation: VisualObservation
    region_interpretation_failures: tuple[RegionInterpretationFailure, ...]
    audit: AuditResult
    evidence_failures: tuple[EvidenceExtractionFailure, ...] = ()
    calibration_failures: tuple[CalibrationFailure, ...] = ()
    evidence_metrics: tuple[EvidenceSlotMetrics, ...] = ()
    #: Preenchido apenas quando o extractor denso caiu para um backend de
    #: fallback; ``None`` significa que o backend configurado executou.
    feature_fallback_reason: str | None = None
    #: As proposals cruas de discovery, antes do merge geométrico. Existem
    #: aqui porque discovery não é determinística: recomputá-las depois
    #: produziria proposals diferentes das que geraram estas regiões, e a
    #: ligação com ``ObservedRegion.contributing_proposal_ids`` deixaria de
    #: ser verdadeira. Consumidas por ferramentas de diagnóstico que
    #: precisam comparar o estágio pré-merge com o pós-merge.
    #:
    #: WARNING: carrega as masks em resolução plena de todas as proposals.
    #: Não retenha um ``PipelineResult`` entre frames.
    proposals: tuple[RegionProposal, ...] = ()
    #: As proposals descartadas pela filtragem, com motivo e medida. Existem
    #: aqui para que o descarte seja auditável: ``proposals`` e estas somam
    #: exatamente o que discovery produziu, e o diagnóstico as usa para afirmar
    #: que nenhuma proposta sobrou fora da área válida ou sobre o rig.
    rejected_proposals: tuple[RejectedProposal, ...] = ()
    #: As áreas declaradas deste frame, já rasterizadas. ``None`` quando a
    #: composição não declarou nenhuma.
    area_masks: ImageAreaMasks | None = None
    #: Os vetores por região produzidos pelo estágio de evidência. Até a #217
    #: eles eram calculados — 121 chamadas de encoder por frame na configuração
    #: real — e descartados aqui dentro, de modo que o consumidor recebia só a
    #: string de ``artifact_ref`` e nenhum vetor. Expor os dois é o que permite
    #: persistí-los por referência e o que alimenta o suporte de hipótese.
    visual_embeddings: tuple[VisualEmbedding, ...] = ()
    language_embeddings: tuple[LanguageEmbedding, ...] = ()
    #: Falhas isoladas ao produzir sinais de suporte de hipótese (#214).
    signal_failures: tuple[SignalExtractionFailure, ...] = ()
    #: Histórico append-only do refinamento seletivo (#204): alvo, razão,
    #: evidência anterior, evidência nova, produtor e fingerprint por iteração.
    refinement_history: tuple[RefinementStep, ...] = ()
    #: O que a reconciliação intra-frame concluiu por região (#205).
    reconciliation_records: tuple[ReconciliationRecord, ...] = ()
    #: Falhas isoladas ao inferir uma relação semântica de um par (#206).
    relation_failures: tuple[RelationInferenceFailure, ...] = ()
    #: Chamadas de modelo gastas por estágio, para que o custo de cada
    #: capacidade nova seja atribuível no manifest de validação.
    stage_model_calls: Mapping[str, int] = field(default_factory=dict)

    # Expõe os grupos propostos pela reconciliação sem obrigar o consumidor a
    # navegar até a observação. Existe porque três consumidores (diagnóstico,
    # artifacts, integração downstream) já precisam deles.
    @property
    def entity_hypotheses(self) -> tuple[ContextualEntityHypothesis, ...]:
        """Os grupos de entidade contextual propostos para este frame."""
        return self.observation.entity_hypotheses


# Ponto de entrada principal do módulo: conduz uma observação de imagem
# validada por todos os stages do pipeline canônico até a VisualObservation
# final. É o que mapping-runtime e benchmarks chamam para processar um frame
# RGB.
def run_canonical_pipeline(
    image: ImageObservation,
    payload: ImagePayload,
    config: ModuleConfig,
    ports: PerceptionPorts,
) -> PipelineResult:
    """Transforma uma observação de imagem validada em uma observação visual canônica."""
    area_masks = config.image_area.rasterize(payload.width, payload.height)
    discovered = _discover_regions(payload, config, ports.region_discoverer)
    # A exclusão do rig e da área fora da lente acontece aqui, sobre as
    # máscaras, e nunca pintando os pixels de entrada: a #212 mediu que zerar a
    # faixa do rig antes do SAM corrompia a análise de cena e colapsava 45 de 45
    # regiões em ``curved wall``.
    proposals, rejected_proposals = filter_proposals(
        discovered, area_masks=area_masks, config=config.proposal_filter
    )
    regions = merge_regions(image.observation_id, proposals, config.merge)

    evidence, feature_fallback_reason = _extract_evidence(regions, payload, config, ports)
    regions = evidence.regions
    views = evidence.views

    # A cena é analisada sobre a área válida menos a área do ego: o rig não
    # pode entrar no inventário do ambiente que depois condiciona cada região.
    scene_context = analyze_scene(
        payload, ports.multimodal_reasoner, config.multimodal_reasoning, area_masks=area_masks
    )
    regions, failures = interpret_regions(
        regions, payload, views, scene_context, ports.multimodal_reasoner, config.multimodal_reasoning
    )

    support = attach_hypothesis_signals(
        regions,
        evidence.language_embeddings,
        ports.language_encoder,
        config.hypothesis_support,
        config.language_embedding,
    )
    regions = support.regions

    calibrator = build_calibrator(config.calibration)
    regions, scene_context, calibration_failures = calibrate_observation_claims(
        regions, scene_context, calibrator, config.calibration
    )

    # O refinamento opera sobre uma observação, e não sobre a tupla de regiões,
    # porque as suas razões dependem do estado inteiro do frame. A observação
    # intermediária existe só para isso e nunca é auditada: o audit que vale é
    # o final, depois de todos os estágios que acrescentam claim ou relação.
    staged = _stage_observation(image, scene_context, regions, ())
    staged, refinement_history = refine_observation(
        staged, payload, views, ports.multimodal_reasoner, config.multimodal_reasoning, config.refinement
    )
    regions = staged.regions
    if refinement_history:
        # As hipóteses que o refinamento acrescentou passam pelos mesmos dois
        # crivos que as originais: o canal independente as mede, e a calibração
        # decide. Um segundo passe que pulasse isso entregaria claims novas com
        # menos escrutínio que as antigas, o que é o inverso do que a razão de
        # refiná-las dizia.
        rescored = attach_hypothesis_signals(
            regions,
            evidence.language_embeddings,
            ports.language_encoder,
            config.hypothesis_support,
            config.language_embedding,
        )
        support = dataclasses.replace(
            support,
            failures=support.failures + rescored.failures,
            text_encode_calls=support.text_encode_calls + rescored.text_encode_calls,
            measured_signals=support.measured_signals + rescored.measured_signals,
        )
        regions, scene_context, extra_failures = calibrate_observation_claims(
            rescored.regions, scene_context, calibrator, config.calibration
        )
        calibration_failures = calibration_failures + extra_failures

    geometric_relations = generate_relations(regions, config.merge)

    reconciliation = reconcile_observation(
        regions,
        image.observation_id,
        config.reconciliation,
        visual_embeddings=evidence.visual_embeddings,
        visual_space=_visual_space(evidence.visual_embeddings, config),
    )
    regions = reconciliation.regions

    inferred = infer_semantic_relations(
        regions,
        payload,
        ports.multimodal_reasoner,
        config.semantic_relations,
        config.multimodal_reasoning,
        entities=reconciliation.entities,
        context_expansion=config.multi_context.context_expansion,
    )

    observation = VisualObservation(
        source=image.source,
        image_width=image.width,
        image_height=image.height,
        scene_context=scene_context,
        regions=regions,
        relations=geometric_relations + inferred.relations,
        entity_hypotheses=reconciliation.entities,
    )
    audit = audit_observation(observation)
    return PipelineResult(
        observation=observation,
        region_interpretation_failures=failures,
        audit=audit,
        evidence_failures=evidence.failures,
        calibration_failures=calibration_failures,
        evidence_metrics=evidence.metrics,
        feature_fallback_reason=feature_fallback_reason,
        proposals=proposals,
        rejected_proposals=rejected_proposals,
        area_masks=area_masks,
        visual_embeddings=evidence.visual_embeddings,
        language_embeddings=evidence.language_embeddings,
        signal_failures=support.failures,
        refinement_history=refinement_history,
        reconciliation_records=reconciliation.records,
        relation_failures=inferred.failures,
        stage_model_calls=_stage_model_calls(evidence, support, refinement_history, inferred),
    )


# Produz a evidência multi-contexto quando existem regiões, devolvendo um
# resultado vazio quando não existem. Existe para que o caminho feliz de
# run_canonical_pipeline não carregue um ``if`` de várias linhas com quatro
# variáveis inicializadas antes dele.
def _extract_evidence(
    regions: tuple[ObservedRegion, ...],
    payload: ImagePayload,
    config: ModuleConfig,
    ports: PerceptionPorts,
) -> tuple[MultiContextResult, str | None]:
    """Extrai a evidência de todas as regiões, ou devolve um resultado vazio."""
    if not regions:
        return MultiContextResult(regions=regions, visual_embeddings=(), language_embeddings=(), failures=()), None
    feature_map = ports.feature_extractor.extract(payload, config.feature_extraction)
    # Um fallback de backend denso é uma execução legítima, mas não é a
    # execução pedida. O motivo sobe até o resultado para que o consumidor e o
    # manifest de validação (#190) o registrem, em vez de o run parecer ter
    # usado o backend configurado.
    evidence = extract_region_evidence(
        regions, payload, config, ports.language_encoder, feature_map=feature_map
    )
    return evidence, feature_map.fallback_reason


# Monta a observação intermediária que o refinamento inspeciona. Existe porque
# as razões de refinamento dependem do frame inteiro, e não de uma região
# isolada; a observação final é construída de novo, depois da reconciliação.
def _stage_observation(
    image: ImageObservation,
    scene_context: SceneContext,
    regions: tuple[ObservedRegion, ...],
    entities: tuple[ContextualEntityHypothesis, ...],
) -> VisualObservation:
    """Constrói a observação intermediária usada pelos estágios que veem o frame todo."""
    return VisualObservation(
        source=image.source,
        image_width=image.width,
        image_height=image.height,
        scene_context=scene_context,
        regions=regions,
        relations=(),
        entity_hypotheses=entities,
    )


# Deriva o espaço dos embeddings visuais a partir dos próprios vetores e da
# configuração. Existe para que a coerência de um grupo declare em que espaço
# foi medida; devolve ``None`` sem embeddings, caso em que a reconciliação
# marca os grupos como não corroborados.
def _visual_space(
    embeddings: tuple[VisualEmbedding, ...], config: ModuleConfig
) -> EmbeddingSpace | None:
    """Retorna o :class:`EmbeddingSpace` dos vetores densos, ou ``None`` sem vetores."""
    if not embeddings:
        return None
    return EmbeddingSpace(
        model_id=embeddings[0].model_id,
        checkpoint=config.feature_extraction.checkpoint,
        dimension=embeddings[0].dimension,
        modality=EmbeddingModality.VISUAL_DENSE,
        normalized=embeddings[0].normalized,
    )


# Consolida as chamadas de modelo por estágio. Existe para que o manifest de
# validação possa atribuir custo a cada capacidade nova em vez de reportar um
# total no qual elas ficam invisíveis.
def _stage_model_calls(
    evidence: MultiContextResult,
    support: HypothesisSupportResult,
    refinement_history: tuple[RefinementStep, ...],
    inferred: object,
) -> dict[str, int]:
    """Retorna quantas chamadas de modelo cada estágio gastou neste frame."""
    return {
        "language_aligned_evidence": sum(metric.model_calls for metric in evidence.metrics),
        "hypothesis_support_text": support.text_encode_calls,
        "region_refinement": sum(len(step.target_region_ids) for step in refinement_history),
        "semantic_relations": getattr(inferred, "model_calls", 0),
    }


# Descobre region proposals em nível de tile e as remapeia para
# coordenadas globais da imagem; primeiro estágio de
# run_canonical_pipeline, chamado antes do merge multi-scale.
def _discover_regions(
    payload: ImagePayload, config: ModuleConfig, discoverer: RegionDiscoverer
) -> tuple[RegionProposal, ...]:
    proposals: list[RegionProposal] = []
    for tile in build_tiles(payload, config.tiling):
        for local_proposal in discoverer.discover(tile.payload, config.region_discovery):
            proposals.append(
                remap_to_global(local_proposal, tile, image_width=payload.width, image_height=payload.height)
            )
    return tuple(proposals)


# Mantém a assinatura anterior de views para os consumidores que a importavam
# daqui. Existe apenas como reexport tipado; a construção continua sendo do
# estágio de evidência.
__all__ = ["PerceptionPorts", "PipelineResult", "RegionView", "run_canonical_pipeline"]
