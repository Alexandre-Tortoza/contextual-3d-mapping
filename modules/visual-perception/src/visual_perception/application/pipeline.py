"""Pipeline canônico de percepção visual.

Issues: #169 (pipeline canônico), #194 (evidência multi-contexto),
#196 (calibração e abstenção).

Este é o único ponto de entrada de aplicação primário do módulo: region
discovery -> merge multi-scale -> evidência multi-contexto (features
visuais de foreground + crops alinhados a linguagem) -> scene context ->
semântica de região -> calibração -> relações -> audit.

Falhas isoladas não abortam a execução, e cada tipo é reportado
separadamente: interpretação de região (#165), extração de um slot de
evidência (#194) e calibração de uma claim (#196). As regiões afetadas são
mantidas com sua geometria e com tudo que os outros stages já anexaram.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception.application.multi_context import (
    EvidenceExtractionFailure,
    extract_region_evidence,
)
from visual_perception.application.quality_audit import audit_observation
from visual_perception.application.region_merge import merge_regions
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.application.relation_generation import generate_relations
from visual_perception.application.scene_context import analyze_scene
from visual_perception.application.semantic_calibration import (
    CalibrationFailure,
    build_calibrator,
    calibrate_observation_claims,
)
from visual_perception.application.tiling import build_tiles, remap_to_global
from visual_perception.config import ModuleConfig
from visual_perception.domain.audit import AuditResult
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import RegionProposal
from visual_perception.domain.visual_observation import VisualObservation
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


# Ponto de entrada principal do módulo: conduz uma observação de imagem
# validada por todos os stages do pipeline canônico (discovery, merge,
# features, embeddings, scene context, semântica, relações, audit) até a
# VisualObservation final. É o que mapping-runtime e benchmarks chamam
# para processar um frame RGB.
def run_canonical_pipeline(
    image: ImageObservation,
    payload: ImagePayload,
    config: ModuleConfig,
    ports: PerceptionPorts,
) -> PipelineResult:
    """Transforma uma observação de imagem validada em uma observação visual canônica."""
    proposals = _discover_regions(payload, config, ports.region_discoverer)
    regions = merge_regions(image.observation_id, proposals, config.merge)

    evidence_failures: tuple[EvidenceExtractionFailure, ...] = ()
    if regions:
        feature_map = ports.feature_extractor.extract(payload, config.feature_extraction)
        evidence = extract_region_evidence(
            regions, payload, config, ports.language_encoder, feature_map=feature_map
        )
        regions = evidence.regions
        evidence_failures = evidence.failures

    scene_context = analyze_scene(payload, ports.multimodal_reasoner, config.multimodal_reasoning)
    regions, failures = interpret_regions(
        regions, payload, scene_context, ports.multimodal_reasoner, config.multimodal_reasoning
    )

    calibrator = build_calibrator(config.calibration)
    regions, scene_context, calibration_failures = calibrate_observation_claims(
        regions, scene_context, calibrator, config.calibration
    )
    relations = generate_relations(regions, config.merge)

    observation = VisualObservation(
        source=image.source,
        image_width=image.width,
        image_height=image.height,
        scene_context=scene_context,
        regions=regions,
        relations=relations,
    )
    audit = audit_observation(observation)
    return PipelineResult(
        observation=observation,
        region_interpretation_failures=failures,
        audit=audit,
        evidence_failures=evidence_failures,
        calibration_failures=calibration_failures,
    )


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
