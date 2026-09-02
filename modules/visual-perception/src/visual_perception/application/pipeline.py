"""Canonical visual perception pipeline.

Issue: #169. This is the module's single primary application entry point:
region discovery -> multi-scale merge -> visual features -> language
embeddings -> scene context -> region semantics -> relations -> audit.

Isolated region-level interpretation failures (#165) do not abort the run:
the affected regions are kept with their geometry and whatever claims other
stages already attached, and the failures are reported alongside the
canonical output rather than raised.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from visual_perception.application.language_embedding import encode_regions
from visual_perception.application.pooling import HIGH_RESOLUTION, pool_regions
from visual_perception.application.quality_audit import audit_observation
from visual_perception.application.region_merge import merge_regions
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.application.relation_generation import generate_relations
from visual_perception.application.scene_context import analyze_scene
from visual_perception.application.tiling import build_tiles, remap_to_global
from visual_perception.config import ModuleConfig
from visual_perception.domain.audit import AuditResult
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import ObservedRegion, RegionProposal
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.ports.feature_extraction import DenseFeatureExtractor
from visual_perception.ports.language_embedding import LanguageAlignedEncoder
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner
from visual_perception.ports.region_discovery import RegionDiscoverer


@dataclass(frozen=True)
class PerceptionPorts:
    """The replaceable backends the canonical pipeline is composed against."""

    region_discoverer: RegionDiscoverer
    feature_extractor: DenseFeatureExtractor
    language_encoder: LanguageAlignedEncoder
    multimodal_reasoner: MultimodalReasoner


@dataclass(frozen=True)
class PipelineResult:
    """The canonical output plus everything needed to audit the run."""

    observation: VisualObservation
    region_interpretation_failures: tuple[RegionInterpretationFailure, ...]
    audit: AuditResult


def run_canonical_pipeline(
    image: ImageObservation,
    payload: ImagePayload,
    config: ModuleConfig,
    ports: PerceptionPorts,
) -> PipelineResult:
    """Transform one validated image observation into a canonical visual observation."""
    proposals = _discover_regions(payload, config, ports.region_discoverer)
    regions = merge_regions(image.observation_id, proposals, config.merge)

    if regions:
        feature_map = ports.feature_extractor.extract(payload, config.feature_extraction)
        visual_embeddings = pool_regions(regions, feature_map, method=HIGH_RESOLUTION)
        visual_refs = {embedding.region_id: embedding.embedding_id for embedding in visual_embeddings}
        regions = _attach_visual_refs(regions, visual_refs)
        language_embeddings = encode_regions(
            regions, payload, ports.language_encoder, config.language_embedding
        )
        language_refs = {embedding.region_id: embedding.embedding_id for embedding in language_embeddings}
        regions = _attach_language_refs(regions, language_refs)

    scene_context = analyze_scene(payload, ports.multimodal_reasoner, config.multimodal_reasoning)
    regions, failures = interpret_regions(
        regions, payload, scene_context, ports.multimodal_reasoner, config.multimodal_reasoning
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
    return PipelineResult(observation=observation, region_interpretation_failures=failures, audit=audit)


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


def _attach_visual_refs(
    regions: tuple[ObservedRegion, ...], refs_by_region_id: dict[str, str]
) -> tuple[ObservedRegion, ...]:
    return tuple(
        dataclasses.replace(region, visual_embedding_ref=refs_by_region_id.get(region.region_id))
        for region in regions
    )


def _attach_language_refs(
    regions: tuple[ObservedRegion, ...], refs_by_region_id: dict[str, str]
) -> tuple[ObservedRegion, ...]:
    return tuple(
        dataclasses.replace(region, language_embedding_ref=refs_by_region_id.get(region.region_id))
        for region in regions
    )
