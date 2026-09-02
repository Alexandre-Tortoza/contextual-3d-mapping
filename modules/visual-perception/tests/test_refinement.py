"""Uncertainty-driven selective refinement tests (#183)."""

from __future__ import annotations

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.application.refinement import (
    RefinementConfig,
    refine_observation,
    select_refinement_targets,
)
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


def test_refinement_targets_low_confidence_regions() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    targets = select_refinement_targets(result.observation, RefinementConfig(low_confidence_threshold=1.0))
    assert set(targets) == {region.region_id for region in result.observation.regions}


def test_refinement_loop_terminates_and_preserves_previous_evidence() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())
    original_claim_count = len(result.observation.regions[0].claims)

    refined, history = refine_observation(
        result.observation,
        payload,
        FakeMultimodalReasoner(),
        MultimodalReasoningConfig(),
        RefinementConfig(low_confidence_threshold=1.0, max_iterations=2),
    )

    assert len(history) <= 2
    assert len(refined.regions[0].claims) >= original_claim_count


def test_no_targets_means_no_refinement_needed() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    refined, history = refine_observation(
        result.observation,
        payload,
        FakeMultimodalReasoner(),
        MultimodalReasoningConfig(),
        RefinementConfig(low_confidence_threshold=0.0, small_region_area_px=0, max_iterations=2),
    )
    assert history == ()
    assert refined == result.observation
