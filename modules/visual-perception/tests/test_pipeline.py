"""End-to-end canonical pipeline tests (#169). No GPU, no model downloads."""

from __future__ import annotations

from fixtures import blank_payload, default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline


def test_canonical_pipeline_with_no_regions_passes_audit() -> None:
    result = run_canonical_pipeline(image_observation(), blank_payload(), default_config(), default_ports())

    assert result.observation.regions == ()
    assert result.audit.passed
    assert result.region_interpretation_failures == ()


def test_canonical_pipeline_with_one_region_produces_claims_and_embeddings() -> None:
    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert len(result.observation.regions) == 1
    region = result.observation.regions[0]
    assert region.claims
    assert region.visual_embedding_ref is not None
    assert region.language_embedding_ref is not None
    assert result.audit.passed


def test_canonical_pipeline_with_multiple_regions_generates_relations() -> None:
    payload = payload_with_blobs(
        blobs=(
            (2, 2, 8, 8, (200, 30, 30)),
            (20, 20, 28, 28, (30, 200, 30)),
        )
    )
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert len(result.observation.regions) == 2
    assert result.audit.passed


def test_canonical_pipeline_output_passes_the_quality_auditor() -> None:
    payload = payload_with_blobs()
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    assert result.audit.errors == ()
