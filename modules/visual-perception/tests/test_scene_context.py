"""Scene-level contextual analysis stage tests (#164)."""

from __future__ import annotations

import pytest

from fixtures import payload_with_blobs
from visual_perception.application.scene_context import analyze_scene
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


def test_scene_output_follows_canonical_claim_contracts() -> None:
    scene = analyze_scene(payload_with_blobs(), FakeMultimodalReasoner(), MultimodalReasoningConfig())
    kinds = {claim.kind.value for claim in scene.claims}
    assert "scene_type" in kinds
    assert "scene_description" in kinds


def test_malformed_response_is_rejected() -> None:
    reasoner = FakeMultimodalReasoner(scene_response_fn=lambda image: {"description": "missing scene_type"})
    with pytest.raises(ValueError):
        analyze_scene(payload_with_blobs(), reasoner, MultimodalReasoningConfig())


def test_malformed_attribute_list_is_rejected() -> None:
    reasoner = FakeMultimodalReasoner(
        scene_response_fn=lambda image: {
            "scene_type": "corridor",
            "description": "a corridor",
            "attributes": "not-a-list",
        }
    )
    with pytest.raises(ValueError):
        analyze_scene(payload_with_blobs(), reasoner, MultimodalReasoningConfig())
