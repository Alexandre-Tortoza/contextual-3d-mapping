"""Shared fake-backed ``PerceptionPorts`` fixture, kept separate from ``fixtures.py``
so pure image/observation builders don't force every test to depend on the canonical
pipeline and every fake adapter (#173).
"""

from __future__ import annotations

from visual_perception.application.pipeline import PerceptionPorts
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner
from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer


def default_ports() -> PerceptionPorts:
    return PerceptionPorts(
        region_discoverer=FakeRegionDiscoverer(),
        feature_extractor=FakeDenseFeatureExtractor(),
        language_encoder=FakeLanguageAlignedEncoder(),
        multimodal_reasoner=FakeMultimodalReasoner(),
    )
