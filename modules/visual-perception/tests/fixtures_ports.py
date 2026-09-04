"""Fixture compartilhada de ``PerceptionPorts`` baseada em fakes, mantida separada de
``fixtures.py`` para que os builders puros de imagem/observação não obriguem todo teste
a depender do pipeline canônico e de todo adapter fake (#173).
"""

from __future__ import annotations

from visual_perception.application.pipeline import PerceptionPorts
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner
from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer


# Monta um PerceptionPorts completo usando só fakes GPU-free, para que qualquer teste
# do pipeline canônico rode sem precisar de um backend real; usada por todos os testes
# que exercitam o pipeline de ponta a ponta.
def default_ports() -> PerceptionPorts:
    return PerceptionPorts(
        region_discoverer=FakeRegionDiscoverer(),
        feature_extractor=FakeDenseFeatureExtractor(),
        language_encoder=FakeLanguageAlignedEncoder(),
        multimodal_reasoner=FakeMultimodalReasoner(),
    )
