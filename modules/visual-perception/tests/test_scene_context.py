"""Testes do estágio de análise contextual em nível de cena (#164)."""

from __future__ import annotations

import pytest

from fixtures import payload_with_blobs
from visual_perception.application.scene_context import analyze_scene
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Verifica que a saída de analyze_scene sempre inclui os claims canônicos
# obrigatórios de cena: "scene_type" e "scene_description".
def test_scene_output_follows_canonical_claim_contracts() -> None:
    scene = analyze_scene(payload_with_blobs(), FakeMultimodalReasoner(), MultimodalReasoningConfig())
    kinds = {claim.kind.value for claim in scene.claims}
    assert "scene_type" in kinds
    assert "scene_description" in kinds


# Uma resposta de reasoner que omite o campo obrigatório scene_type deve ser
# rejeitada, em vez de produzir uma cena parcialmente preenchida.
def test_malformed_response_is_rejected() -> None:
    reasoner = FakeMultimodalReasoner(scene_response_fn=lambda image: {"description": "missing scene_type"})
    with pytest.raises(ValueError):
        analyze_scene(payload_with_blobs(), reasoner, MultimodalReasoningConfig())


# O campo "attributes" precisa ser uma lista; uma resposta que o envia como outro
# tipo (string, aqui) deve falhar a validação em vez de ser aceita silenciosamente.
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
