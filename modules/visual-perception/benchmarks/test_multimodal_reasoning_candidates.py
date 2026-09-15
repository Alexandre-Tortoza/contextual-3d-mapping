"""Testes da pontuação dos candidatos de raciocínio multimodal (#248)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.multimodal_reasoning import score_scene_response  # noqa: E402
from visual_perception.infrastructure.adapters.reasoning_prompts import scene_prompt  # noqa: E402

_PRODUCTION_RESPONSE = {
    "scene_type": "corridor",
    "environment": "indoor",
    "layout": "long straight hallway",
    "lighting": "dim overhead light",
    "visibility": "clear to the far end",
    "navigability": "unobstructed",
    "confidence": 0.63,
}

_REJECTED_RESPONSE = {
    "scene_type": "corridor",
    "description": "A long corridor with doors on both sides and a carpet on the floor.",
    "attributes": ["carpeted floor"],
    "hazards": [],
    "confidence": 0.9,
}


# Regressão da #248: o benchmark dava 1,00 ao schema que o pipeline recusa e
# tirava 0,35 do schema de produção. A validade agora é a da produção.
def test_a_production_valid_response_outscores_every_rejected_response() -> None:
    """A resposta aceita pelo pipeline pontua acima de qualquer resposta recusada."""
    assert score_scene_response(json.dumps(_PRODUCTION_RESPONSE)) == 1.0
    assert score_scene_response(json.dumps(_REJECTED_RESPONSE)) == 0.0
    assert score_scene_response("not json at all") == 0.0


# Um único campo recusado torna a resposta inválida, mesmo com todo o resto certo.
@pytest.mark.parametrize("rejected_field", ["description", "attributes", "hazards"])
def test_a_response_with_any_rejected_field_is_invalid(rejected_field: str) -> None:
    """Qualquer campo fora do contrato de cena zera a pontuação."""
    response = {**_PRODUCTION_RESPONSE, rejected_field: "anything"}
    assert score_scene_response(f"```json\n{json.dumps(response)}\n```") == 0.0


# O prompt de cena é função pura e pode ser verificado sem GPU: ele pede
# exatamente os campos que o contrato aceita e nenhum dos recusados.
def test_the_scene_prompt_asks_for_the_contract_fields_only() -> None:
    """O prompt nomeia os campos ambientais e não pede inventário de objetos."""
    prompt = scene_prompt()
    for field in ("scene_type", "environment", "layout", "lighting", "visibility", "navigability"):
        assert field in prompt
    for rejected in ("description", "attributes", "hazards"):
        assert rejected not in prompt
