"""Testes da descoberta de conceitos concretos em nível de cena (#277)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fixtures import image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.application.scene_concept_discovery import (
    discover_scene_concepts,
    parse_scene_concepts,
)
from visual_perception.config import ModuleConfig, MultimodalReasoningConfig, SceneConceptDiscoveryConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.scene_concepts import SceneConceptKind
from visual_perception.infrastructure.adapters.reasoning_prompts import concept_prompt
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Sinais contextuais entram antes das entidades no teto; estruturas genéricas
# sozinhas saem, mas uma condição visível sobre elas fica.
def test_contextual_features_take_priority_and_plain_structure_is_excluded() -> None:
    """Prioriza sinais contextuais, exclui estrutura genérica e respeita o teto."""
    response = {
        "entities": [{"concept": "Metal Door", "confidence": 0.58}, "wall", "warning sign"],
        "contextual_features": [{"concept": " wall crack. ", "confidence": 0.41}, {"concept": "debris pile"}],
    }

    concepts, discarded = parse_scene_concepts(response, SceneConceptDiscoveryConfig(max_concepts=3))

    assert [(c.text, c.kind) for c in concepts] == [
        ("wall crack", SceneConceptKind.CONTEXTUAL_FEATURE),
        ("debris pile", SceneConceptKind.CONTEXTUAL_FEATURE),
        ("metal door", SceneConceptKind.ENTITY),
    ]
    assert concepts[0].confidence == 0.41 and concepts[1].confidence is None
    assert ("wall", "structural") in discarded
    assert ("warning sign", "over_budget") in discarded


# Respostas e itens fora do contract viram descartes com motivo, nunca exceção.
@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ([], "malformed_response"),
        ({"entities": "door"}, "malformed_list"),
        ({"entities": [{"label": "door"}]}, "malformed_item"),
        ({"entities": ["door", "Door"]}, "duplicate"),
        ({"entities": ["a very long description of a door frame"]}, "not_a_noun_phrase"),
        ({"entities": ["  ..  "]}, "empty"),
    ],
)
def test_malformed_or_repeated_items_are_discarded_with_a_reason(response: object, reason: str) -> None:
    """Registra o motivo de cada descarte."""
    _, discarded = parse_scene_concepts(response, SceneConceptDiscoveryConfig())
    assert reason in {item[1] for item in discarded}


# Confiança inválida vira ausência, como no contract de região.
def test_invalid_confidence_is_treated_as_absent() -> None:
    """Não aceita confiança booleana, negativa ou acima de 1."""
    response = {
        "entities": [
            {"concept": "door", "confidence": True},
            {"concept": "pipe", "confidence": 1.4},
            {"concept": "beam", "confidence": float("nan")},
            {"concept": "cable", "confidence": float("inf")},
        ]
    }
    concepts, _ = parse_scene_concepts(response, SceneConceptDiscoveryConfig())
    assert [c.confidence for c in concepts] == [None, None, None, None]


# A descoberta usa a mesma view da cena ambiental (sem rig) e registra proveniência.
def test_discovery_uses_the_scene_view_and_records_provenance() -> None:
    """Recorta o rig antes de consultar o backend e carimba estágio e prompt."""
    seen: list[tuple[int, int]] = []

    def respond(image):  # type: ignore[no-untyped-def]
        seen.append((image.width, image.height))
        return {"entities": ["pallet"], "contextual_features": []}

    ego = np.zeros((32, 32), dtype=np.bool_)
    ego[24:, :] = True
    reasoning = MultimodalReasoningConfig(backend="fake", checkpoint="none")

    result = discover_scene_concepts(
        payload_with_blobs(),
        FakeMultimodalReasoner(concept_response_fn=respond),
        reasoning,
        SceneConceptDiscoveryConfig(),
        area_masks=ImageAreaMasks(ego_vehicle=Mask(ego, 32, 32)),
    )

    assert seen == [(32, 24)]
    assert [c.text for c in result.concepts] == ["pallet"]
    assert result.provenance.stage == "scene_concept_discovery"
    assert result.provenance.prompt_version == "concepts/v1"
    assert '"pallet"' in result.raw_response_json


# No pipeline, a etapa só roda quando habilitada e exige o port explícito.
def test_pipeline_runs_concept_discovery_only_when_enabled() -> None:
    """Desligada devolve None; ligada sem port falha; ligada com port produz conceitos."""
    payload = payload_with_blobs()
    enabled = dataclasses.replace(ModuleConfig(), scene_concept_discovery=SceneConceptDiscoveryConfig(enabled=True))
    ports = default_ports()

    assert run_canonical_pipeline(image_observation(), payload, ModuleConfig(), ports).scene_concepts is None
    with pytest.raises(ValueError, match="scene_concept_discoverer"):
        run_canonical_pipeline(image_observation(), payload, enabled, ports)

    wired = dataclasses.replace(ports, scene_concept_discoverer=ports.multimodal_reasoner)
    result = run_canonical_pipeline(image_observation(), payload, enabled, wired)

    assert result.scene_concepts is not None
    assert [c.text for c in result.scene_concepts.concepts] == ["debris pile", "reddish object"]


# O prompt pede frases curtas com o teto informado e usa placeholders.
def test_concept_prompt_states_the_budget_and_uses_placeholders() -> None:
    """Inclui o teto e não oferece um conceito concreto para ser copiado."""
    prompt = concept_prompt(7)
    assert "at most 7 phrases" in prompt
    assert "<short noun phrase>" in prompt
    assert "wall crack" not in prompt


# Teto e exclusões inválidos falham na fronteira da config.
@pytest.mark.parametrize(
    ("override", "message"),
    [({"max_concepts": 0}, "max_concepts"), ({"structural_exclusions": ("Wall",)}, "structural_exclusions")],
)
def test_scene_concept_config_is_validated(override: dict[str, object], message: str) -> None:
    """Recusa teto não positivo e exclusões fora do formato normalizado."""
    with pytest.raises(ValueError, match=message):
        SceneConceptDiscoveryConfig(**override)  # type: ignore[arg-type]
