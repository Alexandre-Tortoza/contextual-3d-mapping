"""Testes do grounding por conceito como fonte adicional de propostas (#277)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fixtures import image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.config import ConceptGroundingConfig, ModuleConfig, SceneConceptDiscoveryConfig
from visual_perception.domain.errors import BackendExecutionError
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal
from visual_perception.infrastructure.adapters.concept_grounding_backend import (
    Sam3ConceptGroundingAdapter,
    concept_proposals,
)
from visual_perception.infrastructure.adapters.factory import create_perception_ports


# Máscara retangular do tamanho da imagem de teste.
def _mask(x_min: int, y_min: int, x_max: int, y_max: int, size: int = 32) -> np.ndarray:
    """Retorna uma máscara booleana com o retângulo pedido."""
    data = np.zeros((size, size), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    return data


# Grounder fake: devolve uma máscara fixa por conceito e registra as consultas.
class _FakeConceptGrounder:
    """Substitui o SAM3 PCS com uma máscara determinística por conceito."""

    def __init__(self) -> None:
        """Inicia o registro de consultas."""
        self.queries: list[tuple[str, ...]] = []

    def discover_concept_regions(
        self, image: ImagePayload, concepts: tuple[str, ...], config: ConceptGroundingConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Devolve uma proposta por conceito em cantos diferentes da imagem."""
        self.queries.append(concepts)
        proposals = []
        for index, concept in enumerate(concepts):
            mask = Mask(_mask(20, 20 - 10 * index, 30, 30 - 10 * index), image.width, image.height)
            proposals.append(
                LocalRegionProposal(f"concept{index}-0", mask, mask.bounding_box(), 0.8, "fake-pcs", concept=concept)
            )
        return tuple(proposals)


# Config com descoberta de conceitos e grounding ligados.
def _enabled_config() -> ModuleConfig:
    """Retorna a config default com as duas etapas da #277 habilitadas."""
    return dataclasses.replace(
        ModuleConfig(),
        scene_concept_discovery=SceneConceptDiscoveryConfig(enabled=True),
        concept_grounding=ConceptGroundingConfig(backend="sam3"),
    )


# O teto, a área mínima e o score valem por conceito, e o conceito fica na proveniência.
def test_concept_proposals_are_sorted_capped_and_carry_the_concept() -> None:
    """Ordena por score, descarta máscara pequena ou fraca e limita por conceito."""
    image = payload_with_blobs()
    masks = (_mask(0, 0, 4, 4), _mask(0, 0, 10, 10), _mask(10, 10, 20, 20), _mask(20, 20, 30, 30))
    scores = (0.95, 0.6, 0.9, 0.3)
    config = ConceptGroundingConfig(backend="sam3", min_mask_area=20, max_regions_per_concept=1, score_threshold=0.5)

    proposals = concept_proposals(masks, scores, image, "debris pile", 2, config)

    assert [(p.local_id, p.geometric_confidence, p.concept) for p in proposals] == [("concept2-0", 0.9, "debris pile")]
    assert proposals[0].source == "sam3-pcs:facebook/sam3"
    with pytest.raises(BackendExecutionError):
        concept_proposals((np.ones((8, 8), dtype=np.bool_),), (0.9,), image, "door", 0, config)


# As propostas por conceito somam-se às genéricas, passam pelo mesmo merge e ficam
# rastreáveis ao conceito; cada conceito é uma consulta contada no custo do frame.
def test_pipeline_adds_concept_proposals_to_generic_discovery() -> None:
    """Integra conceitos da cena, grounding e merge sem remover o discovery genérico."""
    grounder = _FakeConceptGrounder()
    ports = dataclasses.replace(
        default_ports(), scene_concept_discoverer=default_ports().multimodal_reasoner, concept_region_discoverer=grounder
    )

    result = run_canonical_pipeline(image_observation(), payload_with_blobs(), _enabled_config(), ports)

    assert grounder.queries == [("debris pile", "reddish object")]
    concept_ids = {p.proposal_id: p.concept for p in result.discovered_proposals if p.concept}
    assert concept_ids == {"full-whole-concept0-0": "debris pile", "full-whole-concept1-0": "reddish object"}
    assert any(p.concept is None for p in result.discovered_proposals)
    assert result.stage_model_calls["concept_grounding"] == 2
    assert result.stage_model_calls["scene_concept_discovery"] == 1


# Sem o port explícito a composição falha em vez de ignorar a etapa pedida.
def test_pipeline_requires_the_concept_port_when_grounding_is_enabled() -> None:
    """Recusa grounding habilitado sem ConceptRegionDiscoverer."""
    ports = dataclasses.replace(default_ports(), scene_concept_discoverer=default_ports().multimodal_reasoner)
    with pytest.raises(ValueError, match="concept_region_discoverer"):
        run_canonical_pipeline(image_observation(), payload_with_blobs(), _enabled_config(), ports)


# Grounding sem descoberta de conceitos não tem o que procurar.
def test_concept_grounding_requires_scene_concept_discovery() -> None:
    """A config recusa grounding ligado com a descoberta desligada."""
    with pytest.raises(ValueError, match="scene_concept_discovery"):
        ModuleConfig(concept_grounding=ConceptGroundingConfig(backend="sam3"))


# A factory compõe o adapter real sem carregar SAM3.
def test_factory_selects_the_sam3_concept_grounder_lazily() -> None:
    """Seleciona o adapter PCS e o mesmo VLM como descobridor de conceitos."""
    config = dataclasses.replace(
        _enabled_config(),
        multimodal_reasoning=dataclasses.replace(ModuleConfig().multimodal_reasoning),
    )
    ports = create_perception_ports(config)

    assert isinstance(ports.concept_region_discoverer, Sam3ConceptGroundingAdapter)
    assert ports.scene_concept_discoverer is ports.multimodal_reasoner


# O conceito de origem, quando presente, precisa ser texto normalizado.
def test_proposal_concept_must_be_stripped_text() -> None:
    """Recusa conceito vazio ou com espaços nas pontas."""
    mask = Mask(_mask(0, 0, 4, 4), 32, 32)
    with pytest.raises(ValueError, match="concept"):
        LocalRegionProposal("p0", mask, mask.bounding_box(), 0.9, "fake", concept=" door ")
