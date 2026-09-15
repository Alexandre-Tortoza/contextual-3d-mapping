"""Testes das invariantes de fronteira validadas na entrada (#245)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import PerceptionPorts, run_canonical_pipeline
from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.grounding import SpatialRegionFootprint
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_reasoning import PriorHypothesis
from visual_perception.domain.regions import LocalRegionProposal
from visual_perception.domain.relations import CandidateRelation, RelationSource
from visual_perception.domain.semantics import Evidence


# Constrói uma máscara cheia na resolução pedida.
def _full_mask(width: int, height: int) -> Mask:
    return Mask(np.ones((height, width), dtype=np.bool_), width, height)


# Regressão da #245: com resoluções diferentes o broadcasting do numpy aceitava
# as duas máscaras, e ``scene_box`` devolvia ``None`` em silêncio.
def test_area_masks_with_different_resolutions_are_rejected() -> None:
    """Área válida e ego-veículo precisam declarar a mesma resolução."""
    with pytest.raises(ValueError, match="10x10.*10x1"):
        ImageAreaMasks(valid_area=_full_mask(10, 10), ego_vehicle=_full_mask(10, 1))
    assert ImageAreaMasks(valid_area=_full_mask(10, 10), ego_vehicle=_full_mask(10, 10)).scene_box() is None


# Um discoverer que falha se for chamado: prova que a validação acontece antes
# de qualquer port.
class _ForbiddenDiscoverer:
    """Discoverer que interrompe o teste se o pipeline chegar a consultá-lo."""

    # Falha sempre: o pipeline não deveria ter chegado até aqui.
    def discover(self, image: ImagePayload, config: RegionDiscoveryConfig) -> tuple[LocalRegionProposal, ...]:
        """Levanta ``AssertionError`` indicando que um port foi chamado cedo demais."""
        raise AssertionError("no port may run for a payload that does not match its observation")


# Um payload reescalado rodava discovery, features densas, embeddings e o
# reasoner inteiros antes de falhar na construção da observação final.
def test_a_payload_that_does_not_match_its_observation_fails_before_any_port() -> None:
    """Resolução divergente entre payload e observação falha na entrada do pipeline."""
    ports: PerceptionPorts = dataclasses.replace(default_ports(), region_discoverer=_ForbiddenDiscoverer())
    payload = payload_with_blobs(width=32, height=32)

    with pytest.raises(ValueError, match="does not match the image observation"):
        run_canonical_pipeline(image_observation(width=64, height=48), payload, default_config(), ports)


# ``clipped_to`` promete recorte; uma caixa inteiramente fora da imagem não tem
# recorte, e isso é ``None``, não uma exceção de construção.
def test_clipping_a_box_outside_the_image_returns_none() -> None:
    """Caixa fora da imagem recorta para ``None``; caixa parcial é recortada."""
    assert BoundingBox(200, 200, 300, 300).clipped_to(width=100, height=100) is None
    assert BoundingBox(50, 50, 300, 300).clipped_to(width=100, height=100) == BoundingBox(50, 50, 100, 100)


# ``PriorHypothesis`` atravessa a fronteira do prompt e era o único value object
# de domínio sem validação.
@pytest.mark.parametrize(
    "fields",
    [
        {"concept": " ", "source_region_id": "region-a", "overlap": 0.5},
        {"concept": "door", "source_region_id": "bad id", "overlap": 0.5},
        {"concept": "door", "source_region_id": "region-a", "overlap": 1.5},
        {"concept": "door", "source_region_id": "region-a", "overlap": 0.5, "similarity": -2.0},
    ],
)
def test_prior_hypothesis_rejects_invalid_values(fields: dict[str, object]) -> None:
    """Conceito vazio, identidade inválida e medidas fora de faixa falham na construção."""
    with pytest.raises(ValueError):
        PriorHypothesis(**fields)  # type: ignore[arg-type]


# Um footprint forte sem máscara afirmaria suporte espacial que não existe.
def test_spatial_footprint_validates_identity_and_strength() -> None:
    """Footprint sem identidade válida ou forte sem máscara falha."""
    with pytest.raises(ValueError, match="region_id"):
        SpatialRegionFootprint("", "door", None, False, {})
    with pytest.raises(ValueError, match="strong"):
        SpatialRegionFootprint("region-a", "door", None, True, {})
    assert SpatialRegionFootprint("region-a", "door", _full_mask(4, 4), True, {}).strong


# O vocabulário fechado da #206 vale para qualquer produtor de relação
# inferida: ``above`` foi excluído por exigir geometria 3D.
def test_model_inferred_relations_must_use_the_closed_vocabulary() -> None:
    """Predicado fora do vocabulário é recusado só para relações inferidas por modelo."""
    common = {
        "relation_id": "relation-1",
        "subject_region_id": "region-a",
        "object_region_id": "region-b",
        "confidence": None,
        "evidence": (Evidence("pair views"),),
        "provenance": ModelProvenance(stage="s", producer="p", config_fingerprint="fp"),
    }
    with pytest.raises(ValueError, match="above"):
        CandidateRelation(predicate="above", source=RelationSource.MODEL_INFERRED, **common)  # type: ignore[arg-type]
    assert CandidateRelation(predicate="part_of", source=RelationSource.MODEL_INFERRED, **common)  # type: ignore[arg-type]
    assert CandidateRelation(predicate="contains", source=RelationSource.GEOMETRIC_2D, **common)  # type: ignore[arg-type]
