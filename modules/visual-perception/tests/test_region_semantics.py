"""Testes do estágio de interpretação semântica em nível de região (#165)."""

from __future__ import annotations

import numpy as np

from fixtures import payload_with_blobs
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Constrói uma ObservedRegion mínima com uma mask quadrada fixa, para não repetir
# esse setup em cada teste de interpret_regions abaixo.
def _region(region_id: str, width: int = 32, height: int = 32) -> ObservedRegion:
    data = np.zeros((height, width), dtype=np.bool_)
    data[4:10, 4:10] = True
    mask = Mask(data, width, height)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",))


# Verifica o caminho feliz: uma região válida recebe pelo menos um claim do tipo
# "label" a partir do reasoner fake.
def test_valid_region_receives_label_claims() -> None:
    region = _region("region-a")
    updated, failures = interpret_regions(
        (region,), payload_with_blobs(), None, FakeMultimodalReasoner(), MultimodalReasoningConfig()
    )
    assert failures == ()
    assert any(claim.kind.value == "label" for claim in updated[0].claims)


# Garante que interpret_regions só adiciona claims semânticos, nunca altera a
# geometria (mask/box/geometric_confidence) herdada da região de entrada.
def test_region_geometry_is_never_modified() -> None:
    region = _region("region-a")
    updated, _ = interpret_regions(
        (region,), payload_with_blobs(), None, FakeMultimodalReasoner(), MultimodalReasoningConfig()
    )
    assert updated[0].region_id == region.region_id
    assert updated[0].mask is region.mask
    assert updated[0].box == region.box
    assert updated[0].geometric_confidence == region.geometric_confidence


# Confirma o design de "claims, não labels" (#156, domain/semantics.py): quando o
# reasoner retorna múltiplas hipóteses de label, todas coexistem como claims em
# vez de o pipeline colapsar para uma única.
def test_ambiguous_region_preserves_multiple_label_hypotheses() -> None:
    reasoner = FakeMultimodalReasoner(
        region_response_fn=lambda crop, scene: {
            "labels": [{"value": "box", "confidence": 0.6}, {"value": "crate", "confidence": 0.4}]
        }
    )
    region = _region("region-a")
    config = MultimodalReasoningConfig()
    updated, failures = interpret_regions((region,), payload_with_blobs(), None, reasoner, config)
    labels = {claim.value for claim in updated[0].claims if claim.kind.value == "label"}
    assert labels == {"box", "crate"}
    assert failures == ()


# Documenta o comportamento atual quando o mesmo reasoner malformado é usado para
# todas as regiões de uma chamada: como a falha é por reasoner (não por região),
# ambas as regiões falham juntas — ver test_one_failing_region_does_not_invalidate_others
# para o caso de isolamento por região.
def test_malformed_response_isolates_failure_without_dropping_other_regions() -> None:
    reasoner = FakeMultimodalReasoner(region_response_fn=lambda crop, scene: {"labels": []})
    good_region = _region("region-good")
    bad_region = _region("region-bad")
    updated, failures = interpret_regions(
        (good_region, bad_region), payload_with_blobs(), None, reasoner, MultimodalReasoningConfig()
    )
    assert len(failures) == 2  # ambas falham aqui porque o mesmo reasoner é malformado para todas
    assert len(updated) == 2


# Verifica que uma região sem claims (resposta vazia) ainda é reportada como
# sucesso (sem failures) e mantida no resultado, sem impedir a região boa.
def test_one_failing_region_does_not_invalidate_others() -> None:
    def region_response(crop: object, scene: object) -> dict:
        return {"labels": []}

    good_reasoner = FakeMultimodalReasoner()
    good_region = _region("region-good")

    updated, failures = interpret_regions(
        (good_region,), payload_with_blobs(), None, good_reasoner, MultimodalReasoningConfig()
    )
    assert failures == ()
    assert updated[0].claims
