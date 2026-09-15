"""Testes de contract da saída de observação visual canônica (#154)."""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_contracts import FrameId, ObservationReference, Timestamp

from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.visual_observation import SceneContext, VisualObservation


# Mask mínima reutilizável (um único pixel ativo) para montar regiões de teste sem
# repetir a construção do array booleano em cada caso.
def _mask(width: int = 4, height: int = 4) -> Mask:
    data = np.zeros((height, width), dtype=np.bool_)
    data[0, 0] = True
    return Mask(data, width, height)


# Constrói uma ObservedRegion mínima sobre _mask, para os testes de
# VisualObservation que precisam de pelo menos uma região.
def _region(region_id: str, width: int = 4, height: int = 4) -> ObservedRegion:
    mask = _mask(width, height)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-proposal",))


# ObservationReference mínima e reutilizável, representando a observação RGB de
# origem que toda VisualObservation de teste referencia.
def _source(observation_id: str = "obs-1") -> ObservationReference:
    return ObservationReference(
        observation_id=observation_id,
        dataset_id="ds",
        sequence_id="seq",
        sensor_id="camera_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=0, clock_id="rosbag"),
        frame_id=FrameId("camera_1_optical_frame"),
    )


# Uma VisualObservation sem regiões é válida e deve simplesmente expor uma tupla
# vazia, não None nem erro.
def test_minimal_observation_has_no_regions() -> None:
    observation = VisualObservation(_source(), 4, 4, SceneContext(), (), ())
    assert observation.regions == ()


# observation_id não é um campo próprio da VisualObservation: deve vir sempre da
# ObservationReference de origem (source), preservando um único ponto de verdade
# para a identidade da observação.
def test_observation_id_comes_from_source() -> None:
    observation = VisualObservation(_source("obs-42"), 4, 4, SceneContext(), (), ())
    assert observation.observation_id == "obs-42"


# region_by_id deve encontrar de volta exatamente a mesma instância de região que
# foi passada na construção (identidade, não só igualdade).
def test_complete_observation_round_trips_region_lookup() -> None:
    region = _region("region-a")
    observation = VisualObservation(_source(), 4, 4, SceneContext(), (region,), ())
    assert observation.region_by_id("region-a") is region


# Dois region_id duplicados quebrariam a busca por id (região ambígua) — a
# construção deve rejeitar isso cedo.
def test_rejects_duplicate_region_ids() -> None:
    region = _region("region-a")
    with pytest.raises(ValueError):
        VisualObservation(_source(), 4, 4, SceneContext(), (region, region), ())


# A resolução da mask de uma região precisa bater com a resolução da imagem da
# observação; um descompasso indicaria uma região calculada contra a imagem errada.
def test_rejects_region_mask_resolution_mismatch() -> None:
    region = _region("region-a", width=8, height=8)
    with pytest.raises(ValueError):
        VisualObservation(_source(), 4, 4, SceneContext(), (region,), ())


# Buscar um region_id que não existe deve levantar KeyError, seguindo a semântica
# usual de lookup por chave, em vez de retornar None silenciosamente.
def test_unknown_region_lookup_raises_key_error() -> None:
    observation = VisualObservation(_source(), 4, 4, SceneContext(), (), ())
    with pytest.raises(KeyError):
        observation.region_by_id("missing")


# A partição não pode transformar uma relação em referência pendurada: a
# evidência publicada precisa poder apontar para a superfície que a hospeda,
# e a superfície não é publicada.
def test_a_relation_may_reference_a_structural_context_region() -> None:
    """Relações e grupos são validados contra a união das duas metades."""
    from visual_perception.domain.references import ModelProvenance
    from visual_perception.domain.relations import CandidateRelation, RelationSource
    from visual_perception.domain.semantics import Evidence

    relation = CandidateRelation(
        relation_id="relation-1",
        subject_region_id="region-crack",
        object_region_id="region-wall",
        predicate="part_of",
        confidence=None,
        evidence=(Evidence("measured containment"),),
        provenance=ModelProvenance(stage="s", producer="p", config_fingerprint="fp"),
        source=RelationSource.MODEL_INFERRED,
    )
    observation = VisualObservation(
        _source(),
        4,
        4,
        SceneContext(),
        (_region("region-crack"),),
        (relation,),
        structural_context=(_region("region-wall"),),
    )

    assert observation.region_by_id("region-wall").region_id == "region-wall"
    assert len(observation.all_regions) == 2


# Uma mesma região não pode estar dos dois lados da partição: se pudesse, todo
# consumidor que varre a união a contaria duas vezes.
def test_rejects_a_region_id_present_in_both_halves() -> None:
    """A unicidade de ``region_id`` vale sobre a união, não sobre cada metade."""
    with pytest.raises(ValueError, match="unique region_id"):
        VisualObservation(
            _source(),
            4,
            4,
            SceneContext(),
            (_region("region-1"),),
            (),
            structural_context=(_region("region-1"),),
        )


# A resolução da máscara é validada nas duas metades: uma região suprimida
# continua sendo evidência, e evidência inconsistente precisa falhar igual.
def test_rejects_structural_context_mask_resolution_mismatch() -> None:
    """Uma máscara fora de resolução no contexto estrutural é recusada."""
    with pytest.raises(ValueError, match="mask resolution"):
        VisualObservation(
            _source(),
            4,
            4,
            SceneContext(),
            (),
            (),
            structural_context=(_region("region-1", width=8, height=8),),
        )
