"""Testes da filtragem de proposals entre discovery e merge (#202).

Até esta etapa, ``_discover_regions`` promovia toda proposal a região: em
``corridor-02-002``, 64 proposals viravam 60 regiões, e as únicas exclusões
eram área mínima e score, dentro do adapter do SAM. O resultado tinha 7
regiões majoritariamente fora do círculo útil da lente, 19 sobre o rig e 48
pares com containment acima de 0,8.

Os testes abaixo fixam as regras de **validade**. Redundância geométrica
continua com ``merge_regions``, que une duplicatas preservando os dois
``contributing_proposal_ids``. Nenhum ``top-N``: um corte cego por número
descartaria evidência boa junto com a ruim e tornaria a métrica de regiões
autorrealizável.
"""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.application.proposal_filtering import filter_proposals
from visual_perception.config import ImageAreaConfig, ProposalFilterConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry, ImageAreaMasks
from visual_perception.domain.regions import (
    ProposalRejectionReason,
    RegionProposal,
    TileProvenance,
)

_WIDTH = 100
_HEIGHT = 100


# Constrói uma proposal retangular em coordenadas globais, o formato que chega
# à filtragem depois do remapeamento de tile.
def _proposal(
    proposal_id: str,
    box: tuple[int, int, int, int],
    *,
    confidence: float = 0.9,
) -> RegionProposal:
    x_min, y_min, x_max, y_max = box
    data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    mask = Mask(data, _WIDTH, _HEIGHT)
    return RegionProposal(
        proposal_id=proposal_id,
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=confidence,
        source="sam:test",
        tile=TileProvenance(scale_id="full", tile_id="whole"),
    )


# Máscaras de área para os testes: a válida cobre a metade superior, e o ego
# ocupa a faixa inferior.
def _masks(*, valid: bool = True, ego: bool = True) -> ImageAreaMasks:
    valid_data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    valid_data[0:95, :] = True
    ego_data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    ego_data[80:95, :] = True
    return ImageAreaMasks(
        valid_area=Mask(valid_data, _WIDTH, _HEIGHT) if valid else None,
        ego_vehicle=Mask(ego_data, _WIDTH, _HEIGHT) if ego else None,
    )


# Uma proposal inteiramente dentro da área válida e longe do rig sobrevive.
# É o caso que garante que a filtragem remove evidência ruim, não evidência.
def test_a_valid_proposal_survives() -> None:
    kept, rejected = filter_proposals(
        (_proposal("sam-1", (10, 10, 40, 40)),),
        area_masks=_masks(),
        config=ProposalFilterConfig(),
    )

    assert [proposal.proposal_id for proposal in kept] == ["sam-1"]
    assert rejected == ()


# Uma proposal majoritariamente fora do círculo útil da lente é descartada:
# ali os pixels são o corpo preto da objetiva, não cena.
def test_a_proposal_mostly_outside_the_valid_area_is_rejected() -> None:
    kept, rejected = filter_proposals(
        (_proposal("sam-1", (10, 96, 40, 100)),),
        area_masks=_masks(),
        config=ProposalFilterConfig(),
    )

    assert kept == ()
    assert [item.reason for item in rejected] == [ProposalRejectionReason.OUTSIDE_VALID_AREA]


# Uma proposal com sobreposição alta com o rig é descartada, e o motivo e a
# medida ficam registrados para auditoria.
def test_a_proposal_overlapping_the_ego_vehicle_is_rejected_with_its_measure() -> None:
    kept, rejected = filter_proposals(
        (_proposal("sam-1", (10, 82, 40, 94)),),
        area_masks=_masks(),
        config=ProposalFilterConfig(),
    )

    assert kept == ()
    assert rejected[0].reason is ProposalRejectionReason.EGO_VEHICLE_OVERLAP
    assert rejected[0].proposal_id == "sam-1"
    assert rejected[0].value == pytest.approx(1.0)


# Sem geometria declarada não há exclusão de área: a filtragem roda, mas nada
# é descartado por ego nem por lente. Silenciar isso faria uma sequência sem
# calibração perder regiões sem que nada explicasse por quê.
def test_without_declared_areas_nothing_is_rejected_by_area() -> None:
    proposals = (_proposal("sam-1", (10, 82, 40, 94)), _proposal("sam-2", (10, 96, 40, 100)))

    kept, rejected = filter_proposals(
        proposals, area_masks=ImageAreaMasks(), config=ProposalFilterConfig()
    )

    assert [proposal.proposal_id for proposal in kept] == ["sam-1", "sam-2"]
    assert rejected == ()


# Uma parte dentro de um todo não é descartada — uma maçaneta dentro de uma
# porta é evidência própria. É a mesma política que ``merge_regions`` aplica em
# ``test_contained_part_is_not_removed``, e mantê-las alinhadas é o que impede a
# filtragem de apagar evidência a pretexto de remover duplicata.
def test_a_small_part_inside_a_larger_proposal_is_preserved() -> None:
    whole = _proposal("sam-whole", (10, 10, 50, 50), confidence=0.95)
    part = _proposal("sam-part", (20, 20, 30, 30), confidence=0.80)

    kept, rejected = filter_proposals(
        (whole, part), area_masks=_masks(), config=ProposalFilterConfig()
    )

    assert [proposal.proposal_id for proposal in kept] == ["sam-whole", "sam-part"]
    assert rejected == ()


# Duas proposals quase idênticas sobrevivem à filtragem de propósito: quem as
# reconcilia é ``merge_regions``, que as une numa região só preservando os dois
# identificadores de origem. Descartá-las aqui perderia essa proveniência.
def test_near_duplicates_are_left_for_the_geometric_merge() -> None:
    original = _proposal("sam-1", (10, 10, 50, 50), confidence=0.95)
    duplicate = _proposal("sam-2", (11, 11, 50, 50), confidence=0.90)

    kept, rejected = filter_proposals(
        (original, duplicate), area_masks=_masks(), config=ProposalFilterConfig()
    )

    assert [proposal.proposal_id for proposal in kept] == ["sam-1", "sam-2"]
    assert rejected == ()


# A filtragem não depende da ordem em que discovery devolveu as proposals: só
# há regras locais, e dois runs da mesma configuração produzem o mesmo conjunto.
def test_filtering_is_independent_of_proposal_order() -> None:
    first = _proposal("sam-a", (10, 10, 50, 50), confidence=0.9)
    second = _proposal("sam-b", (10, 82, 40, 94), confidence=0.9)

    forward, forward_rejected = filter_proposals(
        (first, second), area_masks=_masks(), config=ProposalFilterConfig()
    )
    backward, backward_rejected = filter_proposals(
        (second, first), area_masks=_masks(), config=ProposalFilterConfig()
    )

    assert [proposal.proposal_id for proposal in forward] == ["sam-a"]
    assert [proposal.proposal_id for proposal in backward] == ["sam-a"]
    assert [item.proposal_id for item in forward_rejected] == ["sam-b"]
    assert [item.proposal_id for item in backward_rejected] == ["sam-b"]


# Uma proposal minúscula é ruído de sliver do SAM, não evidência de região.
def test_a_proposal_below_the_minimum_relative_area_is_rejected() -> None:
    kept, rejected = filter_proposals(
        (_proposal("sam-1", (10, 10, 12, 12)),),
        area_masks=_masks(),
        config=ProposalFilterConfig(min_relative_area=0.01),
    )

    assert kept == ()
    assert rejected[0].reason is ProposalRejectionReason.BELOW_MIN_RELATIVE_AREA


# Uma proposal que cobre quase o frame inteiro descreve a cena, não uma
# região dela.
def test_a_proposal_above_the_maximum_relative_area_is_rejected() -> None:
    kept, rejected = filter_proposals(
        (_proposal("sam-1", (0, 0, 100, 60)),),
        area_masks=_masks(),
        config=ProposalFilterConfig(max_relative_area=0.4),
    )

    assert kept == ()
    assert rejected[0].reason is ProposalRejectionReason.ABOVE_MAX_RELATIVE_AREA


# Toda proposal descartada aparece exatamente uma vez no registro: a soma de
# mantidas e descartadas tem de fechar com o total recebido, ou a filtragem
# estaria perdendo evidência em silêncio.
def test_kept_and_rejected_account_for_every_proposal() -> None:
    proposals = (
        _proposal("sam-1", (10, 10, 40, 40)),
        _proposal("sam-2", (10, 82, 40, 94)),
        _proposal("sam-3", (60, 10, 62, 12), confidence=0.5),
        _proposal("sam-4", (10, 96, 40, 100)),
    )

    kept, rejected = filter_proposals(
        proposals, area_masks=_masks(), config=ProposalFilterConfig()
    )

    assert len(kept) + len(rejected) == len(proposals)
    identifiers = {proposal.proposal_id for proposal in kept} | {
        item.proposal_id for item in rejected
    }
    assert identifiers == {"sam-1", "sam-2", "sam-3", "sam-4"}


# A filtragem opera sobre máscaras. Ela nunca toca nos pixels de origem, que é
# a regra que a #212 estabeleceu depois de medir o custo de violá-la.
def test_filtering_never_alters_the_proposal_masks() -> None:
    proposal = _proposal("sam-1", (10, 10, 40, 40))
    before = proposal.mask.data.copy()

    filter_proposals((proposal,), area_masks=_masks(), config=ProposalFilterConfig())

    assert np.array_equal(proposal.mask.data, before)


# A configuração de área traduz geometria declarada em máscaras rasterizadas
# uma vez por frame, na resolução do frame.
def test_image_area_config_rasterizes_the_declared_geometry() -> None:
    config = ImageAreaConfig(
        valid_area=ImageAreaGeometry(circle=CircleArea(50.0, 50.0, 40.0)),
        ego_vehicle=ImageAreaGeometry(polygons=(((0.0, 90.0), (99.0, 90.0), (99.0, 99.0), (0.0, 99.0)),)),
    )

    masks = config.rasterize(_WIDTH, _HEIGHT)

    assert masks.valid_area is not None and bool(masks.valid_area.data[50, 50])
    assert masks.ego_vehicle is not None and bool(masks.ego_vehicle.data[95, 50])


# Sem geometria declarada, a configuração produz máscaras ausentes — não
# máscaras vazias, que rejeitariam o frame inteiro.
def test_image_area_config_without_geometry_produces_no_masks() -> None:
    masks = ImageAreaConfig().rasterize(_WIDTH, _HEIGHT)

    assert masks.is_empty
