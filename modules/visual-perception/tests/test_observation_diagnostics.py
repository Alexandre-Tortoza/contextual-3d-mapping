"""Testes do diagnóstico estatístico de uma observação (#212).

O ponto destes testes é que o diagnóstico afirme coisas verdadeiras sobre um
frame sem que ninguém precise abrir uma imagem — em particular, que o colapso
de modo seja detectável por número, e que os dois eixos de confiança nunca se
contaminem.
"""

from __future__ import annotations

import numpy as np

from fixtures import image_observation
from visual_perception.application.observation_diagnostics import diagnose_observation
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import (
    ObservedRegion,
    ProposalRejectionReason,
    RegionProposal,
    RejectedProposal,
    TileProvenance,
)
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation

_WIDTH = 32
_HEIGHT = 32


# Constrói um claim de label com ou sem score, para os testes de distribuição.
def _label(
    value: str,
    confidence: float | None,
    *,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    category: str | None = None,
    region_kind: RegionKind | None = None,
) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        None if confidence is None else ConfidenceScore(confidence, "fake"),
        (Evidence(description="raw multimodal region response"),),
        ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="fp"),
        role=role,
        category=category,
        region_kind=region_kind,
    )


# Constrói uma região com máscara própria e os claims informados; a máscara varia
# por índice para que as regiões não colidam na validação de unicidade.
def _region(
    index: int,
    *,
    label: str | None = None,
    semantic_confidence: float | None = 0.9,
    geometric_confidence: float = 0.95,
    proposals: int = 1,
    category: str | None = None,
    region_kind: RegionKind | None = None,
    claims: tuple[SemanticClaim, ...] | None = None,
) -> ObservedRegion:
    data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    row = index % (_HEIGHT - 2)
    data[row : row + 2, 0:4] = True
    mask = Mask(data, _WIDTH, _HEIGHT)
    if claims is None:
        claims = (
            ()
            if label is None
            else (
                _label(label, semantic_confidence, category=category, region_kind=region_kind),
            )
        )
    return ObservedRegion(
        f"region-{index:04d}",
        mask,
        mask.bounding_box(),
        geometric_confidence,
        tuple(f"region-{index:04d}-p{n}" for n in range(proposals)),
        claims=claims,
    )


# Monta uma observação a partir de regiões e claims de cena já construídas.
def _observation(
    regions: tuple[ObservedRegion, ...], scene: tuple[SemanticClaim, ...] = ()
) -> VisualObservation:
    return VisualObservation(
        source=image_observation(width=_WIDTH, height=_HEIGHT).source,
        image_width=_WIDTH,
        image_height=_HEIGHT,
        scene_context=SceneContext(claims=scene),
        regions=regions,
        relations=(),
    )


# O caso que motivou o diagnóstico inteiro: em corridor-02-002, 45 de 45 regiões
# saíram como "curved wall". Sem esta métrica, esse colapso só aparecia para quem
# abrisse o overlay e contasse à mão.
def test_mode_collapse_reports_the_dominant_label_and_its_fraction() -> None:
    """Nove regiões iguais e uma diferente produzem fração dominante de 0,9."""
    regions = tuple(_region(index, label="wall") for index in range(9))
    regions += (_region(9, label="door"),)

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=10)

    assert diagnostics.mode_collapse.dominant_label == "wall"
    assert diagnostics.mode_collapse.dominant_fraction == 0.9
    assert diagnostics.mode_collapse.distinct_labels == 2


# Uma observação sem regiões é um resultado legítimo (imagem vazia), e o
# diagnóstico precisa descrevê-la em vez de estourar uma divisão por zero.
def test_empty_observation_is_described_without_dividing_by_zero() -> None:
    """Uma observação sem regiões produz diagnóstico vazio, sem erro."""
    diagnostics = diagnose_observation(_observation(()), discovered_proposals=0)

    assert diagnostics.region_count == 0
    assert diagnostics.mode_collapse.dominant_label is None
    assert diagnostics.mode_collapse.dominant_fraction == 0.0
    assert diagnostics.semantic_confidence.mean is None


# Empate precisa ser resolvido de forma estável: sem isso, dois runs idênticos
# poderiam gravar labels dominantes diferentes e sugerir uma mudança que não houve.
def test_tied_dominant_label_is_resolved_deterministically() -> None:
    """Um empate entre labels é desempatado pelo nome, de forma reprodutível."""
    regions = (_region(0, label="wall"), _region(1, label="door"))

    first = diagnose_observation(_observation(regions), discovered_proposals=2)
    second = diagnose_observation(_observation(regions), discovered_proposals=2)

    assert first.mode_collapse.dominant_label == "door"
    assert second.mode_collapse.dominant_label == "door"


# O bug que este módulo existe para tornar impossível: a confiança geométrica
# sendo lida como se fosse a semântica. Aqui elas são deliberadamente distantes,
# e nenhuma estatística de uma pode coincidir com a da outra.
def test_semantic_and_geometric_confidence_never_collapse_into_one_number() -> None:
    """Os dois eixos de confiança são resumidos a partir de fontes distintas."""
    regions = tuple(
        _region(index, label="wall", semantic_confidence=0.10, geometric_confidence=0.97)
        for index in range(4)
    )

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=4)

    assert diagnostics.semantic_confidence.mean == 0.10
    assert diagnostics.geometric_confidence.mean == 0.97
    assert diagnostics.semantic_confidence.missing == 0


# Ausência de score é informação: o contract permite ``confidence=None``, e
# contá-la como zero inventaria uma incerteza que o produtor não afirmou.
def test_unscored_label_counts_as_missing_and_stays_out_of_the_mean() -> None:
    """Um label sem score é contado como ausente, não como zero."""
    regions = (
        _region(0, label="wall", semantic_confidence=0.80),
        _region(1, label="door", semantic_confidence=None),
    )

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=2)

    assert diagnostics.semantic_confidence.missing == 1
    assert diagnostics.semantic_confidence.count == 1
    assert diagnostics.semantic_confidence.mean == 0.80


# Distinguir over-segmentation do SAM de falha do merge exige as duas contagens.
# Sem proposal_count vindo de fora, o diagnóstico só saberia o lado pós-merge.
def test_proposal_and_merge_counts_are_reported_separately() -> None:
    """As contagens de proposals e de regiões mostram o que o merge fez."""
    regions = (_region(0, label="wall", proposals=3), _region(1, label="door", proposals=1))

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=4)

    assert diagnostics.proposal_count == 4
    assert diagnostics.merged_proposal_count == 4
    assert diagnostics.region_count == 2
    assert diagnostics.regions_with_multiple_proposals == 1


# É a métrica que torna comparável o vazamento de contexto de cena entre os
# modos local_first e context_assisted. Precisa contar, e nunca filtrar.
def test_scene_echo_counts_regions_that_repeat_a_scene_claim() -> None:
    """Regiões cujo label repete uma claim de cena são contadas, não descartadas."""
    scene = (
        SemanticClaim(
            ClaimKind.SCENE_TYPE,
            "corridor",
            None,
            (Evidence(description="raw multimodal scene response"),),
            ModelProvenance(stage="scene_context", producer="fake", config_fingerprint="fp"),
        ),
    )
    regions = (
        _region(0, label="Corridor"),
        _region(1, label="corridor"),
        _region(2, label="door"),
    )

    diagnostics = diagnose_observation(_observation(regions, scene), discovered_proposals=3)

    assert diagnostics.scene_echo_label_count == 2
    assert diagnostics.region_count == 3


# O diagnóstico antigo publicava um campo ``kinds`` que contava ``ClaimKind``
# (label/attribute/condition/material), não ``RegionKind``. Em
# ``corridor-02-002`` isso produzia ``[["attribute",130],["label",126],...]``,
# um número sobre a forma da resposta e não sobre a natureza das regiões.
def test_region_kinds_counts_region_kind_and_not_claim_kind() -> None:
    """``region_kinds`` conta thing/stuff/part/unknown, não o tipo do claim."""
    regions = (
        _region(0, label="wall", region_kind=RegionKind.STUFF),
        _region(1, label="carpet", region_kind=RegionKind.STUFF),
        _region(2, label="door", region_kind=RegionKind.THING),
        _region(3, label="handle", region_kind=RegionKind.PART),
    )

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=4)

    assert diagnostics.region_kinds == (("stuff", 2), ("part", 1), ("thing", 1))


# Uma região sem interpretação, ou cujo produtor não informou o kind, conta
# como ``unknown``. Ela não pode sumir da contagem nem virar ``thing``.
def test_regions_without_a_declared_kind_count_as_unknown() -> None:
    """Ausência de kind é contada como ``unknown``, nunca omitida."""
    regions = (
        _region(0, label="wall", region_kind=RegionKind.STUFF),
        _region(1, label="thing-with-no-kind"),
        _region(2),
    )

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=3)

    assert diagnostics.region_kinds == (("unknown", 2), ("stuff", 1))
    assert sum(count for _, count in diagnostics.region_kinds) == len(regions)


# ``category`` é uma informação distinta de ``label`` e precisa ser contável
# separadamente: 18 regiões "plain wall" e 10 "wall" podem ser a mesma
# categoria ``wall``, e só o histograma de categorias mostra isso.
def test_categories_are_counted_separately_from_labels() -> None:
    """O histograma de categorias é independente do de labels."""
    regions = (
        _region(0, label="plain wall", category="wall"),
        _region(1, label="blank wall", category="wall"),
        _region(2, label="carpet", category="flooring"),
    )

    diagnostics = diagnose_observation(_observation(regions), discovered_proposals=3)

    assert diagnostics.category_counts == (("wall", 2), ("flooring", 1))
    assert diagnostics.label_counts == (
        ("blank wall", 1),
        ("carpet", 1),
        ("plain wall", 1),
    )
    assert diagnostics.mode_collapse.dominant_category == "wall"
    assert diagnostics.mode_collapse.dominant_category_fraction == 2 / 3
    assert diagnostics.mode_collapse.distinct_categories == 2


# O run real reportou 0,9 exato nas 60 regiões. O número precisa ser
# preservado como veio — mas o diagnóstico tem de conseguir dizer que aquela
# distribuição não tem variância nenhuma, que é a informação que importa.
def test_a_degenerate_confidence_distribution_is_detectable() -> None:
    """Confiança constante é preservada e sinalizada como degenerada."""
    regions = tuple(_region(index, label=f"l{index}", semantic_confidence=0.9) for index in range(6))

    stats = diagnose_observation(_observation(regions), discovered_proposals=6).semantic_confidence

    assert stats.minimum == 0.9
    assert stats.maximum == 0.9
    assert stats.distinct_values == 1
    assert stats.stddev == 0.0
    assert stats.degenerate


# A contrapartida: uma distribuição com variância real não é degenerada.
def test_a_varied_confidence_distribution_is_not_degenerate() -> None:
    """Confiança com variação real não é sinalizada como degenerada."""
    regions = (
        _region(0, label="a", semantic_confidence=0.2),
        _region(1, label="b", semantic_confidence=0.9),
    )

    stats = diagnose_observation(_observation(regions), discovered_proposals=2).semantic_confidence

    assert stats.distinct_values == 2
    assert stats.stddev is not None and stats.stddev > 0.0
    assert not stats.degenerate


# Um único valor não caracteriza degeneração: não há distribuição a avaliar.
def test_a_single_scored_region_is_not_called_degenerate() -> None:
    """Uma amostra só não é uma distribuição degenerada."""
    stats = diagnose_observation(
        _observation((_region(0, label="wall", semantic_confidence=0.9),)), discovered_proposals=1
    ).semantic_confidence

    assert stats.count == 1
    assert not stats.degenerate


# Ausência continua contada à parte e fora da variância: nenhum valor ausente
# pode virar 0.0 para "completar" a distribuição.
def test_missing_confidence_stays_out_of_the_variance() -> None:
    """Confiança ausente é contada em ``missing`` e não entra na variância."""
    regions = (
        _region(0, label="a", semantic_confidence=0.9),
        _region(1, label="b", semantic_confidence=None),
    )

    stats = diagnose_observation(_observation(regions), discovered_proposals=2).semantic_confidence

    assert stats.count == 1
    assert stats.missing == 1
    assert stats.distinct_values == 1
    assert stats.minimum == 0.9


# Regressão direta do run: sete regiões de ``corridor-02-002`` persistiam
# ``carpet`` duas ou três vezes como hipóteses irmãs. O diagnóstico precisa
# tornar isso contável para que a correção do parser seja verificável.
def test_duplicate_label_hypotheses_are_reported() -> None:
    """Regiões com hipóteses de label repetidas são contadas."""
    duplicated = _region(
        0,
        claims=(
            _label("carpet", 0.9),
            _label("carpet", 0.9, role=HypothesisRole.ALTERNATIVE),
            _label("carpet", 0.1, role=HypothesisRole.ALTERNATIVE),
        ),
    )
    clean = _region(
        1,
        claims=(_label("wall", 0.9), _label("door", 0.2, role=HypothesisRole.ALTERNATIVE)),
    )

    diagnostics = diagnose_observation(_observation((duplicated, clean)), discovered_proposals=2)

    assert diagnostics.duplicate_label_hypotheses == 1


# A deduplicação compara caixa e espaçamento, a mesma regra do parser: se ela
# divergisse, o diagnóstico não mediria o que a correção corrigiu.
def test_duplicate_detection_normalizes_case_and_whitespace() -> None:
    """Hipóteses que diferem só por caixa/espaço contam como duplicadas."""
    region = _region(
        0,
        claims=(
            _label("Plain  Wall", 0.9),
            _label("plain wall", 0.4, role=HypothesisRole.ALTERNATIVE),
        ),
    )

    assert diagnose_observation(_observation((region,)), discovered_proposals=1).duplicate_label_hypotheses == 1


# Constrói uma proposal global retangular, para os testes de pós-condição da
# filtragem.
def _proposal(proposal_id: str, box: tuple[int, int, int, int]) -> RegionProposal:
    x_min, y_min, x_max, y_max = box
    data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    mask = Mask(data, _WIDTH, _HEIGHT)
    return RegionProposal(
        proposal_id=proposal_id,
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=0.9,
        source="sam:test",
        tile=TileProvenance(scale_id="full", tile_id="whole"),
    )


# Os campos de gate medem a **pós-condição** da filtragem, não quantos
# descartes houve. Confundir as duas coisas faria um filtro correto — que
# rejeitou oito proposals fora da lente — parecer reprovado no gate.
def test_the_gate_fields_measure_what_survived_not_what_was_rejected() -> None:
    """Descartes e sobreviventes são contados em campos distintos."""
    valid = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    valid[0:20, :] = True
    masks = ImageAreaMasks(valid_area=Mask(valid, _WIDTH, _HEIGHT))
    rejections = (
        RejectedProposal("sam-1", ProposalRejectionReason.OUTSIDE_VALID_AREA, 0.0),
        RejectedProposal("sam-2", ProposalRejectionReason.OUTSIDE_VALID_AREA, 0.1),
    )

    diagnostics = diagnose_observation(
        _observation((_region(0, label="wall"),)),
        discovered_proposals=3,
        kept_proposals=(_proposal("sam-3", (2, 2, 10, 10)),),
        proposal_rejections=rejections,
        area_masks=masks,
    )

    assert diagnostics.fisheye.proposals_rejected == 2
    assert diagnostics.fisheye.proposals_outside_valid_area == 0
    assert diagnostics.fisheye.applied


# Se a filtragem deixasse passar uma proposal fora da área válida, o gate teria
# de acusar. É a única forma de o campo significar alguma coisa.
def test_a_surviving_proposal_outside_the_valid_area_is_counted() -> None:
    """Uma proposal sobrevivente fora da área válida aparece no gate."""
    valid = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    valid[0:10, :] = True
    masks = ImageAreaMasks(valid_area=Mask(valid, _WIDTH, _HEIGHT))

    diagnostics = diagnose_observation(
        _observation((_region(0, label="wall"),)),
        discovered_proposals=1,
        kept_proposals=(_proposal("sam-1", (2, 20, 10, 30)),),
        area_masks=masks,
    )

    assert diagnostics.fisheye.proposals_outside_valid_area == 1
