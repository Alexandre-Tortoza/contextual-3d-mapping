"""Testes do prior temporal: casamento, derivação e medição de eco."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fixtures import image_observation
from visual_perception.application.observation_diagnostics import _prior_stats
from visual_perception.application.temporal_prior import (
    PriorAssignment,
    match_scene_prior,
    prior_from,
)
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.embeddings import VisualEmbedding
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_reasoning import (
    PriorHypothesis,
    PriorRegion,
    ScenePrior,
    TemporalPriorMode,
    select_region_prior,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")
_WIDTH, _HEIGHT = 96, 32


# Empacota regiões numa observação completa, para que cada teste declare só
# as regiões que lhe interessam.
def _observation(
    regions: tuple[ObservedRegion, ...] = (),
    structural_context: tuple[ObservedRegion, ...] = (),
) -> VisualObservation:
    return VisualObservation(
        source=image_observation(width=_WIDTH, height=_HEIGHT).source,
        image_width=_WIDTH,
        image_height=_HEIGHT,
        scene_context=SceneContext(claims=()),
        regions=regions,
        relations=(),
        structural_context=structural_context,
    )


# Constrói uma claim de identidade com o papel pedido, para que os testes
# controlem qual hipótese o frame concluiu.
def _claim(
    value: str,
    *,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    category: str | None = None,
) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="fake"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=role,
        category=category,
        region_kind=RegionKind.THING,
    )


# Constrói uma região retangular na faixa horizontal pedida, para que a
# sobreposição entre frames seja escolhida pelo teste.
def _region(
    region_id: str,
    x_min: int,
    x_max: int,
    claims: tuple[SemanticClaim, ...] = (),
    *,
    embedding_ref: str | None = None,
) -> ObservedRegion:
    data = np.zeros((32, 96), dtype=np.bool_)
    data[4:20, x_min:x_max] = True
    mask = Mask(data, 96, 32)
    return ObservedRegion(
        region_id,
        mask,
        mask.bounding_box(),
        0.9,
        (f"{region_id}-p",),
        claims=claims,
        visual_embedding_ref=embedding_ref,
    )


# Constrói um embedding unitário na direção pedida, para que o desempate seja
# escolhido pelo teste em vez de emergir de um fake.
def _embedding(embedding_id: str, region_id: str, direction: list[float]) -> VisualEmbedding:
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    return VisualEmbedding(
        embedding_id=embedding_id,
        region_id=region_id,
        vector=tuple(vector.tolist()),
        dimension=len(direction),
        pooling_method="pixel_nearest_highres",
        feature_resolution="4x4",
        model_id="dinov2",
        normalized=True,
    )


def _prior_region(region_id: str, box: BoundingBox, concept: str, **kwargs) -> PriorRegion:
    return PriorRegion(region_id=region_id, box=box, concept=concept, **kwargs)


class TestBoundingBoxIou:
    def test_caixas_identicas_dao_um(self) -> None:
        box = BoundingBox(0.0, 0.0, 10.0, 10.0)
        assert box.iou(box) == pytest.approx(1.0)

    def test_caixas_disjuntas_dao_zero(self) -> None:
        assert BoundingBox(0.0, 0.0, 5.0, 5.0).iou(BoundingBox(6.0, 6.0, 10.0, 10.0)) == 0.0

    def test_caixas_que_so_encostam_dao_zero(self) -> None:
        assert BoundingBox(0.0, 0.0, 5.0, 5.0).iou(BoundingBox(5.0, 0.0, 10.0, 5.0)) == 0.0

    def test_sobreposicao_parcial(self) -> None:
        # 10x10 e 10x10 deslocadas de 5 em x: interseção 5x10=50, união 150.
        left, right = BoundingBox(0.0, 0.0, 10.0, 10.0), BoundingBox(5.0, 0.0, 15.0, 10.0)
        assert left.iou(right) == pytest.approx(50.0 / 150.0)

    def test_e_simetrica(self) -> None:
        left, right = BoundingBox(0.0, 0.0, 10.0, 10.0), BoundingBox(3.0, 3.0, 12.0, 8.0)
        assert left.iou(right) == pytest.approx(right.iou(left))


class TestSelectRegionPrior:
    def test_modo_desligado_nao_entrega_nada(self) -> None:
        prior = ScenePrior((_prior_region("region-a", BoundingBox(0.0, 0.0, 10.0, 10.0), "pallet"),))
        assert select_region_prior(BoundingBox(0.0, 0.0, 10.0, 10.0), prior) is None

    def test_prior_ausente_nao_entrega_nada(self) -> None:
        assert (
            select_region_prior(
                BoundingBox(0.0, 0.0, 10.0, 10.0), None, mode=TemporalPriorMode.BOX_OVERLAP
            )
            is None
        )

    def test_sobreposicao_acima_do_limiar_entrega_o_conceito(self) -> None:
        prior = ScenePrior(
            (_prior_region("region-a", BoundingBox(0.0, 0.0, 10.0, 10.0), "pallet", category="wood"),)
        )
        matched = select_region_prior(
            BoundingBox(1.0, 1.0, 11.0, 11.0), prior, mode=TemporalPriorMode.BOX_OVERLAP
        )
        assert matched is not None
        assert (matched.concept, matched.category, matched.source_region_id) == (
            "pallet",
            "wood",
            "region-a",
        )
        assert matched.overlap > 0.3

    def test_sobreposicao_abaixo_do_limiar_nao_entrega_nada(self) -> None:
        prior = ScenePrior((_prior_region("region-a", BoundingBox(0.0, 0.0, 10.0, 10.0), "pallet"),))
        assert (
            select_region_prior(
                BoundingBox(9.0, 9.0, 19.0, 19.0), prior, mode=TemporalPriorMode.BOX_OVERLAP
            )
            is None
        )

    def test_embedding_desempata_candidatos_sobrepostos(self) -> None:
        # Duas regiões anteriores cobrem a mesma área; só o embedding as separa.
        near = tuple(np.asarray([1.0, 0.0], dtype=np.float64).tolist())
        far = tuple(np.asarray([0.0, 1.0], dtype=np.float64).tolist())
        prior = ScenePrior(
            (
                _prior_region("region-a", BoundingBox(0.0, 0.0, 10.0, 10.0), "wall", embedding=far),
                _prior_region("region-b", BoundingBox(0.0, 0.0, 10.0, 10.0), "pallet", embedding=near),
            )
        )
        matched = select_region_prior(
            BoundingBox(0.0, 0.0, 10.0, 10.0),
            prior,
            mode=TemporalPriorMode.BOX_OVERLAP,
            embedding=near,
        )
        assert matched is not None
        assert matched.concept == "pallet"
        assert matched.similarity == pytest.approx(1.0)

    def test_sem_embedding_o_desempate_e_deterministico(self) -> None:
        prior = ScenePrior(
            (
                _prior_region("region-b", BoundingBox(0.0, 0.0, 10.0, 10.0), "pallet"),
                _prior_region("region-a", BoundingBox(0.0, 0.0, 10.0, 10.0), "wall"),
            )
        )
        box = BoundingBox(0.0, 0.0, 10.0, 10.0)
        first = select_region_prior(box, prior, mode=TemporalPriorMode.BOX_OVERLAP)
        second = select_region_prior(box, prior, mode=TemporalPriorMode.BOX_OVERLAP)
        assert first == second
        assert first is not None and first.similarity is None

    def test_limiar_fora_do_intervalo_e_rejeitado(self) -> None:
        with pytest.raises(ValueError, match="min_overlap"):
            select_region_prior(BoundingBox(0.0, 0.0, 10.0, 10.0), None, min_overlap=1.5)


class TestPriorFrom:
    def test_reconciliada_vence_a_primaria(self) -> None:
        region = _region(
            "region-a",
            0,
            20,
            (
                _claim("smooth surface"),
                _claim("wall", role=HypothesisRole.RECONCILED),
            ),
        )
        prior = prior_from(_observation(regions=(region,)))
        assert [item.concept for item in prior.regions] == ["wall"]

    def test_alternativa_nunca_vira_prior(self) -> None:
        region = _region(
            "region-a",
            0,
            20,
            (_claim("pallet"), _claim("crate", role=HypothesisRole.ALTERNATIVE)),
        )
        prior = prior_from(_observation(regions=(region,)))
        assert [item.concept for item in prior.regions] == ["pallet"]

    def test_contexto_estrutural_tambem_entra(self) -> None:
        published = _region("region-a", 0, 20, (_claim("pallet"),))
        structural = _region("region-b", 40, 60, (_claim("wall"),))
        prior = prior_from(
            _observation(regions=(published,), structural_context=(structural,))
        )
        assert sorted(item.concept for item in prior.regions) == ["pallet", "wall"]

    def test_regiao_sem_identidade_e_omitida(self) -> None:
        assert prior_from(_observation(regions=(_region("region-a", 0, 20),))).regions == ()

    def test_embedding_e_carregado_quando_existe(self) -> None:
        region = _region("region-a", 0, 20, (_claim("pallet"),), embedding_ref="visual-region-a")
        embedding = _embedding("visual-region-a", "region-a", [1.0, 0.0, 0.0, 0.0])
        prior = prior_from(_observation(regions=(region,)), (embedding,))
        assert prior.regions[0].embedding == embedding.vector


class TestMatchScenePrior:
    def test_desligado_nao_casa_nada(self) -> None:
        prior = ScenePrior((_prior_region("region-a", BoundingBox(0.0, 4.0, 20.0, 20.0), "pallet"),))
        assert match_scene_prior(
            (_region("region-a", 0, 20, (_claim("wall"),)),),
            prior,
            (),
            MultimodalReasoningConfig(),
        ) == ()

    def test_casa_a_regiao_sobreposta_e_ignora_a_distante(self) -> None:
        config = MultimodalReasoningConfig(temporal_prior_mode=TemporalPriorMode.BOX_OVERLAP.value)
        overlapping = _region("region-near", 0, 20)
        distant = _region("region-far", 70, 90)
        prior = prior_from(
            _observation(regions=(_region("region-old", 0, 20, (_claim("pallet"),)),))
        )
        assignments = match_scene_prior((overlapping, distant), prior, (), config)
        assert [item.region_id for item in assignments] == ["region-near"]
        assert assignments[0].prior.concept == "pallet"

    def test_prior_vazio_nao_casa_nada(self) -> None:
        config = MultimodalReasoningConfig(temporal_prior_mode=TemporalPriorMode.BOX_OVERLAP.value)
        assert match_scene_prior((_region("region-a", 0, 20),), ScenePrior(), (), config) == ()


class TestPriorStats:
    def _assignment(self, region_id: str, concept: str) -> PriorAssignment:
        return PriorAssignment(
            region_id=region_id,
            prior=PriorHypothesis(concept=concept, source_region_id="region-old", overlap=0.8),
        )

    def test_sem_casamento_reporta_apenas_se_estava_ligado(self) -> None:
        stats = _prior_stats(_observation(), (), True)
        assert (stats.regions_with_prior, stats.applied, stats.mean_overlap) == (0, True, None)

    def test_conta_eco_e_contradicao(self) -> None:
        echoed = _region("region-a", 0, 20, (_claim("pallet"),))
        contradicted = _region("region-b", 30, 50, (_claim("wall"),))
        observation = _observation(regions=(echoed, contradicted))
        stats = _prior_stats(
            observation,
            (self._assignment("region-a", "pallet"), self._assignment("region-b", "pallet")),
            True,
        )
        assert (stats.regions_with_prior, stats.echoed, stats.contradicted) == (2, 1, 1)
        assert stats.mean_overlap == pytest.approx(0.8)

    def test_eco_ignora_caixa_e_espacamento(self) -> None:
        region = _region("region-a", 0, 20, (_claim("Wooden  Pallet"),))
        stats = _prior_stats(
            _observation(regions=(region,)), (self._assignment("region-a", "wooden pallet"),), True
        )
        assert stats.echoed == 1


class TestConfiguracaoDosBracos:
    def test_default_mantem_o_prior_desligado(self) -> None:
        assert MultimodalReasoningConfig().temporal_prior_mode == TemporalPriorMode.DISABLED.value

    def test_modo_desconhecido_e_rejeitado(self) -> None:
        with pytest.raises(ValueError, match="temporal_prior_mode"):
            MultimodalReasoningConfig(temporal_prior_mode="sometimes")

    def test_sobreposicao_fora_do_intervalo_e_rejeitada(self) -> None:
        with pytest.raises(ValueError, match="temporal_prior_min_overlap"):
            MultimodalReasoningConfig(temporal_prior_min_overlap=-0.1)

    def test_ligar_o_prior_muda_o_fingerprint_da_sub_config(self) -> None:
        from visual_perception.application.support import fingerprint_of

        off = MultimodalReasoningConfig()
        on = dataclasses.replace(off, temporal_prior_mode=TemporalPriorMode.BOX_OVERLAP.value)
        assert fingerprint_of(off) != fingerprint_of(on)
