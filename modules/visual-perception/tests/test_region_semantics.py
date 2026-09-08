"""Testes do estágio de interpretação semântica em nível de região (#165, #202, #203)."""

from __future__ import annotations

import dataclasses

import numpy as np

from fixtures import default_config, payload_with_blobs
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.application.region_views import build_region_views
from visual_perception.config import ModuleConfig, MultimodalReasoningConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import (
    IDENTITY_CLAIM_KINDS,
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Constrói uma ObservedRegion mínima com uma mask quadrada fixa, para não repetir
# esse setup em cada teste de interpret_regions abaixo.
def _region(region_id: str, width: int = 32, height: int = 32) -> ObservedRegion:
    data = np.zeros((height, width), dtype=np.bool_)
    data[4:10, 4:10] = True
    mask = Mask(data, width, height)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",))


# Constrói uma ObservedRegion a partir de uma mask arbitrária, para os testes de
# geometria irregular, fina, aninhada e encostada na borda.
def _region_from_mask(region_id: str, data: np.ndarray) -> ObservedRegion:
    height, width = data.shape
    mask = Mask(data, width, height)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",))


# Habilita o crop contextual, que fica desligado no default de MultiContextConfig.
# Existe para que os testes de evidência distinguível não dependam do perfil
# default continuar sendo o perfil reduzido.
def _config_with_context() -> ModuleConfig:
    config = default_config()
    return dataclasses.replace(
        config, multi_context=dataclasses.replace(config.multi_context, contextual_crop_enabled=True)
    )


# Pede explicitamente o canal visual de contexto. Existe porque o default do
# reasoner é local-first: um teste que exercita a view contextual precisa dizer
# isso, em vez de depender de um default que mudou de propósito.
def _reasoning_with_contextual_view() -> MultimodalReasoningConfig:
    return MultimodalReasoningConfig(
        region_views=("foreground_dense", "tight_crop", "contextual_crop")
    )


# Pede explicitamente o canal textual de contexto de cena. É o default hoje, mas
# um teste que exercita o contexto não deve depender disso: o default já mudou
# uma vez em função de medição e pode mudar de novo.
def _context_assisted() -> MultimodalReasoningConfig:
    return MultimodalReasoningConfig(scene_context_mode="context_assisted")


# Constrói uma claim de cena com proveniência completa; usada pelos testes da
# #202, que verificam que a claim atravessa a fronteira inteira e não achatada.
def _scene_claim(kind: ClaimKind, value: str, confidence: float | None = None) -> SemanticClaim:
    return SemanticClaim(
        kind,
        value,
        None if confidence is None else ConfidenceScore(confidence, "fake"),
        (Evidence(description="raw multimodal scene response"),),
        ModelProvenance(stage="scene_context", producer="fake", config_fingerprint="scene-fp"),
        role=HypothesisRole.PRIMARY if kind in IDENTITY_CLAIM_KINDS else None,
    )


# Verifica o caminho feliz: uma região válida recebe pelo menos um claim do tipo
# "label" a partir do reasoner fake.
def test_valid_region_receives_label_claims() -> None:
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())
    updated, failures = interpret_regions(
        (region,), image, views, None, FakeMultimodalReasoner(), MultimodalReasoningConfig()
    )
    assert failures == ()
    assert any(claim.kind.value == "label" for claim in updated[0].claims)


# Garante que interpret_regions só adiciona claims semânticos, nunca altera a
# geometria (mask/box/geometric_confidence) herdada da região de entrada.
def test_region_geometry_is_never_modified() -> None:
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())
    updated, _ = interpret_regions(
        (region,), image, views, None, FakeMultimodalReasoner(), MultimodalReasoningConfig()
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
        region_response_fn=lambda request: {
            "label": "box",
            "confidence": 0.6,
            "kind": "thing",
            "alternatives": [{"label": "crate", "confidence": 0.4}],
        }
    )
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())
    config = MultimodalReasoningConfig()
    updated, failures = interpret_regions((region,), image, views, None, reasoner, config)
    labels = {claim.value for claim in updated[0].claims if claim.kind.value == "label"}
    assert labels == {"box", "crate"}
    assert failures == ()


# Documenta o comportamento atual quando o mesmo reasoner malformado é usado para
# todas as regiões de uma chamada: como a falha é por reasoner (não por região),
# ambas as regiões falham juntas — ver test_one_failing_region_does_not_invalidate_others
# para o caso de isolamento por região.
def test_malformed_response_isolates_failure_without_dropping_other_regions() -> None:
    reasoner = FakeMultimodalReasoner(region_response_fn=lambda request: {"label": ""})
    good_region = _region("region-good")
    bad_region = _region("region-bad")
    image = payload_with_blobs()
    views = build_region_views((good_region, bad_region), image, default_config())
    updated, failures = interpret_regions(
        (good_region, bad_region), image, views, None, reasoner, MultimodalReasoningConfig()
    )
    assert len(failures) == 2  # ambas falham aqui porque o mesmo reasoner é malformado para todas
    assert len(updated) == 2


# Verifica que uma região sem claims (resposta vazia) ainda é reportada como
# sucesso (sem failures) e mantida no resultado, sem impedir a região boa.
def test_one_failing_region_does_not_invalidate_others() -> None:
    good_reasoner = FakeMultimodalReasoner()
    good_region = _region("region-good")
    image = payload_with_blobs()
    views = build_region_views((good_region,), image, default_config())

    updated, failures = interpret_regions(
        (good_region,), image, views, None, good_reasoner, MultimodalReasoningConfig()
    )
    assert failures == ()
    assert updated[0].claims


# Protege o critério central da #203: o reasoner recebe foreground e contexto
# como evidências separadas e identificáveis, e não um recorte único em que os
# dois estão misturados.
def test_request_carries_foreground_and_context_as_distinguishable_views() -> None:
    """A região chega ao reasoner com foreground e contexto distinguíveis por slot."""
    seen: list[object] = []
    reasoner = FakeMultimodalReasoner(
        region_response_fn=lambda request: seen.append(request) or {"label": "wall", "kind": "stuff"}
    )
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, _config_with_context())

    interpret_regions(
        (region,), image, views, None, reasoner, _reasoning_with_contextual_view()
    )

    request = seen[0]
    assert {view.slot for view in request.foreground_views} == {  # type: ignore[attr-defined]
        EvidenceSlot.FOREGROUND_DENSE,
        EvidenceSlot.TIGHT_CROP,
    }
    assert [view.slot for view in request.contextual_views] == [  # type: ignore[attr-defined]
        EvidenceSlot.CONTEXTUAL_CROP
    ]
    assert request.region_id == "region-a"  # type: ignore[attr-defined]


# Garante que a falha do contexto opcional não descarta a evidência válida de
# foreground: sem isso, um crop contextual degenerado silenciaria uma região que
# ainda tinha evidência local perfeitamente utilizável.
def test_missing_optional_context_view_keeps_valid_foreground_evidence() -> None:
    """Contexto ausente reduz as views, nunca invalida a interpretação da região."""
    seen: list[object] = []
    reasoner = FakeMultimodalReasoner(
        region_response_fn=lambda request: seen.append(request) or {"label": "door", "kind": "thing"}
    )
    region = _region("region-a")
    image = payload_with_blobs()
    # Perfil sem crop contextual: só as views de foreground existem.
    views = build_region_views((region,), image, default_config())

    updated, failures = interpret_regions(
        (region,), image, views, None, reasoner, MultimodalReasoningConfig()
    )

    assert failures == ()
    assert seen[0].contextual_views == ()  # type: ignore[attr-defined]
    assert seen[0].foreground_views  # type: ignore[attr-defined]
    assert any(claim.value == "door" for claim in updated[0].claims)


# Protege o critério "no background-only pixel is presented as foreground
# evidence": a view de foreground zera tudo que está fora da mask, mesmo quando
# ``masked_tight_crop`` está desligado para o slot de embedding.
def test_foreground_view_contains_no_background_pixel() -> None:
    """A view de foreground zera todo pixel fora da mask da região."""
    data = np.zeros((32, 32), dtype=np.bool_)
    # Diagonal: a caixa cobre 8x8, mas só 8 pixels pertencem à região.
    for offset in range(8):
        data[4 + offset, 4 + offset] = True
    region = _region_from_mask("region-diag", data)
    image = payload_with_blobs(blobs=((0, 0, 32, 32, (10, 200, 10)),))
    views = build_region_views((region,), image, default_config())

    foreground = next(
        view for view in views["region-diag"] if view.slot is EvidenceSlot.FOREGROUND_DENSE
    )
    window = data[4:12, 4:12]
    assert bool(np.all(foreground.payload.pixels[~window] == 0))
    assert bool(np.all(foreground.payload.pixels[window] == (10, 200, 10)))


# Verifica que masks irregulares, finas, aninhadas e encostadas na borda
# atravessam a preparação de evidência com a geometria exata: cada view descreve
# a caixa que diz descrever, e regiões aninhadas não compartilham recorte.
def test_irregular_nested_and_border_masks_preserve_their_exact_geometry() -> None:
    """Cada view recorta exatamente a sua própria caixa, inclusive na borda."""
    thin = np.zeros((32, 32), dtype=np.bool_)
    thin[:, 31] = True  # coluna de 1 pixel encostada na borda direita
    outer = np.zeros((32, 32), dtype=np.bool_)
    outer[4:20, 4:20] = True
    inner = np.zeros((32, 32), dtype=np.bool_)
    inner[8:12, 8:12] = True  # aninhada dentro de outer

    regions = (
        _region_from_mask("region-thin", thin),
        _region_from_mask("region-outer", outer),
        _region_from_mask("region-inner", inner),
    )
    image = payload_with_blobs()
    views = build_region_views(regions, image, _config_with_context())

    for region in regions:
        for view in views[region.region_id]:
            expected_width = int(view.crop_box.x_max) - int(view.crop_box.x_min)
            expected_height = int(view.crop_box.y_max) - int(view.crop_box.y_min)
            assert view.payload.width == expected_width
            assert view.payload.height == expected_height

    thin_view = next(v for v in views["region-thin"] if v.slot is EvidenceSlot.FOREGROUND_DENSE)
    assert thin_view.payload.width == 1

    outer_box = next(
        v for v in views["region-outer"] if v.slot is EvidenceSlot.FOREGROUND_DENSE
    ).crop_box
    inner_box = next(
        v for v in views["region-inner"] if v.slot is EvidenceSlot.FOREGROUND_DENSE
    ).crop_box
    assert outer_box != inner_box


# Garante que uma região sem nenhuma view de foreground falha localmente, com a
# região preservada, em vez de ser interpretada a partir do seu entorno.
def test_region_without_foreground_view_fails_locally() -> None:
    """Sem evidência de foreground a região falha isolada e mantém sua geometria."""
    good = _region("region-good")
    orphan = _region("region-orphan")
    image = payload_with_blobs()
    views = build_region_views((good, orphan), image, default_config())
    del views["region-orphan"]

    updated, failures = interpret_regions(
        (good, orphan), image, views, None, FakeMultimodalReasoner(), MultimodalReasoningConfig()
    )

    assert [failure.region_id for failure in failures] == ["region-orphan"]
    assert updated[1].mask is orphan.mask
    assert updated[0].claims


# Protege o critério central da #202: as claims de cena chegam inteiras ao
# raciocínio de região — com kind, confiança e proveniência próprias — em vez de
# achatadas na primeira string de descrição, como fazia o resumo anterior.
def test_scene_claims_reach_the_reasoner_structured_with_their_provenance() -> None:
    """O contexto de cena atravessa a fronteira como claims, não como um resumo."""
    seen: list[object] = []
    reasoner = FakeMultimodalReasoner(
        region_response_fn=lambda request: seen.append(request) or {"label": "floor", "kind": "stuff"}
    )
    scene = SceneContext(
        claims=(
            _scene_claim(ClaimKind.SCENE_TYPE, "corridor", 0.82),
            _scene_claim(ClaimKind.ENVIRONMENT, "indoor"),
            _scene_claim(ClaimKind.LAYOUT, "a narrow corridor running away from the camera"),
            _scene_claim(ClaimKind.NAVIGABILITY, "open path ahead"),
            # Inventário de objetos não é contexto ambiental e não atravessa a
            # fronteira desde a #202: era por ali que uma "suitcase" alucinada,
            # que era o próprio rig, condicionava a interpretação de cada região.
            _scene_claim(ClaimKind.SCENE_DESCRIPTION, "There is a suitcase in the foreground."),
            _scene_claim(ClaimKind.ATTRIBUTE, "suitcase"),
            _scene_claim(ClaimKind.HAZARD, "wet floor"),
            # Uma claim de região no nível de cena não é contexto de cena e não
            # pode atravessar: promovê-la seria dar à região um label que
            # nenhuma evidência local sustenta.
            _scene_claim(ClaimKind.LABEL, "corridor"),
        )
    )
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())

    interpret_regions((region,), image, views, scene, reasoner, _context_assisted())

    claims = seen[0].scene_claims  # type: ignore[attr-defined]
    assert [claim.kind for claim in claims] == [
        ClaimKind.SCENE_TYPE,
        ClaimKind.ENVIRONMENT,
        ClaimKind.LAYOUT,
        ClaimKind.NAVIGABILITY,
    ]
    scene_type = claims[0]
    assert scene_type.confidence is not None and scene_type.confidence.value == 0.82
    assert scene_type.provenance.stage == "scene_context"
    assert claims[1].confidence is None  # descrição livre continua sem score


# Verifica que claims de cena contraditórias permanecem lado a lado e
# distinguíveis na entrada do raciocínio: a seleção restringe kinds, nunca
# escolhe entre hipóteses de cena.
def test_contradictory_scene_claims_remain_distinguishable() -> None:
    """Duas hipóteses de scene_type coexistem na entrada do raciocínio de região."""
    seen: list[object] = []
    reasoner = FakeMultimodalReasoner(
        region_response_fn=lambda request: seen.append(request) or {"label": "wall", "kind": "stuff"}
    )
    scene = SceneContext(
        claims=(
            _scene_claim(ClaimKind.SCENE_TYPE, "corridor", 0.55),
            _scene_claim(ClaimKind.SCENE_TYPE, "field", 0.45),
        )
    )
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())

    interpret_regions((region,), image, views, scene, reasoner, _context_assisted())

    values = [claim.value for claim in seen[0].scene_claims]  # type: ignore[attr-defined]
    assert values == ["corridor", "field"]


# Garante que evidência de cena ausente ou parcial não invalida a interpretação
# da região: a região continua interpretável apenas pela sua própria evidência.
def test_missing_scene_context_still_interprets_the_region() -> None:
    """Sem contexto de cena, a região ainda é interpretada pela evidência local."""
    region = _region("region-a")
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())

    updated, failures = interpret_regions(
        (region,), image, views, SceneContext(), FakeMultimodalReasoner(), MultimodalReasoningConfig()
    )

    assert failures == ()
    assert any(claim.kind is ClaimKind.LABEL for claim in updated[0].claims)
