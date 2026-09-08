"""Testes do controle estrutural de contexto de cena no raciocínio de região (#212).

A limitação 1 de ``docs/known-limitations.md`` registra que proibir textualmente
a repetição da cena não impede o modelo de repetir: o prompt já diz que as
claims descrevem a cena e "must not be repeated as the label", e regiões
continuavam saindo rotuladas com a cena inteira. Estes testes fixam a garantia
estrutural que substituiu a instrução — o contexto simplesmente não entra no
request — e, principalmente, que os dois modos diferem *apenas* nisso.
"""

from __future__ import annotations

import numpy as np

from fixtures import default_config, payload_with_blobs
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.application.region_views import build_region_views
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform, Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import EvidenceSlot, SubjectEmphasis
from visual_perception.domain.region_reasoning import (
    RegionReasoningRequest,
    RegionView,
    SceneContextMode,
    select_region_scene_claims,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import _region_prompt
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Constrói uma região mínima com máscara quadrada, no mesmo padrão dos demais
# testes de interpretação.
def _region(region_id: str = "region-a") -> ObservedRegion:
    data = np.zeros((32, 32), dtype=np.bool_)
    data[4:10, 4:10] = True
    mask = Mask(data, 32, 32)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",))


# Constrói um contexto de cena com as kinds que atravessam a fronteira de região.
def _scene() -> SceneContext:
    provenance = ModelProvenance(stage="scene_context", producer="fake", config_fingerprint="fp")
    evidence = (Evidence(description="raw multimodal scene response"),)
    return SceneContext(
        claims=(
            SemanticClaim(
                ClaimKind.SCENE_TYPE, "corridor", ConfidenceScore(0.82, "fake"), evidence, provenance
            ),
            SemanticClaim(ClaimKind.ENVIRONMENT, "indoor", None, evidence, provenance),
        )
    )


# Captura o request que chegou ao reasoner sob o modo informado.
def _request_under(mode: str) -> object:
    seen: list[object] = []
    reasoner = FakeMultimodalReasoner(
        region_response_fn=lambda request: seen.append(request) or {"label": "floor", "kind": "stuff"}
    )
    region = _region()
    image = payload_with_blobs()
    views = build_region_views((region,), image, default_config())
    interpret_regions(
        (region,),
        image,
        views,
        _scene(),
        reasoner,
        MultimodalReasoningConfig(scene_context_mode=mode),
    )
    return seen[0]


# A garantia estrutural: em local_first o contexto não atravessa a fronteira.
# Uma instrução de prompt pode ser ignorada pelo modelo; uma tupla vazia não.
def test_local_first_keeps_scene_claims_out_of_the_request() -> None:
    """Em local-first a região chega ao reasoner sem nenhuma claim de cena."""
    request = _request_under("local_first")
    assert request.scene_claims == ()  # type: ignore[attr-defined]


# O modo assistido continua existindo e funcionando: o default mudou, a
# capacidade não foi removida.
def test_context_assisted_still_carries_the_scene_claims() -> None:
    """Em context-assisted as claims de cena continuam acompanhando a região."""
    request = _request_under("context_assisted")
    values = [claim.value for claim in request.scene_claims]  # type: ignore[attr-defined]
    assert values == ["corridor", "indoor"]


# O contexto de cena continua sendo analisado e permanece na observação: o modo
# governa o que atravessa a fronteira de região, não se a cena existe. Confundir
# as duas coisas faria "o modelo não viu a cena" parecer "a cena sumiu".
def test_local_first_selects_nothing_even_when_a_scene_exists() -> None:
    """A seleção devolve vazio por decisão de modo, não por ausência de cena."""
    scene = _scene()
    assert select_region_scene_claims(scene, mode=SceneContextMode.LOCAL_FIRST) == ()
    assert select_region_scene_claims(scene, mode=SceneContextMode.CONTEXT_ASSISTED) != ()
    assert scene.claims != ()


# Constrói um request mínimo com as mesmas views nos dois modos, variando só as
# claims de cena — é o que permite comparar os dois prompts diretamente.
def _prompt_request(scene_claims: tuple[SemanticClaim, ...]) -> RegionReasoningRequest:
    box = BoundingBox(0.0, 0.0, 4.0, 4.0)
    view = RegionView(
        slot=EvidenceSlot.FOREGROUND_DENSE,
        payload=payload_with_blobs(width=4, height=4),
        crop_box=box,
        transform=CoordinateTransform(1.0, 1.0, 0.0, 0.0),
        emphasis=SubjectEmphasis.ZERO_FILL,
    )
    return RegionReasoningRequest(
        region_id="region-a",
        region_box=box,
        image_width=32,
        image_height=32,
        views=(view,),
        scene_claims=scene_claims,
    )


# A propriedade que torna a ablation honesta. O módulo já mediu que o texto do
# prompt é variável de primeira ordem — trocar só a frase de enquadramento moveu
# o label ``door`` de 1/54 para 45/54 e de volta para 3/54. Se local_first
# mudasse qualquer outra parte do prompt, a comparação entre os modos estaria
# medindo duas variáveis ao mesmo tempo e não valeria nada.
def test_the_two_modes_differ_only_by_the_appended_scene_block() -> None:
    """O prompt de local-first é prefixo exato do de context-assisted."""
    local = _region_prompt(_prompt_request(()))
    assisted = _region_prompt(_prompt_request(_scene().claims))

    assert assisted.startswith(local)
    assert len(assisted) > len(local)
    assert "corridor" in assisted.removeprefix(local)
