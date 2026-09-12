"""Regressões de discovery ampla, grounding, conectividade e abstenção."""

from dataclasses import replace

import numpy as np
import pytest

from fixtures import image_observation
from fixtures_ports import default_ports
from visual_perception import (
    GroundingPrediction,
    GroundingStatus,
    ImagePayload,
    Mask,
    SemanticGroundingConfig,
    build_spatial_footprints,
    deserialize_observation,
    ground_regions,
    run_canonical_pipeline,
    serialize_observation,
)
from visual_perception.config import ModuleConfig
from visual_perception.domain.geometry import BoundingBox
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantics import ClaimKind, Evidence, HypothesisRole, RegionKind, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext, VisualObservation
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner

_PROVENANCE = ModelProvenance("semantic_grounding", "fixture", "grounding-test/1")


# Cria máscaras no mesmo frame para testar transformações espaciais completas.
def _mask(*boxes: tuple[int, int, int, int]) -> Mask:
    """Rasteriza retângulos em um frame de 32×32 pixels."""
    data = np.zeros((32, 32), dtype=np.bool_)
    for x0, y0, x1, y1 in boxes:
        data[y0:y1, x0:x1] = True
    return Mask(data, 32, 32)


# Produz uma região com claim independente da máscara que a originou.
def _region(mask: Mask, *, name: str = "region-a", kind: RegionKind = RegionKind.THING) -> ObservedRegion:
    """Cria a região reconhecida, preservando a geometria de discovery."""
    claim = SemanticClaim(
        ClaimKind.LABEL, "door", None, (Evidence("Objeto reconhecido na região."),),
        _PROVENANCE, role=HypothesisRole.PRIMARY, region_kind=kind,
    )
    return ObservedRegion(name, mask, mask.bounding_box(), 0.9, (f"{name}-proposal",), (claim,))


# Substitui modelos por predições espaciais controladas; a validação, ownership
# e serialização reais continuam executando nos testes.
class Grounder:
    """Backend de fixture que segmenta o objeto indicado pelo teste."""

    # Recebe geometrias específicas sem inferir máscaras da própria discovery.
    def __init__(self, mask: Mask | None, box: BoundingBox | None = None) -> None:
        """Guarda a máscara independente e a localização do conceito."""
        self.mask, self.box = mask, box

    # Retorna a predição pelo mesmo contract usado pelo backend real.
    def ground(self, image, requests, config):
        """Fornece máscara nova ou falha explícita a cada região."""
        return tuple(GroundingPrediction(
            item.region_id, item.concept, self.mask,
            self.box or (None if self.mask is None or self.mask.is_empty else self.mask.bounding_box()),
            provenance=(_PROVENANCE,),
            status=GroundingStatus.FAILED if self.mask is None else GroundingStatus.REFINED,
            reason="Segmentação indisponível para esta hipótese." if self.mask is None else None,
        ) for item in requests)


# Executa o estágio real com uma imagem válida, evitando repetir composição.
def _ground(region, grounder, *, area_masks=None, config=None):
    """Retorna a região após a fronteira pública de grounding."""
    payload = ImagePayload(np.zeros((32, 32, 3), dtype=np.uint8), 32, 32)
    return ground_regions((region,), payload, grounder, config or SemanticGroundingConfig(), area_masks)[0]


# A porta reconhecida em uma região com parede perde apenas o footprint errado;
# a claim e os pixels usados para discovery continuam intactos para auditoria.
def test_door_grounding_reduces_wall_pixels_and_round_trips() -> None:
    """A geometria semântica e a de discovery sobrevivem distintas ao JSON."""
    raw, door = _mask((2, 2, 29, 28)), _mask((9, 5, 17, 25))
    before = _region(raw)
    region = _ground(before, Grounder(door))
    assert region.mask is raw and region.claims == before.claims and region.box == before.box
    assert region.grounding.semantic_mask == door
    assert not region.grounding.semantic_mask.data[10, 25]
    assert region.grounding.diagnostics["area_ratio"] == pytest.approx(door.area() / raw.area())
    observation = VisualObservation(image_observation().source, 32, 32, SceneContext(), (region,), ())
    assert deserialize_observation(serialize_observation(observation)) == observation


# O fluxo canônico deve chamar grounding depois que há um conceito; testar só
# o helper de clipping deixaria o stage ausente da execução real passar.
def test_canonical_pipeline_calls_grounding_after_recognition() -> None:
    """Discovery ampla do fake vira footprint menor sem alterar o label."""
    from fixtures import payload_with_blobs

    payload = payload_with_blobs(blobs=((2, 2, 25, 27, (180, 20, 20)),))
    ports = replace(default_ports(), semantic_grounder=Grounder(_mask((9, 5, 17, 25))),
        multimodal_reasoner=FakeMultimodalReasoner(region_response_fn=lambda request: {"label": "door", "kind": "thing"}))
    result = run_canonical_pipeline(image_observation(), payload, ModuleConfig(), ports)
    assert result.observation.regions
    for region in result.observation.regions:
        assert primary_label_claim(region).value == "door"
        assert region.grounding.status is GroundingStatus.REFINED
        assert region.grounding.semantic_mask.area() < region.mask.area()


# Não se elege o maior componente: o suporte localizado deve sustentar a parte
# menor, enquanto a parede maior é rejeitada mesmo com mais pixels.
def test_thing_uses_localized_support_instead_of_largest_component() -> None:
    """Um componente maior distante não herda a identidade do objeto."""
    region = _region(_mask((0, 0, 32, 32)))
    candidate = _mask((1, 2, 8, 29), (20, 11, 25, 18))
    grounded = _ground(region, Grounder(candidate, BoundingBox(0, 0, 32, 30)))
    assert grounded.grounding.semantic_mask == _mask((20, 11, 25, 18))
    assert len(grounded.grounding.diagnostics["components"]) == 2


# A mesma geometria descontínua é válida como superfície extensa de stuff.
def test_stuff_allows_extensive_disconnected_support() -> None:
    """Stuff não é forçado à política de identidade de um objeto contável."""
    candidate = _mask((1, 2, 8, 29), (20, 11, 25, 18))
    region = _ground(_region(_mask((0, 0, 32, 32)), kind=RegionKind.STUFF), Grounder(candidate))
    assert region.grounding.semantic_mask == candidate
    assert region.grounding.diagnostics["component_count_after"] == 2


# Ownership corta uma faixa sem remover o suporte principal; apenas o lado
# que ainda contém a âncora semântica pode continuar associado fortemente.
def test_ownership_fragment_keeps_only_component_with_original_support() -> None:
    """Restos de parede à esquerda deixam de receber door depois de ownership."""
    large = _ground(_region(_mask((2, 4, 29, 27))), Grounder(_mask((2, 4, 29, 27))))
    blocker = _ground(_region(_mask((10, 4, 13, 27)), name="region-b"), Grounder(_mask((10, 4, 13, 27))))
    footprints = build_spatial_footprints((large, blocker), _mask((0, 0, 32, 32)))
    result = next(item for item in footprints if item.region_id == large.region_id)
    assert result.mask == _mask((13, 4, 29, 27))
    assert result.diagnostics["component_count_after_ownership"] == 2
    assert result.diagnostics["fragment_removed_pixel_count"] > 0


# Uma âncora tomada por outra região não pode ser transferida ao resto maior.
def test_ownership_losing_support_abstains_and_retains_claim() -> None:
    """Perder o suporte principal torna o footprint forte indisponível."""
    large = _ground(_region(_mask((2, 4, 29, 27))), Grounder(_mask((2, 4, 29, 27))))
    blocker = _ground(_region(_mask((14, 4, 18, 27)), name="region-b"), Grounder(_mask((14, 4, 18, 27))))
    result = next(item for item in build_spatial_footprints((large, blocker), _mask((0, 0, 32, 32))) if item.region_id == large.region_id)
    assert result.mask is None
    assert result.diagnostics["semantic_grounding_status"] == "mask_fragmented"
    assert primary_label_claim(large).value == "door"


# Falha ou ausência de backend nunca fazem uma região contável recair na mask.
@pytest.mark.parametrize("grounder", [None, Grounder(None)])
def test_grounding_failure_has_no_strong_discovery_fallback(grounder) -> None:
    """A claim sobrevive e o footprint espacial fica ausente."""
    region = _ground(_region(_mask((2, 2, 28, 28))), grounder)
    footprint, = build_spatial_footprints((region,), _mask((0, 0, 32, 32)), stuff_discovery_fallback=True)
    assert footprint.mask is None and not footprint.strong
    assert primary_label_claim(region).value == "door"


# Fallback de stuff é opt-in e continua tentativo, sem simular grounding.
def test_stuff_fallback_is_explicitly_tentative() -> None:
    """Geometria extensa sem grounding não ganha força por ser stuff."""
    region = _ground(_region(_mask((2, 2, 28, 28)), kind=RegionKind.STUFF), None)
    footprint, = build_spatial_footprints((region,), _mask((0, 0, 32, 32)), stuff_discovery_fallback=True)
    assert footprint.mask == region.mask and not footprint.strong
    assert footprint.diagnostics["stuff_tentative_fallback"]


# Mesmo um segmentador que extrapole a lente e o rig deve passar pela fronteira
# de validade; não se modifica o RGB entregue ao modelo para conseguir isso.
def test_grounding_preserves_valid_area_and_ego_exclusion() -> None:
    """Grounding nunca reintroduz pixels de fora da lente ou do ego vehicle."""
    valid, ego = _mask((4, 4, 28, 28)), _mask((4, 24, 28, 28))
    region = _ground(_region(_mask((0, 0, 32, 32))), Grounder(_mask((0, 0, 32, 32))), area_masks=ImageAreaMasks(valid, ego))
    accepted = region.grounding.semantic_mask.data
    assert not (accepted & ~valid.data).any() and not (accepted & ego.data).any()


# Mínimo explícito rejeita footprint sem utilidade, registrando razão sem
# perder a máscara bruta nem promover a descoberta como substituta.
def test_too_small_refinement_and_inconsistent_segmentation_are_observable() -> None:
    """Falhas de tamanho e de alinhamento espacial preservam a hipótese."""
    region = _region(_mask((2, 2, 15, 28)))
    tiny = _ground(region, Grounder(_mask((4, 4, 5, 5))), config=SemanticGroundingConfig(min_mask_area=2))
    outside = _ground(region, Grounder(_mask((20, 4, 28, 20))))
    assert tiny.grounding.status is GroundingStatus.TOO_SMALL
    assert outside.grounding.status is GroundingStatus.INCONSISTENT
    assert tiny.grounding.semantic_mask is outside.grounding.semantic_mask is None


# Boxes independentes de stuff não autorizam o vão entre elas: a box de
# extensão serve para diagnóstico, e não substitui as localizações reais.
def test_stuff_multiple_prompt_boxes_do_not_ground_the_gap() -> None:
    """Preserva as boxes e rejeita pixels fora de qualquer prompt localizado."""
    region = _region(_mask((0, 0, 32, 32)), kind=RegionKind.STUFF)
    backend = Grounder(_mask((0, 0, 32, 32)))
    original = backend.ground

    # Modela um segmentador que preenche também o intervalo entre duas boxes.
    def predictions(image, requests, config):
        """Entrega suporte bruto amplo e duas localizações de superfície."""
        return tuple(replace(item, prompt_boxes=(BoundingBox(1, 2, 8, 29), BoundingBox(20, 11, 25, 18)))
            for item in original(image, requests, config))

    backend.ground = predictions
    grounded = _ground(region, backend)
    assert grounded.grounding.semantic_mask == _mask((1, 2, 8, 29), (20, 11, 25, 18))
    observation = VisualObservation(image_observation().source, 32, 32, SceneContext(), (grounded,), ())
    assert deserialize_observation(serialize_observation(observation)) == observation


# Falhas de carregamento reais precisam liberar o lifecycle e se tornar uma
# abstenção da capacidade, sem escapar como erro de IO ou mudar a claim.
def test_model_load_failure_releases_lifecycle_and_preserves_claim(monkeypatch) -> None:
    """O port real transforma checkpoint ausente em grounding indisponível."""
    from visual_perception.application.lifecycle import ModelLifecycleManager
    from visual_perception.infrastructure.adapters.semantic_grounding_backend import (
        RealSemanticGroundingAdapter,
    )

    lifecycle = ModelLifecycleManager()
    backend = RealSemanticGroundingAdapter(lifecycle)
    released = []

    # Simula ausência dos pesos no ponto em que o adapter consultaria o cache.
    def fail(*args):
        """Falha sem carregar modelo ou depender da rede."""
        raise OSError("Checkpoint unavailable")

    monkeypatch.setattr(backend, "_detect", fail)
    monkeypatch.setattr(lifecycle, "release_active", lambda: released.append(True))
    grounded = _ground(_region(_mask((2, 2, 28, 28))), backend)
    assert released
    assert grounded.grounding.status is GroundingStatus.UNAVAILABLE
    assert grounded.grounding.semantic_mask is None
    assert primary_label_claim(grounded).value == "door"


# Detecta respostas incompletas para que ausência de predição não seja
# confundida com ausência de configuração do backend.
def test_missing_prediction_is_an_observable_backend_failure(monkeypatch) -> None:
    """Request omitido não herda geometria de discovery."""
    backend = Grounder(None)
    monkeypatch.setattr(backend, "ground", lambda *args: ())
    grounded = _ground(_region(_mask((2, 2, 28, 28))), backend)
    assert grounded.grounding.status is GroundingStatus.FAILED
    assert "omitiu" in grounded.grounding.diagnostics["reason"]
