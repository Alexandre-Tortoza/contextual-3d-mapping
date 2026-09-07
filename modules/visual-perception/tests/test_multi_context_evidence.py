"""Testes de contract e de extração da evidência multi-contexto de região (#193, #194)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fixtures import default_config, payload_with_blobs
from visual_perception.application.multi_context import (
    SCENE_EVIDENCE_REF,
    extract_region_evidence,
)
from visual_perception.config import LanguageEmbeddingConfig, ModuleConfig, MultiContextConfig
from visual_perception.domain.embeddings import (
    EmbeddingModality,
    EmbeddingSpace,
    IncompatibleEmbeddingSpaceError,
)
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform, Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import (
    EvidenceSlot,
    EvidenceState,
    RegionEvidenceSlot,
    evidence_from_dict,
    evidence_to_dict,
    require_comparable_slots,
    slots_in_state,
)
from visual_perception.domain.regions import ObservedRegion
from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder

_VISUAL_SPACE = EmbeddingSpace("dinov2", "facebook/dinov2-base", 768, EmbeddingModality.VISUAL_DENSE)
_LANGUAGE_SPACE = EmbeddingSpace("clip", "openai/clip", 512, EmbeddingModality.LANGUAGE_ALIGNED)


# Constrói uma região retangular simples alinhada a uma imagem 32x32, para
# que os testes de extração não dependam do resultado do region discovery.
def _region(region_id: str = "region-a", box: tuple[int, int, int, int] = (4, 4, 12, 12)) -> ObservedRegion:
    x_min, y_min, x_max, y_max = box
    data = np.zeros((32, 32), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    mask = Mask(data, 32, 32)
    return ObservedRegion(
        region_id=region_id,
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=0.8,
        contributing_proposal_ids=("proposal-1",),
    )


# Monta um slot disponível válido, para os testes que só variam um campo.
def _available(slot: EvidenceSlot, region_id: str = "region-a", **overrides: object) -> RegionEvidenceSlot:
    defaults: dict[str, object] = {
        "slot": slot,
        "region_id": region_id,
        "state": EvidenceState.AVAILABLE,
        "crop_box": BoundingBox(4.0, 4.0, 12.0, 12.0),
        "transform": CoordinateTransform.identity(),
        "preprocessing": "crop",
        "artifact_ref": f"language-{region_id}",
        "space": _LANGUAGE_SPACE,
    }
    defaults.update(overrides)
    return RegionEvidenceSlot(**defaults)  # type: ignore[arg-type]


def test_one_region_retains_several_slots_without_overwriting_them() -> None:
    region = dataclasses.replace(
        _region(),
        evidence=(
            _available(EvidenceSlot.TIGHT_CROP),
            _available(EvidenceSlot.CONTEXTUAL_CROP),
            RegionEvidenceSlot(
                slot=EvidenceSlot.SCENE_CONDITIONED,
                region_id="region-a",
                state=EvidenceState.MISSING,
                reason="slot disabled by configuration",
            ),
        ),
    )

    assert len(region.evidence) == 3
    assert region.evidence_for(EvidenceSlot.TIGHT_CROP) is not None
    assert region.evidence_for(EvidenceSlot.FOREGROUND_DENSE) is None
    assert slots_in_state(region.evidence, EvidenceState.MISSING)[0].slot is EvidenceSlot.SCENE_CONDITIONED


def test_the_same_slot_cannot_silently_overwrite_itself() -> None:
    with pytest.raises(ValueError, match="duplicate evidence"):
        dataclasses.replace(
            _region(),
            evidence=(_available(EvidenceSlot.TIGHT_CROP), _available(EvidenceSlot.TIGHT_CROP)),
        )


def test_view_specific_evidence_coexists_with_the_default_view() -> None:
    region = dataclasses.replace(
        _region(),
        evidence=(
            _available(EvidenceSlot.TIGHT_CROP),
            _available(EvidenceSlot.TIGHT_CROP, view_id="view-2"),
        ),
    )

    assert region.evidence_for(EvidenceSlot.TIGHT_CROP, view_id="view-2") is not None
    assert region.evidence_for(EvidenceSlot.TIGHT_CROP).view_id is None


def test_an_available_slot_must_identify_its_geometry_and_embedding_space() -> None:
    with pytest.raises(ValueError, match="artifact_ref and space"):
        _available(EvidenceSlot.TIGHT_CROP, artifact_ref=None)
    with pytest.raises(ValueError, match="crop_box and transform"):
        _available(EvidenceSlot.TIGHT_CROP, crop_box=None)


def test_a_missing_or_failed_slot_must_explain_itself_and_carry_no_artifact() -> None:
    with pytest.raises(ValueError, match="must explain itself"):
        RegionEvidenceSlot(
            slot=EvidenceSlot.CONTEXTUAL_CROP, region_id="region-a", state=EvidenceState.MISSING
        )
    with pytest.raises(ValueError, match="must not reference an artifact"):
        RegionEvidenceSlot(
            slot=EvidenceSlot.CONTEXTUAL_CROP,
            region_id="region-a",
            state=EvidenceState.FAILED,
            reason="encoder crashed",
            artifact_ref="language-region-a",
        )


def test_evidence_slots_must_belong_to_their_own_region() -> None:
    with pytest.raises(ValueError, match="belongs to"):
        dataclasses.replace(_region(), evidence=(_available(EvidenceSlot.TIGHT_CROP, "region-other"),))


def test_incompatible_embedding_spaces_are_never_compared_silently() -> None:
    compatible = (_available(EvidenceSlot.TIGHT_CROP), _available(EvidenceSlot.CONTEXTUAL_CROP))
    incompatible = (
        _available(EvidenceSlot.TIGHT_CROP),
        _available(EvidenceSlot.FOREGROUND_DENSE, space=_VISUAL_SPACE, artifact_ref="visual-region-a"),
    )

    assert require_comparable_slots(compatible) == _LANGUAGE_SPACE
    with pytest.raises(IncompatibleEmbeddingSpaceError, match="not comparable"):
        require_comparable_slots(incompatible)


def test_evidence_round_trips_through_serialization() -> None:
    slot = _available(
        EvidenceSlot.CONTEXTUAL_CROP,
        support_ratio=0.5,
        mask_ref="mask-region-a",
        view_id="view-2",
    )

    restored = evidence_from_dict(evidence_to_dict(slot))

    assert restored == slot


def test_malformed_serialized_evidence_is_rejected_instead_of_repaired() -> None:
    payload = evidence_to_dict(_available(EvidenceSlot.TIGHT_CROP))
    payload["artifact_ref"] = None

    with pytest.raises(ValueError, match="artifact_ref and space"):
        evidence_from_dict(payload)


# ---------------------------------------------------------------- extração (#194)


def test_extraction_produces_foreground_and_contextual_evidence_that_stay_distinguishable() -> None:
    config = dataclasses.replace(
        default_config(),
        multi_context=MultiContextConfig(contextual_crop_enabled=True, scene_conditioned_enabled=True),
    )
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    result = extract_region_evidence(
        (_region(),), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    )

    region = result.regions[0]
    foreground = region.evidence_for(EvidenceSlot.FOREGROUND_DENSE)
    contextual = region.evidence_for(EvidenceSlot.CONTEXTUAL_CROP)
    assert foreground is not None and foreground.is_foreground
    assert contextual is not None and not contextual.is_foreground
    assert contextual.crop_box.width > region.box.width
    assert result.failures == ()


def test_every_representation_stays_linked_to_its_region_and_source_coordinates() -> None:
    config = dataclasses.replace(
        default_config(), multi_context=MultiContextConfig(contextual_crop_enabled=True)
    )
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    result = extract_region_evidence(
        (_region("region-a"), _region("region-b", (16, 16, 24, 24))),
        payload,
        config,
        FakeLanguageAlignedEncoder(),
        feature_map=feature_map,
    )

    for region in result.regions:
        for slot in region.evidence:
            assert slot.region_id == region.region_id
        crop = region.evidence_for(EvidenceSlot.TIGHT_CROP)
        assert crop.transform.offset_x == crop.crop_box.x_min
        assert crop.crop_box.x_min >= 0.0
    assert {embedding.region_id for embedding in result.visual_embeddings} == {"region-a", "region-b"}


def test_global_context_never_changes_the_region_geometry() -> None:
    config = dataclasses.replace(
        default_config(), multi_context=MultiContextConfig(scene_conditioned_enabled=True)
    )
    payload = payload_with_blobs()
    original = _region()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    region = extract_region_evidence(
        (original,), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    ).regions[0]

    scene = region.evidence_for(EvidenceSlot.SCENE_CONDITIONED)
    assert region.mask == original.mask and region.box == original.box
    assert scene.artifact_ref == SCENE_EVIDENCE_REF
    assert (scene.crop_box.x_max, scene.crop_box.y_max) == (float(payload.width), float(payload.height))


def test_a_small_mask_still_produces_foreground_evidence() -> None:
    config = default_config()
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)
    tiny = _region("region-tiny", (5, 5, 7, 7))

    region = extract_region_evidence(
        (tiny,), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    ).regions[0]

    foreground = region.evidence_for(EvidenceSlot.FOREGROUND_DENSE)
    assert foreground.state is EvidenceState.AVAILABLE
    assert foreground.support_ratio == pytest.approx(1.0)


def test_a_masked_tight_crop_zeroes_the_background_it_was_configured_to_drop() -> None:
    masked_config = dataclasses.replace(
        default_config(), multi_context=MultiContextConfig(masked_tight_crop=True)
    )
    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)), (16, 16, 20, 20, (10, 200, 10))))
    feature_map = FakeDenseFeatureExtractor().extract(payload, masked_config.feature_extraction)
    encoder = FakeLanguageAlignedEncoder()

    masked = extract_region_evidence(
        (_region(),), payload, masked_config, encoder, feature_map=feature_map
    )
    plain = extract_region_evidence(
        (_region(),), payload, default_config(), encoder, feature_map=feature_map
    )

    assert masked.regions[0].evidence_for(EvidenceSlot.TIGHT_CROP).preprocessing.startswith("masked_crop")
    assert plain.regions[0].evidence_for(EvidenceSlot.TIGHT_CROP).preprocessing.startswith("crop")
    assert masked.regions[0].evidence_for(EvidenceSlot.TIGHT_CROP).mask_ref == "mask-region-a"


def test_missing_optional_context_is_explicit_and_keeps_valid_foreground_evidence() -> None:
    config = default_config()  # contextual e scene desligados por default
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    region = extract_region_evidence(
        (_region(),), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    ).regions[0]

    contextual = region.evidence_for(EvidenceSlot.CONTEXTUAL_CROP)
    assert contextual.state is EvidenceState.MISSING
    assert contextual.reason == "slot disabled by configuration"
    assert region.evidence_for(EvidenceSlot.FOREGROUND_DENSE).state is EvidenceState.AVAILABLE


def test_a_failure_in_one_optional_slot_preserves_the_other_valid_slots() -> None:
    class BrokenOnContext:
        """Encoder que falha somente no crop expandido, para isolar um slot."""

        def __init__(self) -> None:
            self._inner = FakeLanguageAlignedEncoder()

        def encode_image(self, image: ImagePayload, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
            if image.width > 8:  # o crop justo tem 8px; só o contextual é maior
                raise ValueError("context encoder is unavailable")
            return self._inner.encode_image(image, config)

        def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
            return self._inner.encode_text(text, config)

    config = dataclasses.replace(
        default_config(), multi_context=MultiContextConfig(contextual_crop_enabled=True)
    )
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    result = extract_region_evidence(
        (_region(),), payload, config, BrokenOnContext(), feature_map=feature_map
    )

    region = result.regions[0]
    assert region.evidence_for(EvidenceSlot.CONTEXTUAL_CROP).state is EvidenceState.FAILED
    assert region.evidence_for(EvidenceSlot.TIGHT_CROP).state is EvidenceState.AVAILABLE
    assert region.evidence_for(EvidenceSlot.FOREGROUND_DENSE).state is EvidenceState.AVAILABLE
    assert [failure.slot for failure in result.failures] == [EvidenceSlot.CONTEXTUAL_CROP]


def test_evidence_extraction_is_deterministic_and_records_its_provenance() -> None:
    config = dataclasses.replace(
        default_config(), multi_context=MultiContextConfig(contextual_crop_enabled=True)
    )
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    first = extract_region_evidence(
        (_region(),), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    )
    second = extract_region_evidence(
        (_region(),), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    )

    assert first.regions == second.regions
    assert first.visual_embeddings == second.visual_embeddings
    foreground = first.regions[0].evidence_for(EvidenceSlot.FOREGROUND_DENSE)
    assert foreground.space.modality is EmbeddingModality.VISUAL_DENSE
    assert foreground.space.checkpoint == config.feature_extraction.checkpoint
    assert foreground.preprocessing == "mask_aware_pooling:pixel_nearest_highres"


def test_without_a_dense_feature_map_the_foreground_slot_says_so() -> None:
    config: ModuleConfig = default_config()
    payload = payload_with_blobs()

    region = extract_region_evidence(
        (_region(),), payload, config, FakeLanguageAlignedEncoder(), feature_map=None
    ).regions[0]

    foreground = region.evidence_for(EvidenceSlot.FOREGROUND_DENSE)
    assert foreground.state is EvidenceState.MISSING
    assert foreground.reason == "no dense feature map available"
    assert region.visual_embedding_ref is None


# Verifica que o report operacional contabiliza custo e cobertura por slot,
# inclusive para os contextos habilitados no perfil de referência (#209).
def test_multi_context_metrics_report_actual_calls_and_states_per_slot() -> None:
    """Conta uma chamada por crop/região e uma única chamada global de cena."""
    config = dataclasses.replace(
        default_config(),
        multi_context=MultiContextConfig(
            contextual_crop_enabled=True,
            scene_conditioned_enabled=True,
        ),
    )
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, config.feature_extraction)

    result = extract_region_evidence(
        (_region(),), payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    )
    metrics = {metric.slot: metric for metric in result.metrics}

    assert metrics[EvidenceSlot.FOREGROUND_DENSE].available == 1
    assert metrics[EvidenceSlot.FOREGROUND_DENSE].model_calls == 0
    assert metrics[EvidenceSlot.TIGHT_CROP].model_calls == 1
    assert metrics[EvidenceSlot.CONTEXTUAL_CROP].model_calls == 1
    assert metrics[EvidenceSlot.SCENE_CONDITIONED].model_calls == 1
    assert all(metric.failed == 0 for metric in metrics.values())


# Compara baseline e perfil completo sobre as mesmas regiões para proteger
# a regra de que evidência contextual nunca altera IDs, masks ou boxes.
def test_enabling_all_context_slots_preserves_canonical_geometry_and_ids() -> None:
    """Mantém identidade e geometria idênticas ao habilitar contexto e cena."""
    payload = payload_with_blobs()
    base_config = default_config()
    full_config = dataclasses.replace(
        base_config,
        multi_context=MultiContextConfig(
            contextual_crop_enabled=True,
            scene_conditioned_enabled=True,
        ),
    )
    source_regions = (_region(), _region("region-b", (16, 16, 24, 24)))
    feature_map = FakeDenseFeatureExtractor().extract(payload, base_config.feature_extraction)

    baseline = extract_region_evidence(
        source_regions,
        payload,
        base_config,
        FakeLanguageAlignedEncoder(),
        feature_map=feature_map,
    )
    full = extract_region_evidence(
        source_regions,
        payload,
        full_config,
        FakeLanguageAlignedEncoder(),
        feature_map=feature_map,
    )

    assert tuple(region.region_id for region in full.regions) == tuple(
        region.region_id for region in baseline.regions
    )
    assert tuple(region.mask for region in full.regions) == tuple(
        region.mask for region in baseline.regions
    )
    assert tuple(region.box for region in full.regions) == tuple(
        region.box for region in baseline.regions
    )


# Protege a garantia que faz a #203 valer: a view em pixels entregue ao
# raciocínio semântico é exatamente o recorte que gerou o slot persistido. Se
# as duas geometrias divergirem, o reasoner passa a olhar uma área e a
# evidência serializada a descrever outra, sem nada acusar.
def test_each_view_matches_the_geometry_of_the_slot_it_produced() -> None:
    """Cada view compartilha crop_box e transform com o slot correspondente."""
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, default_config().feature_extraction)
    config = dataclasses.replace(
        default_config(),
        multi_context=MultiContextConfig(contextual_crop_enabled=True, scene_conditioned_enabled=True),
    )
    regions = (_region("region-a"), _region("region-b", box=(16, 16, 28, 28)))

    result = extract_region_evidence(
        regions, payload, config, FakeLanguageAlignedEncoder(), feature_map=feature_map
    )

    for region in result.regions:
        by_slot = {view.slot: view for view in result.views[region.region_id]}
        for evidence in region.evidence:
            if evidence.state is not EvidenceState.AVAILABLE:
                continue
            if evidence.slot is EvidenceSlot.FOREGROUND_DENSE:
                # O slot denso guarda o transform da grade de features, não o
                # do recorte: só a caixa é comparável entre os dois.
                assert by_slot[evidence.slot].crop_box == evidence.crop_box
                continue
            view = by_slot[evidence.slot]
            assert view.crop_box == evidence.crop_box
            assert view.transform == evidence.transform


# Garante que um slot desabilitado por configuração não gera view: sem isso, o
# perfil de ablation continuaria alimentando o reasoner com a evidência que ele
# diz estar medindo a ausência.
def test_a_disabled_slot_produces_no_view_for_the_reasoner() -> None:
    """O perfil baseline não entrega ao reasoner as views que desligou."""
    payload = payload_with_blobs()
    feature_map = FakeDenseFeatureExtractor().extract(payload, default_config().feature_extraction)
    baseline = dataclasses.replace(
        default_config(),
        multi_context=MultiContextConfig(contextual_crop_enabled=False, scene_conditioned_enabled=False),
    )

    result = extract_region_evidence(
        (_region(),), payload, baseline, FakeLanguageAlignedEncoder(), feature_map=feature_map
    )

    slots = {view.slot for view in result.views["region-a"]}
    assert EvidenceSlot.CONTEXTUAL_CROP not in slots
    assert EvidenceSlot.SCENE_CONDITIONED not in slots
    assert EvidenceSlot.FOREGROUND_DENSE in slots
