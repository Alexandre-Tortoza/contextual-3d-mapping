"""Testes do suporte de hipótese por alinhamento language-aligned (#214)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.hypothesis_support import (
    attach_hypothesis_signals,
    language_space,
)
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.config import HypothesisSupportConfig, LanguageEmbeddingConfig
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace, LanguageEmbedding
from visual_perception.domain.geometry import CoordinateTransform, Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import (
    EvidenceSlot,
    EvidenceState,
    RegionEvidenceSlot,
)
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantic_support import SupportSignalStatus
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    SemanticClaim,
    measured_signals,
)
from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")
_EMBEDDING_CONFIG = LanguageEmbeddingConfig(backend="fake", checkpoint="none", dimension=8)


# Encoder controlado: devolve vetores canônicos escolhidos pelo teste, para que
# o desfecho de cada sinal seja consequência da geometria do espaço e não do
# hash de um fake. É a única forma de testar "concorda", "discorda" e "empata"
# como três casos distintos e não como um acidente.
class _ScriptedEncoder:
    """Encoder alinhado a linguagem cujos vetores o teste escolhe."""

    # Guarda os vetores por texto e por identidade de artifact.
    def __init__(self, texts: dict[str, list[float]]) -> None:
        self._texts = texts
        self.calls = 0

    # Não é exercitado nestes testes: o suporte de hipótese reusa os embeddings
    # de imagem já produzidos pelo estágio de evidência.
    def encode_image(self, image: object, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        raise AssertionError("hypothesis support must not re-encode images")

    # Devolve o vetor roteirizado do texto pedido.
    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        self.calls += 1
        if text not in self._texts:
            raise ValueError(f"unscripted text {text!r}")
        return tuple(self._texts[text])


# Normaliza um vetor para que os cossenos do teste sejam legíveis.
def _unit(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return list(array / np.linalg.norm(array))


# Constrói uma claim de identidade com o papel pedido.
def _claim(value: str, role: HypothesisRole = HypothesisRole.PRIMARY) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="fake"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=role,
    )


# Constrói uma região com um slot alinhado a linguagem disponível.
def _region(
    claims: tuple[SemanticClaim, ...],
    *,
    artifact_ref: str | None = "language-region-a",
    region_id: str = "region-a",
) -> ObservedRegion:
    data = np.zeros((16, 16), dtype=np.bool_)
    data[2:8, 2:8] = True
    mask = Mask(data, 16, 16)
    state = EvidenceState.AVAILABLE if artifact_ref else EvidenceState.FAILED
    slot = RegionEvidenceSlot(
        slot=EvidenceSlot.TIGHT_CROP,
        region_id=region_id,
        state=state,
        crop_box=mask.bounding_box() if artifact_ref else None,
        transform=CoordinateTransform.identity() if artifact_ref else None,
        preprocessing="crop:expansion=0.0" if artifact_ref else None,
        artifact_ref=artifact_ref,
        space=language_space(_EMBEDDING_CONFIG) if artifact_ref else None,
        reason=None if artifact_ref else "encoder failed",
    )
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, ("p",), claims=claims, evidence=(slot,))


# Constrói o embedding de imagem que o slot referencia.
def _language_embedding(vector: list[float]) -> LanguageEmbedding:
    return LanguageEmbedding(
        embedding_id="language-region-a",
        region_id="region-a",
        vector=tuple(vector),
        dimension=len(vector),
        model_id="fake",
        checkpoint="none",
        normalized=True,
    )


_CONFIG = HypothesisSupportConfig(slots=(EvidenceSlot.TIGHT_CROP.value,))


# O caso central: a evidência de imagem está mais perto do texto da primária do
# que do texto da alternativa, e o sinal registra isso como apoio — com o score
# bruto e a margem preservados, e nenhuma conversão em confiança.
def test_a_primary_closer_to_the_evidence_is_supported() -> None:
    """O alinhamento apoia a hipótese mais próxima, preservando score e margem."""
    encoder = _ScriptedEncoder(
        {
            "a photo of door": _unit([1.0, 0.0, 0.0, 0, 0, 0, 0, 0]),
            "a photo of wall panel": _unit([0.0, 1.0, 0.0, 0, 0, 0, 0, 0]),
        }
    )
    region = _region((_claim("door"), _claim("wall panel", HypothesisRole.ALTERNATIVE)))
    embedding = _language_embedding(_unit([0.9, 0.1, 0.0, 0, 0, 0, 0, 0]))

    result = attach_hypothesis_signals(
        (region,), (embedding,), encoder, _CONFIG, _EMBEDDING_CONFIG
    )

    primary = primary_label_claim(result.regions[0])
    assert primary is not None
    signal = primary.signals[0]
    assert signal.status is SupportSignalStatus.SUPPORTS
    assert signal.source == "clip_alignment"
    assert signal.slot is EvidenceSlot.TIGHT_CROP
    assert signal.score == pytest.approx(0.9939, abs=1e-3)
    assert signal.margin > _CONFIG.indistinguishable_margin
    # O sinal nunca vira confiança: o score bruto do produtor segue intacto.
    assert primary.confidence is not None
    assert primary.confidence.value == pytest.approx(0.9)
    # E o espaço em que a medida foi feita acompanha o sinal.
    assert signal.space is not None
    assert signal.space.modality is EmbeddingModality.LANGUAGE_ALIGNED


# O desfecho simétrico: quando a alternativa fica mais perto, a primária recebe
# contradição — e a alternativa recebe apoio, porque o sinal é por hipótese.
def test_an_alternative_closer_to_the_evidence_contradicts_the_primary() -> None:
    """Uma alternativa mais próxima contradiz a primária e apoia a si mesma."""
    encoder = _ScriptedEncoder(
        {
            "a photo of door": _unit([1.0, 0.0, 0.0, 0, 0, 0, 0, 0]),
            "a photo of wall panel": _unit([0.0, 1.0, 0.0, 0, 0, 0, 0, 0]),
        }
    )
    region = _region((_claim("door"), _claim("wall panel", HypothesisRole.ALTERNATIVE)))
    embedding = _language_embedding(_unit([0.1, 0.9, 0.0, 0, 0, 0, 0, 0]))

    result = attach_hypothesis_signals(
        (region,), (embedding,), encoder, _CONFIG, _EMBEDDING_CONFIG
    )

    statuses = {claim.value: claim.signals[0].status for claim in result.regions[0].claims}
    assert statuses == {
        "door": SupportSignalStatus.CONTRADICTS,
        "wall panel": SupportSignalStatus.SUPPORTS,
    }


# O terceiro desfecho, que é o motivo de o contract não ter só dois: quando a
# margem fica abaixo do piso, o sinal declara que não distingue, em vez de
# eleger um vencedor que a medida não sustenta.
def test_a_margin_below_the_floor_is_indistinguishable() -> None:
    """Uma margem abaixo do piso configurado não elege vencedor."""
    encoder = _ScriptedEncoder(
        {
            "a photo of door": _unit([1.0, 0.0, 0.0, 0, 0, 0, 0, 0]),
            "a photo of wall panel": _unit([0.999, 0.0447, 0.0, 0, 0, 0, 0, 0]),
        }
    )
    region = _region((_claim("door"), _claim("wall panel", HypothesisRole.ALTERNATIVE)))
    embedding = _language_embedding(_unit([1.0, 0.0, 0.0, 0, 0, 0, 0, 0]))

    result = attach_hypothesis_signals(
        (region,), (embedding,), encoder, _CONFIG, _EMBEDDING_CONFIG
    )

    primary = primary_label_claim(result.regions[0])
    assert primary is not None
    assert primary.signals[0].status is SupportSignalStatus.INDISTINGUISHABLE
    assert abs(primary.signals[0].margin) < _CONFIG.indistinguishable_margin


# Sem hipótese concorrente não há o que arbitrar, e isso é declarado em vez de
# produzir um score solto que pareceria suporte.
def test_a_single_hypothesis_yields_an_unavailable_signal() -> None:
    """Uma região com uma só hipótese recebe sinal ``unavailable`` explicado."""
    encoder = _ScriptedEncoder({})
    region = _region((_claim("door"),))

    result = attach_hypothesis_signals(
        (region,), (_language_embedding(_unit([1.0] + [0.0] * 7)),), encoder, _CONFIG, _EMBEDDING_CONFIG
    )

    primary = primary_label_claim(result.regions[0])
    assert primary is not None
    assert primary.signals[0].status is SupportSignalStatus.UNAVAILABLE
    assert "no competing concept" in (primary.signals[0].reason or "")
    assert measured_signals(primary) == ()
    assert encoder.calls == 0


# Um slot que falhou não vira sinal silenciosamente ausente: ele vira um sinal
# indisponível com motivo, que é o que permite ao refinamento contá-lo.
def test_a_failed_evidence_slot_yields_an_unavailable_signal() -> None:
    """Um slot sem embedding produz sinal ``unavailable`` com motivo."""
    encoder = _ScriptedEncoder(
        {
            "a photo of door": _unit([1.0] + [0.0] * 7),
            "a photo of wall panel": _unit([0.0, 1.0] + [0.0] * 6),
        }
    )
    region = _region(
        (_claim("door"), _claim("wall panel", HypothesisRole.ALTERNATIVE)), artifact_ref=None
    )

    result = attach_hypothesis_signals((region,), (), encoder, _CONFIG, _EMBEDDING_CONFIG)

    primary = primary_label_claim(result.regions[0])
    assert primary is not None
    assert primary.signals[0].status is SupportSignalStatus.UNAVAILABLE
    assert "no language-aligned embedding" in (primary.signals[0].reason or "")


# O estágio desligado é um caminho de custo zero explícito.
def test_a_disabled_stage_leaves_the_regions_untouched() -> None:
    """Desligar o suporte de hipótese devolve as regiões sem sinais."""
    region = _region((_claim("door"),))

    result = attach_hypothesis_signals(
        (region,), (), _ScriptedEncoder({}), dataclasses.replace(_CONFIG, enabled=False), _EMBEDDING_CONFIG
    )

    assert result.regions[0].claims[0].signals == ()
    assert result.text_encode_calls == 0


# O texto de cada conceito é codificado uma vez por frame, e não uma vez por
# região: num frame real, ``wall`` é a hipótese de dezenas de regiões.
def test_text_embeddings_are_encoded_once_per_frame() -> None:
    """Conceitos repetidos entre regiões custam um único encoding de texto."""
    encoder = _ScriptedEncoder(
        {
            "a photo of door": _unit([1.0] + [0.0] * 7),
            "a photo of wall panel": _unit([0.0, 1.0] + [0.0] * 6),
        }
    )
    claims = (_claim("door"), _claim("wall panel", HypothesisRole.ALTERNATIVE))
    first = _region(claims)
    second = _region(claims, artifact_ref="language-region-b", region_id="region-b")
    embeddings = (
        _language_embedding(_unit([1.0] + [0.0] * 7)),
        dataclasses.replace(
            _language_embedding(_unit([1.0] + [0.0] * 7)),
            embedding_id="language-region-b",
            region_id="region-b",
        ),
    )

    attach_hypothesis_signals((first, second), embeddings, encoder, _CONFIG, _EMBEDDING_CONFIG)

    assert encoder.calls == 2


# A garantia de fronteira: o pipeline canônico anexa sinais e o faz sem
# recodificar nenhuma imagem — os vetores vêm do estágio de evidência.
def test_the_canonical_pipeline_attaches_signals_and_exposes_the_embeddings() -> None:
    """O pipeline anexa sinais às hipóteses e devolve os embeddings produzidos."""
    config = dataclasses.replace(
        default_config(),
        language_embedding=LanguageEmbeddingConfig(backend="fake", checkpoint="none", dimension=512),
    )
    result = run_canonical_pipeline(
        image_observation(),
        payload_with_blobs(blobs=((2, 2, 10, 10, (200, 30, 30)),)),
        config,
        default_ports(),
    )

    assert result.language_embeddings
    assert result.visual_embeddings
    assert result.stage_model_calls["hypothesis_support_text"] >= 0
    for region in result.observation.regions:
        primary = primary_label_claim(region)
        assert primary is not None
        assert primary.signals


# O espaço declarado pelo sinal precisa ser o mesmo que a configuração declara,
# para que dois sinais de checkpoints diferentes nunca sejam agregados juntos.
def test_the_declared_space_matches_the_configuration() -> None:
    """O espaço do sinal é derivado da configuração de embedding."""
    space = language_space(_EMBEDDING_CONFIG)

    assert space == EmbeddingSpace(
        model_id="fake",
        checkpoint="none",
        dimension=8,
        modality=EmbeddingModality.LANGUAGE_ALIGNED,
        normalized=True,
    )
    assert isinstance(FakeLanguageAlignedEncoder().encode_text("a photo of wall", _EMBEDDING_CONFIG), tuple)
