"""Testes unitários dos adapters reais sem depender de GPU ou checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import payload_with_blobs
from visual_perception.config import LanguageEmbeddingConfig, RegionDiscoveryConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import EvidenceSlot, SubjectEmphasis
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionView
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.infrastructure.adapters._runtime import require_checkpoint
from visual_perception.infrastructure.adapters.factory import create_perception_ports
from visual_perception.infrastructure.adapters.language_embedding_backend import _to_vector
from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import (
    _describe_scene_claims,
    _describe_views,
    _parse_json_object,
    _region_prompt,
)
from visual_perception.infrastructure.adapters.region_discovery_backend import _proposal_from_mask


# Garante que uma máscara+score válidos do SAM sejam convertidos para a
# geometria canônica do módulo, preservando resolução, box semiaberta e
# confiança.
def test_sam_mask_becomes_local_region_proposal() -> None:
    """Converte uma máscara+score válidos do SAM em proposta local canônica."""
    image = payload_with_blobs(width=8, height=8, blobs=((2, 3, 6, 7, (0, 0, 0)),))
    segmentation = np.zeros((8, 8), dtype=np.bool_)
    segmentation[3:7, 2:6] = True

    proposal = _proposal_from_mask(
        segmentation,
        0.8,
        0,
        image,
        RegionDiscoveryConfig(backend="sam", checkpoint="facebook/sam-vit-huge", min_mask_area=1),
    )

    assert proposal is not None
    assert proposal.box.x_min == 2
    assert proposal.box.y_min == 3
    assert proposal.box.x_max == 6
    assert proposal.box.y_max == 7
    assert proposal.geometric_confidence == 0.8


# Protege a fronteira contra máscaras devolvidas em resolução diferente da
# entrada, que não podem ser remapeadas corretamente pelo tiling.
def test_sam_mask_with_wrong_resolution_is_rejected() -> None:
    """Rejeita máscara do SAM que não corresponde à resolução do payload."""
    image = payload_with_blobs(width=8, height=8)
    with pytest.raises(BackendExecutionError):
        _proposal_from_mask(
            np.ones((7, 8), dtype=np.bool_),
            0.8,
            0,
            image,
            RegionDiscoveryConfig(backend="sam", checkpoint="facebook/sam-vit-huge"),
        )


# Confirma que o adapter aceita JSON puro e JSON envolto em markdown, formas
# comuns de saída de VLMs instruction-tuned.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"scene_type": "room"}', {"scene_type": "room"}),
        ('```json\n{"label": "chair"}\n```', {"label": "chair"}),
    ],
)
def test_vlm_json_parser_extracts_one_object(text: str, expected: dict[str, object]) -> None:
    """Extrai um objeto JSON de respostas textuais típicas de VLM."""
    assert _parse_json_object(text) == expected


# Confirma que texto sem objeto JSON não é uma falha de transporte: a camada
# application receberá um dict vazio e emitirá seu diagnóstico de schema.
def test_vlm_json_parser_returns_empty_object_for_invalid_output() -> None:
    """Retorna objeto vazio para saída VLM malformada."""
    assert _parse_json_object("não é JSON") == {}


# Exercita a validação de dimensão e a normalização do adapter CLIP sem
# carregar a biblioteca nem um checkpoint real.
def test_clip_vector_is_normalized_and_matches_configured_dimension() -> None:
    """Normaliza o vetor CLIP e preserva a dimensão configurada."""
    config = LanguageEmbeddingConfig(backend="clip", checkpoint="weights", dimension=3)

    class TensorLike:
        """Simula a cadeia mínima de conversão de tensor usada pelo adapter."""

        def __getitem__(self, _: int) -> TensorLike:
            """Mantém o próprio objeto ao selecionar o primeiro batch."""
            return self

        def detach(self) -> TensorLike:
            """Simula a desconexão do grafo de autograd."""
            return self

        def float(self) -> TensorLike:
            """Simula a conversão do tensor para float."""
            return self

        def cpu(self) -> TensorLike:
            """Simula a transferência do tensor para CPU."""
            return self

        def numpy(self) -> np.ndarray:
            """Devolve um vetor finito de dimensão conhecida."""
            return np.asarray([3.0, 4.0, 0.0])

    vector = _to_vector(TensorLike(), config, "clip")
    assert vector == pytest.approx((0.6, 0.8, 0.0))


# Garante uma mensagem precoce e acionável quando alguém seleciona backend
# real mas esquece de substituir o checkpoint placeholder dos fakes.
def test_real_backend_requires_explicit_checkpoint() -> None:
    """Rejeita o checkpoint placeholder usado pelos backends fake."""
    with pytest.raises(BackendUnavailableError):
        require_checkpoint("none", "clip")


# Confirma que a composição default continua selecionando somente fakes e não
# exige carregar bibliotecas de ML para construir PerceptionPorts.
def test_port_factory_keeps_fake_defaults_gpu_free() -> None:
    """Compõe os quatro ports fake para a configuração default do módulo."""
    from visual_perception.config import ModuleConfig
    from visual_perception.infrastructure.fakes.fake_feature_extractor import FakeDenseFeatureExtractor
    from visual_perception.infrastructure.fakes.fake_language_encoder import FakeLanguageAlignedEncoder
    from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner
    from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer

    ports = create_perception_ports(ModuleConfig())
    assert isinstance(ports.region_discoverer, FakeRegionDiscoverer)
    assert isinstance(ports.feature_extractor, FakeDenseFeatureExtractor)
    assert isinstance(ports.language_encoder, FakeLanguageAlignedEncoder)
    assert isinstance(ports.multimodal_reasoner, FakeMultimodalReasoner)


# Constrói um RegionReasoningRequest mínimo para os testes de tradução de
# prompt, sem passar pelo pipeline nem carregar checkpoint.
def _reasoning_request(
    slots: tuple[EvidenceSlot, ...], scene_claims: tuple[SemanticClaim, ...] = ()
) -> RegionReasoningRequest:
    box = BoundingBox(0.0, 0.0, 4.0, 4.0)
    views = tuple(
        RegionView(
            slot=slot,
            payload=payload_with_blobs(width=4, height=4),
            crop_box=box,
            transform=CoordinateTransform(1.0, 1.0, 0.0, 0.0),
            emphasis=(
                SubjectEmphasis.ZERO_FILL
                if slot is EvidenceSlot.FOREGROUND_DENSE
                else SubjectEmphasis.NONE
            ),
        )
        for slot in slots
    )
    return RegionReasoningRequest(
        region_id="region-a",
        region_box=box,
        image_width=32,
        image_height=32,
        views=views,
        scene_claims=scene_claims,
    )


# Verifica a metade textual da evidência distinguível da #203: o prompt numera
# as imagens e diz qual delas é o sujeito. Sem esse rótulo, um VLM que recebe
# várias imagens não tem como saber qual região deve descrever.
def test_region_prompt_numbers_and_labels_each_view() -> None:
    """O preâmbulo identifica cada imagem enviada e marca o contexto como contexto."""
    request = _reasoning_request(
        (EvidenceSlot.FOREGROUND_DENSE, EvidenceSlot.TIGHT_CROP, EvidenceSlot.CONTEXTUAL_CROP)
    )

    described = _describe_views(request)

    assert "3 image(s)" in described
    assert described.index("Image 1") < described.index("Image 2") < described.index("Image 3")
    assert "isolated on a black background" in described
    assert "CONTEXT ONLY" in described


# Protege o critério da #202 de que nenhuma propriedade de cena vira verdade da
# região: o bloco de contexto preserva kind e confiança de cada claim e instrui
# explicitamente que ele descreve a cena, não o sujeito.
def test_scene_claims_reach_the_prompt_typed_and_marked_as_scene_level() -> None:
    """O contexto de cena entra no prompt por kind, com score, e marcado como da cena."""
    provenance = ModelProvenance(stage="scene_context", producer="fake", config_fingerprint="fp")
    claims = (
        SemanticClaim(
            ClaimKind.SCENE_TYPE,
            "corridor",
            ConfidenceScore(0.82, "fake"),
            (Evidence(description="raw"),),
            provenance,
        ),
        SemanticClaim(ClaimKind.HAZARD, "wet floor", None, (Evidence(description="raw"),), provenance),
    )
    request = _reasoning_request((EvidenceSlot.FOREGROUND_DENSE,), scene_claims=claims)

    described = _describe_scene_claims(request)

    assert "scene_type: corridor (confidence 0.82)" in described
    assert "hazard: wet floor" in described
    assert "never of the subject region" in described


# Sem claims de cena o bloco desaparece por inteiro, em vez de virar um rótulo
# vazio que o modelo trataria como informação.
def test_absent_scene_context_adds_nothing_to_the_prompt() -> None:
    """Um request sem claims de cena não acrescenta bloco de contexto ao prompt."""
    request = _reasoning_request((EvidenceSlot.FOREGROUND_DENSE,))
    assert _describe_scene_claims(request) == ""


# O exemplo de formato do prompt não pode ser uma resposta plausível. Com o
# exemplo concreto anterior, o modelo o reproduzia inteiro em 27/54 regiões de
# um frame; um placeholder mantém a forma inequívoca e torna o eco detectável
# pelo parser em vez de plausível.
def test_the_region_prompt_example_uses_placeholders_not_answerable_values() -> None:
    """O exemplo de formato do prompt não oferece um label copiável."""
    prompt = _region_prompt(_reasoning_request((EvidenceSlot.FOREGROUND_DENSE,)))

    assert '"label": "<one noun naming the subject>"' in prompt
    assert '"label": "door"' not in prompt
    assert '"material": "wood"' not in prompt
    assert '"label": "panel"' not in prompt
    assert "never copy a placeholder" in prompt
