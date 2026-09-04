"""Testes unitários dos adapters reais sem depender de GPU ou checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import payload_with_blobs
from visual_perception.config import LanguageEmbeddingConfig, RegionDiscoveryConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.infrastructure.adapters._runtime import require_checkpoint
from visual_perception.infrastructure.adapters.factory import create_perception_ports
from visual_perception.infrastructure.adapters.language_embedding_backend import _to_vector
from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import _parse_json_object
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
        ('```json\n{"labels": ["chair"]}\n```', {"labels": ["chair"]}),
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
