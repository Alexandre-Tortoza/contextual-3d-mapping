"""Testes unitários dos adapters reais sem depender de GPU ou checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixtures import payload_with_blobs
from visual_perception.config import LanguageEmbeddingConfig, MultimodalReasoningConfig, RegionDiscoveryConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_evidence import EvidenceSlot, SubjectEmphasis
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionView
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim
from visual_perception.infrastructure.adapters._runtime import require_checkpoint
from visual_perception.infrastructure.adapters.factory import create_perception_ports
from visual_perception.infrastructure.adapters.florence2_region_discovery_backend import (
    _proposals_from_florence_output,
)
from visual_perception.infrastructure.adapters.language_embedding_backend import _to_vector
from visual_perception.infrastructure.adapters.reasoning_prompts import (
    describe_scene_claims,
    describe_views,
    parse_json_object,
    region_prompt,
)
from visual_perception.infrastructure.adapters.region_discovery_backend import (
    _proposal_from_mask,
    _proposals_from_mask_outputs,
)


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


# Garante que a máscara do segment everything do SAM3 mantém o contract
# geométrico do port e registra backend e checkpoint na proveniência.
def test_sam3_mask_becomes_local_region_proposal_with_checkpoint_provenance() -> None:
    """Converte máscara do SAM3 tracker e preserva a proveniência do checkpoint."""
    image = payload_with_blobs(width=8, height=8)
    segmentation = np.zeros((8, 8), dtype=np.bool_)
    segmentation[1:5, 2:7] = True

    proposal = _proposal_from_mask(
        segmentation,
        0.9,
        0,
        image,
        RegionDiscoveryConfig(backend="sam3", checkpoint="facebook/sam3", min_mask_area=1),
    )

    assert proposal is not None
    assert proposal.local_id == "sam3-0"
    assert proposal.box == proposal.mask.bounding_box()
    assert proposal.source == "sam3:facebook/sam3"


# Confirma que o adapter mantém todas as propostas geométricas válidas da
# geração automática, ordenadas por score.
def test_sam3_outputs_multiple_local_region_proposals() -> None:
    """Converte múltiplas masks em propostas locais ordenadas por confiança."""
    image = payload_with_blobs(width=8, height=8)
    first = np.zeros((8, 8), dtype=np.bool_)
    second = np.zeros((8, 8), dtype=np.bool_)
    first[1:3, 1:3] = True
    second[4:7, 4:7] = True

    proposals = _proposals_from_mask_outputs(
        (first, second),
        (0.7, 0.9),
        image,
        RegionDiscoveryConfig(backend="sam3", checkpoint="facebook/sam3", min_mask_area=1),
    )

    assert len(proposals) == 2
    assert [proposal.geometric_confidence for proposal in proposals] == [0.9, 0.7]
    assert all(proposal.source == "sam3:facebook/sam3" for proposal in proposals)


# Garante que o output de REGION_PROPOSAL seja convertido para a geometria
# canônica, com clipping nas bordas e uma máscara retangular explícita.
def test_florence2_region_proposals_become_clipped_rectangular_masks() -> None:
    """Converte caixas Florence-2 em propostas locais rastreáveis."""
    image = payload_with_blobs(width=8, height=6)
    proposals = _proposals_from_florence_output(
        {
            "<REGION_PROPOSAL>": {
                "bboxes": [[1.2, 2.1, 5.0, 5.8], [-3, 0, 2, 3], [4, 4, 4, 5]],
            }
        },
        image,
        RegionDiscoveryConfig(
            backend="florence2", checkpoint="florence-community/Florence-2-large", min_mask_area=1
        ),
    )

    assert [proposal.local_id for proposal in proposals] == ["florence2-0", "florence2-1"]
    assert proposals[0].box == BoundingBox(1.0, 2.0, 5.0, 6.0)
    assert proposals[0].mask.area() == 16
    assert proposals[1].box == BoundingBox(0.0, 0.0, 2.0, 3.0)
    assert all(proposal.source == "florence2:florence-community/Florence-2-large" for proposal in proposals)


# Protege o contract contra respostas incompletas ou com tipos inesperados do
# post-processamento remoto do checkpoint.
def test_florence2_region_proposal_rejects_invalid_top_level_output() -> None:
    """Recusa um payload Florence-2 que não contém o objeto de tarefa esperado."""
    image = payload_with_blobs(width=4, height=4)
    with pytest.raises(BackendExecutionError, match="formato inválido"):
        _proposals_from_florence_output(
            [],
            image,
            RegionDiscoveryConfig(backend="florence2", checkpoint="florence-community/Florence-2-large"),
        )


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
        # Regressão do Gemini Robotics ER: objeto correto seguido de uma chave sobrando.
        ('{"scene_type": "hallway", "nested": {"a": 1}}\n}', {"scene_type": "hallway", "nested": {"a": 1}}),
        ('resposta: {"label": "door"} fim', {"label": "door"}),
    ],
)
def test_vlm_json_parser_extracts_one_object(text: str, expected: dict[str, object]) -> None:
    """Extrai um objeto JSON de respostas textuais típicas de VLM."""
    assert parse_json_object(text) == expected


# Confirma que texto sem objeto JSON não é uma falha de transporte: a camada
# application receberá um dict vazio e emitirá seu diagnóstico de schema.
def test_vlm_json_parser_returns_empty_object_for_invalid_output() -> None:
    """Retorna objeto vazio para saída VLM malformada."""
    assert parse_json_object("não é JSON") == {}


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


# Confirma que um `fallback_backend` declarado envolve o reasoner primário
# num FallbackMultimodalReasoningAdapter, sem exigir GPU nem rede para
# compor os ports (#292) — ambos os adapters concretos são preguiçosos.
def test_port_factory_wraps_multimodal_reasoner_with_fallback_when_declared() -> None:
    """Compõe o reasoner primário com fallback quando a config declara um."""
    from visual_perception.config import ModuleConfig
    from visual_perception.infrastructure.adapters.gemini_reasoning_backend import (
        GeminiRoboticsReasoningAdapter,
    )
    from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import (
        RealMultimodalReasoningAdapter,
    )
    from visual_perception.infrastructure.adapters.multimodal_reasoning_fallback import (
        FallbackMultimodalReasoningAdapter,
    )

    ports = create_perception_ports(
        ModuleConfig(
            multimodal_reasoning=MultimodalReasoningConfig(
                backend="gemini_robotics_er",
                checkpoint="gemini-robotics-er-2-preview",
                fallback_backend="qwen_vl",
                fallback_checkpoint="Qwen/Qwen2.5-VL-3B-Instruct",
            )
        )
    )

    reasoner = ports.multimodal_reasoner
    assert isinstance(reasoner, FallbackMultimodalReasoningAdapter)
    assert isinstance(reasoner._primary, GeminiRoboticsReasoningAdapter)  # noqa: SLF001
    assert isinstance(reasoner._fallback, RealMultimodalReasoningAdapter)  # noqa: SLF001


# Sem `fallback_backend`, a composição continua devolvendo o reasoner
# primário puro — o wrapper de fallback não deve aparecer sem ser pedido.
def test_port_factory_skips_fallback_wrapper_without_fallback_backend() -> None:
    """Sem `fallback_backend`, o reasoner primário é devolvido sem wrapper."""
    from visual_perception.config import ModuleConfig
    from visual_perception.infrastructure.adapters.gemini_reasoning_backend import (
        GeminiRoboticsReasoningAdapter,
    )

    ports = create_perception_ports(
        ModuleConfig(
            multimodal_reasoning=MultimodalReasoningConfig(
                backend="gemini_robotics_er", checkpoint="gemini-robotics-er-2-preview",
            )
        )
    )

    assert isinstance(ports.multimodal_reasoner, GeminiRoboticsReasoningAdapter)


# Confirma que SAM3 é selecionável sem importar o runtime opcional durante a
# composição; os pesos continuam sendo carregados apenas no primeiro discovery.
def test_port_factory_selects_sam3_region_discoverer_lazily() -> None:
    """Seleciona o adapter de geração automática sem exigir torch, Transformers ou checkpoint."""
    from visual_perception.config import ModuleConfig
    from visual_perception.infrastructure.adapters.region_discovery_backend import RealRegionDiscoveryAdapter

    ports = create_perception_ports(
        ModuleConfig(region_discovery=RegionDiscoveryConfig(backend="sam3", checkpoint="facebook/sam3"))
    )

    assert isinstance(ports.region_discoverer, RealRegionDiscoveryAdapter)


# Confirma que Florence-2 é composto preguiçosamente, sem baixar checkpoint nem
# importar Torch durante a criação dos ports.
def test_port_factory_selects_florence2_region_discoverer_lazily() -> None:
    """Seleciona o adapter Florence-2 sem carregar o runtime opcional."""
    from visual_perception.config import ModuleConfig
    from visual_perception.infrastructure.adapters.florence2_region_discovery_backend import (
        Florence2RegionDiscoveryAdapter,
    )

    ports = create_perception_ports(
        ModuleConfig(
            region_discovery=RegionDiscoveryConfig(
                backend="florence2", checkpoint="florence-community/Florence-2-large"
            )
        )
    )

    assert isinstance(ports.region_discoverer, Florence2RegionDiscoveryAdapter)


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

    described = describe_views(request)

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

    described = describe_scene_claims(request)

    assert "scene_type: corridor (confidence 0.82)" in described
    assert "hazard: wet floor" in described
    assert "never of the subject region" in described


# Sem claims de cena o bloco desaparece por inteiro, em vez de virar um rótulo
# vazio que o modelo trataria como informação.
def test_absent_scene_context_adds_nothing_to_the_prompt() -> None:
    """Um request sem claims de cena não acrescenta bloco de contexto ao prompt."""
    request = _reasoning_request((EvidenceSlot.FOREGROUND_DENSE,))
    assert describe_scene_claims(request) == ""


# O exemplo de formato do prompt não pode ser uma resposta plausível. Com o
# exemplo concreto anterior, o modelo o reproduzia inteiro em 27/54 regiões de
# um frame; um placeholder mantém a forma inequívoca e torna o eco detectável
# pelo parser em vez de plausível.
def test_the_region_prompt_example_uses_placeholders_not_answerable_values() -> None:
    """O exemplo de formato do prompt (schema de 8 campos, v8/v9) não oferece um label copiável."""
    prompt = region_prompt(_reasoning_request((EvidenceSlot.FOREGROUND_DENSE,)), "v8")

    assert '"label": "<one noun naming the subject>"' in prompt
    assert '"label": "door"' not in prompt
    assert '"material": "wood"' not in prompt
    assert '"label": "panel"' not in prompt
    assert "never copy a placeholder" in prompt


# Mesma garantia acima, para o schema reduzido de 2 campos (v10/v11) que só o
# backend Qwen seleciona — ver reasoning_prompts._MINIMAL_SCHEMA_BODY.
def test_the_minimal_region_prompt_example_uses_placeholders_not_answerable_values() -> None:
    """O exemplo de formato do schema reduzido também não oferece um label copiável."""
    prompt = region_prompt(_reasoning_request((EvidenceSlot.FOREGROUND_DENSE,)), "v10")

    assert '"label": "<one noun>"' in prompt
    assert '"label": "door"' not in prompt
    assert "never copy them" in prompt


# Os dois schemas de prompt de região vivem sob versões distintas: nada deve
# aceitar uma versão fora do vocabulário fechado (v8/v9/v10/v11).
def test_region_prompt_rejects_an_unknown_prompt_version() -> None:
    """Uma ``prompt_version`` desconhecida é rejeitada em vez de cair num default silencioso."""
    with pytest.raises(ValueError, match="prompt_version"):
        region_prompt(_reasoning_request((EvidenceSlot.FOREGROUND_DENSE,)), "v42")


# O exemplo de formato de um prompt é copiado de volta pelo modelo quando o
# valor nele é plausível. Foi medido duas vezes — 26 de 27 regiões reproduzindo
# a assinatura inteira do exemplo (#212), e 15 de 16 respostas de relação com
# ``0.95`` exato — e corrigido nos dois prompts. O de cena era o terceiro, e
# escapou por só ser consultado uma vez por frame.
def test_no_prompt_offers_a_round_confidence_for_the_model_to_copy() -> None:
    """Nenhum dos três prompts embute um ``confidence`` de exemplo redondo."""
    import re

    from visual_perception.infrastructure.adapters import reasoning_prompts

    source = Path(reasoning_prompts.__file__).read_text()
    offered = re.findall(r'"confidence":\s*([0-9.]+)', source)

    assert offered, "os prompts precisam continuar mostrando o formato esperado"
    for value in offered:
        assert float(value) * 100 % 10 != 0, (
            f"o exemplo de confidence {value} é redondo o bastante para ser copiado de volta"
        )
