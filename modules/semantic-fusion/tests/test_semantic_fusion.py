"""Testes da regra de fusão semântica multi-observação."""

from __future__ import annotations

import pytest
from semantic_fusion import (
    LanguageEmbeddingReference,
    PointVisualFeature,
    SemanticContribution,
    VisualCoherencePolicy,
    fuse_language_embeddings,
    fuse_point_contributions,
    fuse_point_features,
)


# Constrói uma contribuição completa para variar um sinal por vez nos testes.
def _contribution(
    observation_id: str,
    label: str,
    *,
    timestamp_ns: int = 10,
    confidence: float | None = 0.9,
    calibrated_confidence: float | None = None,
    visual_support: float | None = None,
    region_quality: float | None = None,
) -> SemanticContribution:
    """Cria uma contribuição semântica de teste."""
    return SemanticContribution(
        observation_id=observation_id,
        timestamp_ns=timestamp_ns,
        region_id=f"region-{observation_id}",
        label=label,
        confidence=confidence,
        calibrated_confidence=calibrated_confidence,
        support_state="raw",
        visual_support=visual_support,
        region_quality=region_quality,
    )


# Um único keyframe continua sendo o caso mais comum de um trecho curto, e a
# fusão não pode alterar o resultado nesse caso.
def test_fonte_unica_preserva_o_claim_original() -> None:
    """Confere que uma contribuição isolada vence com concordância total."""
    fused = fuse_point_contributions([_contribution("frame-a", "porta")])
    assert fused.label == "porta"
    assert fused.agreement == 1.0
    assert fused.contribution_count == 1
    assert fused.confidence == 0.9


# Concordância é evidência: quando vários keyframes veem a mesma superfície e
# dizem a mesma coisa, isso precisa aparecer no artifact.
def test_fontes_concordantes_registram_concordancia_total() -> None:
    """Confere concordância quando todas as observações apontam o mesmo label."""
    fused = fuse_point_contributions(
        [_contribution("frame-a", "porta"), _contribution("frame-b", "porta", timestamp_ns=20)]
    )
    assert fused.agreement == 1.0
    assert fused.contribution_count == 2


# O conflito é o caso que justifica o módulo: a confiança decide, e a proposta
# perdedora continua registrada em vez de desaparecer.
def test_conflito_escolhe_a_maior_confianca_sem_descartar_a_perdedora() -> None:
    """Confere escolha por confiança e preservação dos contribuintes."""
    fused = fuse_point_contributions(
        [
            _contribution("frame-a", "parede", confidence=0.4),
            _contribution("frame-b", "porta", confidence=0.8, timestamp_ns=20),
        ]
    )
    assert fused.label == "porta"
    assert fused.observation_id == "frame-b"
    assert fused.agreement == 0.5
    assert [item.label for item in fused.contributions] == ["porta", "parede"]


# Confiança calibrada, quando existe, tem precedência sobre a confiança bruta
# declarada pelo modelo, que este pipeline ainda não calibra.
def test_confianca_calibrada_tem_precedencia_sobre_a_bruta() -> None:
    """Confere que a calibração vence a confiança declarada."""
    fused = fuse_point_contributions(
        [
            _contribution("frame-a", "parede", confidence=0.95),
            _contribution("frame-b", "porta", confidence=0.10, calibrated_confidence=0.99),
        ]
    )
    assert fused.label == "porta"
    assert fused.confidence == 0.99


# Sem determinismo, dois runs sobre a mesma entrada produziriam mapas
# diferentes, e nenhuma comparação de experimento seria válida.
def test_empate_e_resolvido_de_forma_deterministica() -> None:
    """Confere o desempate por suporte, instante e identificador de região."""
    primeiro = _contribution("frame-a", "parede", timestamp_ns=10)
    segundo = _contribution("frame-b", "porta", timestamp_ns=20)
    assert fuse_point_contributions([primeiro, segundo]).label == "parede"
    assert fuse_point_contributions([segundo, primeiro]).label == "parede"
    com_suporte = _contribution("frame-b", "porta", timestamp_ns=20, visual_support=0.9)
    assert fuse_point_contributions([primeiro, com_suporte]).label == "porta"


# Entradas impossíveis precisam falhar na fronteira, e não produzir um label
# vazio que o viewer apresentaria como classificação.
def test_entradas_invalidas_falham_na_fronteira() -> None:
    """Confere validação de lista vazia, label vazio e medida fora de faixa."""
    with pytest.raises(ValueError, match="at least one contribution"):
        fuse_point_contributions([])
    with pytest.raises(ValueError, match="label must not be empty"):
        _contribution("frame-a", "   ")
    with pytest.raises(ValueError, match="confidence"):
        _contribution("frame-a", "porta", confidence=1.5)


# Regressão: bool é subtipo de int em Python e era convertido silenciosamente
# em score 0 ou 1, alterando o ranking de uma contribuição inválida.
def test_boolean_confidence_is_rejected() -> None:
    """Recusa booleano onde o contract exige uma medida contínua."""
    with pytest.raises(ValueError, match="confidence"):
        _contribution("frame-a", "porta", confidence=True)


# Mede a coerência como diagnóstico independente de confiança e agreement.
def test_coerencia_visual_detecta_features_contraditorias() -> None:
    """Publica contradição para vetores opostos no mesmo espaço."""
    first = _contribution("frame-a", "porta")
    second = _contribution("frame-b", "porta", timestamp_ns=20)
    first = first.__class__(**{**first.__dict__, "point_feature": PointVisualFeature((1.0, 0.0), "dino", "dino:cfg", "artifact://a")})
    second = second.__class__(**{**second.__dict__, "point_feature": PointVisualFeature((-1.0, 0.0), "dino", "dino:cfg", "artifact://b")})
    fused = fuse_point_contributions((first, second), visual_coherence_policy=VisualCoherencePolicy.DIAGNOSTIC)
    assert fused.visual_coherence is not None
    assert fused.visual_coherence.state == "contradictory"
    assert fused.visual_coherence.mean_cosine_similarity == -1.0


# Espaços incompatíveis não podem virar uma similaridade numérica enganosa.
def test_coerencia_visual_recusa_espacos_incompativeis() -> None:
    """Marca coerência indisponível quando produtores não são substituíveis."""
    first = _contribution("frame-a", "porta")
    second = _contribution("frame-b", "porta", timestamp_ns=20)
    first = first.__class__(**{**first.__dict__, "point_feature": PointVisualFeature((1.0, 0.0), "dino-a", "dino", "artifact://a")})
    second = second.__class__(**{**second.__dict__, "point_feature": PointVisualFeature((1.0, 0.0), "dino-b", "dino", "artifact://b")})
    coherence = fuse_point_contributions((first, second)).visual_coherence
    assert coherence is not None
    assert coherence.state == "unavailable"
    assert coherence.reason == "incompatible_feature_space"


# A representação vetorial da #222 permanece reproduzível e rastreia o peso
# efetivo sem alterar a política categórica da fusão de labels.
def test_fusao_de_features_normaliza_e_preserva_pesos_e_artifacts() -> None:
    """Agrega duas features compatíveis em ordem independente."""
    first = _contribution("frame-a", "porta", confidence=0.25)
    second = _contribution("frame-b", "porta", confidence=0.75, timestamp_ns=20)
    first = first.__class__(**{**first.__dict__, "point_feature": PointVisualFeature((1.0, 0.0), "clip", "clip:cfg", "artifact://a")})
    second = second.__class__(**{**second.__dict__, "point_feature": PointVisualFeature((0.0, 1.0), "clip", "clip:cfg", "artifact://b")})
    fused = fuse_point_features((second, first))
    assert fused.values == pytest.approx((0.316227766, 0.948683298))
    assert fused.contributor_artifacts == ("artifact://a", "artifact://b")
    assert fused.effective_weights == pytest.approx((0.25, 0.75))


# Constrói uma contribuição com referência de embedding de linguagem para
# exercitar fuse_language_embeddings sem repetir a identidade CLIP em cada teste.
def _language_contribution(
    observation_id: str,
    embedding_id: str,
    *,
    timestamp_ns: int = 10,
    confidence: float | None = None,
    calibrated_confidence: float | None = None,
    visual_support: float | None = None,
    region_quality: float | None = None,
    dimension: int = 2,
) -> SemanticContribution:
    """Cria uma contribuição de teste com referência de embedding CLIP."""
    base = _contribution(
        observation_id,
        "porta",
        timestamp_ns=timestamp_ns,
        confidence=confidence,
        calibrated_confidence=calibrated_confidence,
        visual_support=visual_support,
        region_quality=region_quality,
    )
    reference = LanguageEmbeddingReference(
        embedding_id=embedding_id,
        archive_uri=f"artifact://{embedding_id}",
        embedding_space="clip",
        dimension=dimension,
        producer="clip:cfg",
    )
    return base.__class__(**{**base.__dict__, "language_embedding": reference})


# Mapa fixo entre embedding_id e vetor bruto, para que os testes controlem
# exatamente o que o resolver devolve sem depender de um archive real.
def _vector_resolver(vectors: dict[str, tuple[float, ...]]) -> object:
    """Cria um resolver de teste que devolve vetores de um mapa fixo."""
    def resolve(reference: LanguageEmbeddingReference) -> tuple[float, ...]:
        return vectors[reference.embedding_id]

    return resolve


# Uma única referência é o caso mais comum, e a fusão precisa devolver o
# mesmo vetor normalizado sem introduzir ruído de agregação.
def test_fusao_de_linguagem_fonte_unica_preserva_o_vetor_normalizado() -> None:
    """Confere que uma única referência produz o vetor normalizado por L2."""
    contribution = _language_contribution("frame-a", "emb-a")
    fused = fuse_language_embeddings(
        [contribution], _vector_resolver({"emb-a": (3.0, 4.0)})
    )
    assert fused.values == pytest.approx((0.6, 0.8))
    assert fused.embedding_space == "clip"
    assert fused.dimension == 2
    assert fused.producer == "clip:cfg"
    assert fused.contributor_references == (contribution.language_embedding,)
    assert fused.effective_weights == pytest.approx((1.0,))


# A ordem de entrada não pode influenciar o resultado: a fusão ordena
# internamente por observação, região e identificador de embedding.
def test_fusao_de_linguagem_varias_fontes_e_independente_de_ordem() -> None:
    """Confere agregação ponderada e determinismo por ordem de entrada."""
    first = _language_contribution("frame-a", "emb-a", calibrated_confidence=0.25)
    second = _language_contribution("frame-b", "emb-b", timestamp_ns=20, calibrated_confidence=0.75)
    resolver = _vector_resolver({"emb-a": (1.0, 0.0), "emb-b": (0.0, 1.0)})
    forward = fuse_language_embeddings([first, second], resolver)
    backward = fuse_language_embeddings([second, first], resolver)
    assert forward.values == pytest.approx(backward.values)
    assert forward.values == pytest.approx((0.316227766, 0.948683298))
    assert forward.effective_weights == pytest.approx((0.25, 0.75))
    assert forward.contributor_references == (first.language_embedding, second.language_embedding)


# Sem nenhum sinal de peso declarado, todas as fontes devem contribuir
# igualmente em vez de a primeira dominar por acidente de ranking zero.
def test_fusao_de_linguagem_pesos_ausentes_ou_zero_ficam_uniformes() -> None:
    """Confere pesos uniformes quando toda contribuição tem peso zero/ausente."""
    first = _language_contribution("frame-a", "emb-a", visual_support=0.0, region_quality=0.0)
    second = _language_contribution("frame-b", "emb-b", timestamp_ns=20, visual_support=0.0, region_quality=0.0)
    resolver = _vector_resolver({"emb-a": (1.0, 0.0), "emb-b": (0.0, 1.0)})
    fused = fuse_language_embeddings([first, second], resolver)
    assert fused.effective_weights == pytest.approx((0.5, 0.5))
    assert fused.values == pytest.approx((0.707106781, 0.707106781))


# Espaço, dimensão e produtor divergentes indicam evidência incompatível, e
# uma média entre eles produziria um vetor sem significado geométrico.
def test_fusao_de_linguagem_recusa_dimensao_declarada_incompativel() -> None:
    """Recusa fundir referências com dimensão declarada divergente."""
    first = _language_contribution("frame-a", "emb-a", dimension=2)
    second = _language_contribution("frame-b", "emb-b", timestamp_ns=20, dimension=3)
    resolver = _vector_resolver({"emb-a": (1.0, 0.0), "emb-b": (0.0, 1.0, 0.0)})
    with pytest.raises(ValueError, match="compatible embedding space, dimension and producer"):
        fuse_language_embeddings([first, second], resolver)


# O resolver é a fronteira com o archive externo, e um vetor com forma
# diferente da dimensão declarada é um erro de produção, não uma média válida.
def test_fusao_de_linguagem_recusa_vetor_do_resolver_com_forma_invalida() -> None:
    """Recusa quando o resolver devolve um vetor com dimensão inesperada."""
    contribution = _language_contribution("frame-a", "emb-a", dimension=2)
    resolver = _vector_resolver({"emb-a": (1.0, 0.0, 0.0)})
    with pytest.raises(ValueError, match="invalid vector dimensions or values"):
        fuse_language_embeddings([contribution], resolver)


# Um vetor nulo não tem direção, e publicar um "normalizado" de zeros
# corromperia qualquer busca por similaridade de cosseno no semantic-memory.
def test_fusao_de_linguagem_recusa_vetor_nulo() -> None:
    """Recusa quando a agregação ponderada resulta em vetor de norma zero."""
    first = _language_contribution("frame-a", "emb-a")
    second = _language_contribution("frame-b", "emb-b", timestamp_ns=20)
    resolver = _vector_resolver({"emb-a": (1.0, 0.0), "emb-b": (-1.0, 0.0)})
    with pytest.raises(ValueError, match="zero-norm embedding"):
        fuse_language_embeddings([first, second], resolver)


# Contribuições sem referência de linguagem não participam da fusão CLIP:
# ela convive com claims puramente categóricos no mesmo ponto.
def test_fusao_de_linguagem_ignora_contribuicoes_sem_referencia() -> None:
    """Ignora contribuições sem language_embedding em vez de falhar."""
    with_reference = _language_contribution("frame-a", "emb-a")
    without_reference = _contribution("frame-b", "porta", timestamp_ns=20)
    fused = fuse_language_embeddings(
        [with_reference, without_reference], _vector_resolver({"emb-a": (1.0, 0.0)})
    )
    assert fused.contributor_references == (with_reference.language_embedding,)


# Nenhuma contribuição com referência é o caso degenerado: publicar um
# embedding fundido vazio esconderia a ausência total de evidência CLIP.
def test_fusao_de_linguagem_sem_nenhuma_referencia_falha_na_fronteira() -> None:
    """Recusa fundir quando nenhuma contribuição tem language_embedding."""
    with pytest.raises(ValueError, match="at least one embedding reference"):
        fuse_language_embeddings([_contribution("frame-a", "porta")], lambda ref: (0.0,))
