"""Fusão multi-observação de claims semânticos ancorados em um ponto do mapa."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .models import (
    FusedLanguageEmbedding,
    FusedPointContext,
    FusedVisualEmbedding,
    SemanticContribution,
    VisualCoherence,
    VisualCoherencePolicy,
)

# Resolve as referências CLIP apenas na fronteira de fusão. O alias mantém a
# dependência do semantic-fusion estreita: archives e formatos ficam no produtor.
LanguageEmbeddingResolver = Callable[[object], tuple[float, ...]]


# Funde vetores alinhados à linguagem de uma geometria usando somente fontes
# explicitamente compatíveis. É chamada pelo semantic-map ao materializar estado.
def fuse_language_embeddings(
    contributions: Iterable[SemanticContribution], resolver: LanguageEmbeddingResolver,
) -> FusedLanguageEmbedding:
    """Resolve e agrega embeddings CLIP por média ponderada normalizada.

    Argumentos:
        contributions: contribuições da mesma geometria persistente.
        resolver: fronteira que resolve uma referência no archive do produtor.
    Retorna:
        embedding fundido, pesos efetivos e referências de evidência.
    Levanta:
        ValueError: se não houver referências ou se forem incompatíveis.
    """
    selected = sorted(
        (item for item in contributions if item.language_embedding is not None),
        key=lambda item: (item.observation_id, item.region_id, item.language_embedding.embedding_id),
    )
    if not selected:
        raise ValueError("language fusion requires at least one embedding reference.")
    first = selected[0].language_embedding
    assert first is not None
    signature = (first.embedding_space, first.dimension, first.producer, first.normalized)
    if any((item.language_embedding.embedding_space, item.language_embedding.dimension, item.language_embedding.producer, item.language_embedding.normalized) != signature for item in selected):
        raise ValueError("language fusion requires compatible embedding space, dimension and producer.")
    vectors = np.asarray([resolver(item.language_embedding) for item in selected], dtype=np.float64)
    if vectors.shape != (len(selected), first.dimension) or not np.isfinite(vectors).all():
        raise ValueError("language embedding resolver returned invalid vector dimensions or values.")
    raw = np.asarray([
        item.calibrated_confidence if item.calibrated_confidence is not None else (
            item.confidence if item.confidence is not None else (
                item.visual_support if item.visual_support is not None else (
                    item.region_quality if item.region_quality is not None else 1.0
                )
            )
        ) for item in selected
    ], dtype=np.float64)
    if float(raw.sum()) == 0.0:
        raw[:] = 1.0
    weights = raw / raw.sum()
    values = np.average(vectors, axis=0, weights=weights)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("language fusion produced a zero-norm embedding.")
    return FusedLanguageEmbedding(
        values=tuple(float(value / norm) for value in values), embedding_space=first.embedding_space,
        dimension=first.dimension, producer=first.producer, normalized=True,
        contributor_references=tuple(item.language_embedding for item in selected),
        effective_weights=tuple(float(value) for value in weights),
    )


# Agrega features de uma mesma geometria em espaço normalizado. Existe para a
# representação vetorial estável, separada da escolha de label e da ablação de
# coerência: callers persistem a referência resultante no módulo dono.
def fuse_point_features(contributions: Iterable[SemanticContribution]) -> FusedVisualEmbedding:
    """Funde features compatíveis por média ponderada e normalização L2.

    Argumentos:
        contributions: contribuições do mesmo ponto com features pontuais.
    Retorna:
        embedding fundido, identidade de espaço e pesos por artifact.
    Levanta:
        ValueError: se não há features ou se espaço, produtor ou dimensão divergem.
    """
    features = sorted(
        ((item, item.point_feature) for item in contributions if item.point_feature is not None),
        key=lambda pair: (pair[0].observation_id, pair[0].region_id),
    )
    if not features:
        raise ValueError("feature fusion requires at least one point feature.")
    first = features[0][1]
    assert first is not None
    signature = (first.embedding_space, first.producer, len(first.values))
    if any((feature.embedding_space, feature.producer, len(feature.values)) != signature for _, feature in features):
        raise ValueError("feature fusion requires compatible embedding space, producer and dimension.")
    raw_weights = np.asarray([
        item.calibrated_confidence if item.calibrated_confidence is not None else (
            item.confidence if item.confidence is not None else 1.0
        )
        for item, _ in features
    ], dtype=np.float64)
    if float(raw_weights.sum()) == 0.0:
        raw_weights[:] = 1.0
    weights = raw_weights / raw_weights.sum()
    values = np.average(np.asarray([feature.values for _, feature in features], dtype=np.float64), axis=0, weights=weights)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("feature fusion produced a zero-norm embedding.")
    return FusedVisualEmbedding(
        values=tuple(float(value / norm) for value in values),
        embedding_space=first.embedding_space,
        producer=first.producer,
        contributor_artifacts=tuple(feature.artifact_reference for _, feature in features),
        effective_weights=tuple(float(weight) for weight in weights),
    )


# Compara somente vetores realmente intercambiáveis. Existe para impedir que
# uma mudança de backbone seja apresentada como contradição da cena.
def measure_visual_coherence(contributions: Iterable[SemanticContribution], *, contradiction_threshold: float = 0.35) -> VisualCoherence:
    """Mede similaridade cosseno entre features compatíveis de um ponto.

    Argumentos:
        contributions: propostas de um único ponto persistente.
        contradiction_threshold: média abaixo da qual há contradição visual.
    Retorna:
        diagnóstico explícito, inclusive quando não há cobertura comparável.
    """
    if not  -1.0 <= contradiction_threshold <= 1.0:
        raise ValueError("contradiction_threshold must be in [-1, 1].")
    features = [item.point_feature for item in contributions if item.point_feature is not None]
    if len(features) < 2:
        return VisualCoherence("unavailable", 0, None, "insufficient_feature_coverage")
    spaces = {(item.embedding_space, item.producer, len(item.values)) for item in features}
    if len(spaces) != 1:
        return VisualCoherence("unavailable", 0, None, "incompatible_feature_space")
    vectors = np.asarray([item.values for item in features], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        return VisualCoherence("unavailable", 0, None, "zero_norm_feature")
    normalized = vectors / norms[:, None]
    similarities = normalized @ normalized.T
    values = similarities[np.triu_indices(len(features), k=1)]
    mean = float(values.mean())
    return VisualCoherence(
        "contradictory" if mean < contradiction_threshold else "coherent",
        int(len(values)), mean,
    )


# Funde as propostas concorrentes de um ponto do mapa em um contexto único.
# Existe porque, a partir de múltiplos keyframes, o mesmo ponto persistido é
# classificado por várias observações: sem uma regra explícita, a última
# gravação venceria por acidente de ordem de execução. É chamada pelo
# mapping-runtime ao compor um trecho contextual.
def fuse_point_contributions(
    contributions: Iterable[SemanticContribution],
    *, visual_coherence_policy: VisualCoherencePolicy = VisualCoherencePolicy.DIAGNOSTIC,
    contradiction_threshold: float = 0.35,
) -> FusedPointContext:
    """Escolhe o label primário de um ponto preservando todos os contribuintes.

    A ordenação usa confiança calibrada quando existe, confiança bruta como
    alternativa, e sinais de suporte apenas como desempate. Empates restantes
    são resolvidos pelo instante da observação e pelo identificador da região,
    para que a mesma entrada produza sempre a mesma saída.

    Argumentos:
        contributions: propostas de classificação para o mesmo ponto.
    Retorna:
        contexto fundido com label primário, concordância e contribuintes.
    Levanta:
        ValueError: se nenhuma contribuição for informada.
    """
    ranked = sorted(contributions, key=lambda item: item.ranking_key)
    if not ranked:
        raise ValueError("fusion requires at least one contribution.")
    coherence = None if visual_coherence_policy is VisualCoherencePolicy.DISABLED else measure_visual_coherence(
        ranked, contradiction_threshold=contradiction_threshold
    )
    if visual_coherence_policy is VisualCoherencePolicy.DOWNRANK_CONTRADICTIONS and coherence and coherence.state == "contradictory":
        ranked = sorted(ranked, key=lambda item: (item.point_feature is None, item.ranking_key))
    primary = ranked[0]
    agreeing = sum(1 for item in ranked if item.label == primary.label)
    return FusedPointContext(
        label=primary.label,
        observation_id=primary.observation_id,
        region_id=primary.region_id,
        confidence=(
            primary.calibrated_confidence
            if primary.calibrated_confidence is not None
            else primary.confidence
        ),
        agreement=agreeing / len(ranked),
        contributions=tuple(ranked),
        visual_coherence=coherence,
    )
