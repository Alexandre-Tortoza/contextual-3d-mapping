"""Contracts públicos da fusão semântica ancorada em geometria."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


# Valida uma medida opcional no intervalo unitário. Existe porque confiança e
# sinais de suporte chegam do pipeline visual como floats opcionais, e um valor
# fora de faixa indicaria erro de produção, não incerteza.
def _unit_interval(value: float | None, name: str) -> float | None:
    """Valida uma medida opcional em [0, 1]."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite value in [0, 1].")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1].")
    return number


# Separa o diagnóstico da política que pode agir sobre ele. Existe para que a
# primeira ablação exponha contradição visual sem inventar uma soma de scores.
class VisualCoherencePolicy(StrEnum):
    """Política de uso da coerência visual multi-view."""

    DISABLED = "disabled"
    DIAGNOSTIC = "diagnostic"
    DOWNRANK_CONTRADICTIONS = "downrank_contradictions"


# Transporta uma feature pontual com identidade suficiente para comparação
# segura. O vetor fica no artifact de execução, não no payload do viewer.
@dataclass(frozen=True)
class PointVisualFeature:
    """Feature visual amostrada no pixel associado a um ponto.

    Argumentos:
        values: vetor denso finito, normalizado pelo produtor quando declarado.
        embedding_space: espaço comparável somente a features de mesmo valor.
        producer: estágio produtor e configuração efetiva.
        artifact_reference: artifact que permite reabrir a evidência bruta.
    """

    values: tuple[float, ...]
    embedding_space: str
    producer: str
    artifact_reference: str

    # Recusa vetores e proveniência incompletos antes da fusão multi-view.
    def __post_init__(self) -> None:
        """Valida vetor, espaço de embedding e proveniência da feature."""
        if not self.values or not all(isfinite(float(value)) for value in self.values):
            raise ValueError("PointVisualFeature.values must be a non-empty finite vector.")
        for name in ("embedding_space", "producer", "artifact_reference"):
            if not getattr(self, name).strip():
                raise ValueError(f"PointVisualFeature.{name} must not be empty.")


# Referencia a evidência CLIP sem carregar o vetor no contract de fusão. Existe
# porque a representação alinhada à linguagem pode ser grande e pertence ao
# archive do produtor, enquanto a fusão só precisa resolvê-la no momento exato.
@dataclass(frozen=True)
class LanguageEmbeddingReference:
    """Referência rastreável a um embedding alinhado à linguagem.

    Argumentos:
        embedding_id: chave do vetor no archive imutável do produtor.
        archive_uri: localização auditável do archive que contém a chave.
        embedding_space: identidade completa e estável do espaço CLIP.
        dimension: dimensão declarada do vetor.
        producer: modelo e configuração que produziram a evidência.
        normalized: se o vetor declarado usa norma L2 unitária.
    """

    embedding_id: str
    archive_uri: str
    embedding_space: str
    dimension: int
    producer: str
    normalized: bool = True

    # Rejeita referências incompletas antes que um consumidor tente abrir um
    # archive arbitrário ou compare espaços semanticamente incompatíveis.
    def __post_init__(self) -> None:
        """Valida identidade, espaço, dimensão e proveniência da referência."""
        if any(not getattr(self, name).strip() for name in ("embedding_id", "archive_uri", "embedding_space", "producer")):
            raise ValueError("LanguageEmbeddingReference requires complete identity and provenance.")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("LanguageEmbeddingReference.dimension must be positive.")
        if not isinstance(self.normalized, bool):
            raise ValueError("LanguageEmbeddingReference.normalized must be boolean.")


# Expõe a medição sem a confundir com confiança, agreement ou suporte espacial.
@dataclass(frozen=True)
class VisualCoherence:
    """Diagnóstico de similaridade visual entre contribuições do mesmo ponto."""

    state: str
    compared_count: int
    mean_cosine_similarity: float | None
    reason: str | None = None

    # Mantém o payload pequeno e não permite publicar uma métrica sem pares.
    def __post_init__(self) -> None:
        """Valida o diagnóstico de coerência visual."""
        if self.state not in {"unavailable", "coherent", "contradictory"}:
            raise ValueError("VisualCoherence.state is invalid.")
        if self.compared_count < 0:
            raise ValueError("VisualCoherence.compared_count must be non-negative.")
        if self.mean_cosine_similarity is not None and not -1.0 <= self.mean_cosine_similarity <= 1.0:
            raise ValueError("VisualCoherence.mean_cosine_similarity must be in [-1, 1].")


# Materializa a agregação de embeddings compatíveis fora dos payloads do viewer.
# Existe para que a representação vetorial estável da #222 não se confunda com
# o diagnóstico de coerência visual usado na política categórica.
@dataclass(frozen=True)
class FusedVisualEmbedding:
    """Embedding normalizado e rastreável fundido de múltiplas observações."""

    values: tuple[float, ...]
    embedding_space: str
    producer: str
    contributor_artifacts: tuple[str, ...]
    effective_weights: tuple[float, ...]

    # Impede agregar espaços ou vetores inválidos e preserva peso por fonte.
    def __post_init__(self) -> None:
        """Valida vetor normalizado, referências e pesos da fusão."""
        if not self.values or not all(isfinite(float(value)) for value in self.values):
            raise ValueError("FusedVisualEmbedding.values must be a non-empty finite vector.")
        if not self.embedding_space.strip() or not self.producer.strip():
            raise ValueError("FusedVisualEmbedding requires embedding space and producer.")
        if len(self.contributor_artifacts) != len(self.effective_weights) or not self.contributor_artifacts:
            raise ValueError("FusedVisualEmbedding contributors and weights must agree.")
        if any(weight < 0.0 or not isfinite(weight) for weight in self.effective_weights):
            raise ValueError("FusedVisualEmbedding weights must be finite and non-negative.")


# Publica a agregação CLIP sem confundi-la com features DINO densas. O payload
# com vetor é persistido em archive pelo semantic-map, não no JSON contextual.
@dataclass(frozen=True)
class FusedLanguageEmbedding:
    """Embedding alinhado à linguagem fundido para uma geometria persistente."""

    values: tuple[float, ...]
    embedding_space: str
    dimension: int
    producer: str
    normalized: bool
    contributor_references: tuple[LanguageEmbeddingReference, ...]
    effective_weights: tuple[float, ...]

    # Impede publicar vetor sem provenance suficiente para futura reabertura.
    def __post_init__(self) -> None:
        """Valida vetor, espaço e contribuição da fusão de linguagem."""
        if len(self.values) != self.dimension or not self.values or not all(isfinite(float(value)) for value in self.values):
            raise ValueError("FusedLanguageEmbedding values must be finite and match dimension.")
        if not self.embedding_space.strip() or not self.producer.strip():
            raise ValueError("FusedLanguageEmbedding requires space and producer.")
        if len(self.contributor_references) != len(self.effective_weights) or not self.contributor_references:
            raise ValueError("FusedLanguageEmbedding contributors and weights must agree.")


# Descreve a vizinhança usada para medir suporte espacial. A aresta é um
# parâmetro porque depende da densidade do mapa: em um mapa esparso demais a
# vizinhança fica vazia, e em um denso demais ela atravessa superfícies.
@dataclass(frozen=True)
class SpatialNeighbourhood:
    """Vizinhança de voxels usada para medir concordância local de label.

    Argumentos:
        voxel_edge_m: aresta do voxel em metros; a vizinhança é o bloco 3×3×3
            de voxels ao redor do ponto.
        minimum_neighbours: quantidade mínima de vizinhos rotulados para que o
            suporte seja definido.
    """

    voxel_edge_m: float = 0.30
    minimum_neighbours: int = 4

    # Uma aresta não positiva não define uma grade, e um mínimo não positivo
    # aceitaria medir suporte a partir de nenhuma evidência.
    def __post_init__(self) -> None:
        """Valida os parâmetros da vizinhança."""
        if isinstance(self.voxel_edge_m, bool) or not isfinite(float(self.voxel_edge_m)) or self.voxel_edge_m <= 0.0:
            raise ValueError("voxel_edge_m must be a finite positive length.")
        if isinstance(self.minimum_neighbours, bool) or not isinstance(self.minimum_neighbours, int) or self.minimum_neighbours < 1:
            raise ValueError("minimum_neighbours must be at least one.")


# Representa uma classificação proposta por uma observação para um ponto do
# mapa. Existe porque, com múltiplos keyframes, o mesmo ponto passa a receber
# várias propostas concorrentes, e nenhuma delas pode ser descartada sem
# registro: a fusão escolhe uma primária, não apaga as demais.
@dataclass(frozen=True)
class SemanticContribution:
    """Classificação proposta por uma observação visual para um ponto.

    Argumentos:
        observation_id: observação visual que produziu a proposta.
        timestamp_ns: instante da observação, usado como desempate determinístico.
        region_id: região visual que cobre o ponto naquela observação.
        label: claim de label bruto proposto pela região.
        confidence: confiança declarada pelo produtor, não calibrada.
        calibrated_confidence: confiança calibrada, quando o produtor a fornece.
        support_state: estado de suporte declarado para o claim.
        visual_support: fração de evidência visual favorável ao claim.
        region_quality: qualidade geométrica da máscara da região.
    """

    observation_id: str
    timestamp_ns: int
    region_id: str
    label: str
    confidence: float | None = None
    calibrated_confidence: float | None = None
    support_state: str | None = None
    visual_support: float | None = None
    region_quality: float | None = None
    point_feature: PointVisualFeature | None = None
    language_embedding: LanguageEmbeddingReference | None = None

    # Rejeita contribuições sem identidade ou com medidas impossíveis antes que
    # elas influenciem a escolha do label primário.
    def __post_init__(self) -> None:
        """Valida identidade, label e medidas da contribuição."""
        if not self.observation_id.strip():
            raise ValueError("observation_id must not be empty.")
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        for name in ("confidence", "calibrated_confidence", "visual_support", "region_quality"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))

    # Ordena propostas usando primeiro a confiança calibrada, quando ela existe,
    # e caindo para a confiança bruta. Os sinais de suporte só desempatam,
    # porque uma soma ponderada arbitrária deles fingiria uma calibração que
    # este pipeline ainda não tem.
    @property
    def ranking_key(self) -> tuple[float, float, float, int, str]:
        """Chave de ordenação determinística entre contribuições concorrentes."""
        effective = self.calibrated_confidence
        if effective is None:
            effective = self.confidence if self.confidence is not None else 0.0
        return (
            -effective,
            -(self.visual_support if self.visual_support is not None else 0.0),
            -(self.region_quality if self.region_quality is not None else 0.0),
            self.timestamp_ns,
            self.region_id,
        )


# Resultado da fusão para um ponto do mapa. Preserva a lista completa de
# contribuintes porque a concordância entre observações é evidência, e porque
# um label primário sem seus concorrentes não é auditável.
@dataclass(frozen=True)
class FusedPointContext:
    """Contexto semântico fundido de um ponto de geometria persistente.

    Argumentos:
        label: label primário escolhido entre as contribuições.
        observation_id: observação que sustentou o label primário.
        region_id: região que sustentou o label primário.
        confidence: confiança declarada pela contribuição primária.
        agreement: fração de contribuições que concordam com o label primário.
        contributions: todas as contribuições, na ordem de ranqueamento.
    """

    label: str
    observation_id: str
    region_id: str
    confidence: float | None
    agreement: float
    contributions: tuple[SemanticContribution, ...]
    visual_coherence: VisualCoherence | None = None

    # Expõe a quantidade de observações que enxergaram o ponto sem obrigar o
    # consumidor a contar a tupla, que é a pergunta feita pelo viewer.
    @property
    def contribution_count(self) -> int:
        """Quantidade de contribuições consideradas na fusão."""
        return len(self.contributions)
