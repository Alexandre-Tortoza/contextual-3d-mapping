"""Contracts de embedding de região e identidade de espaço vetorial.

Issues: #162 (pooling visual com mask-aware), #163 (embedding alinhado a
linguagem), #193 (identidade de espaço por slot de evidência).

Embeddings visuais e alinhados a linguagem são deliberadamente dois tipos
separados: eles vivem em espaços vetoriais diferentes (e possivelmente
incompatíveis) e nunca devem ser confundidos por trás de um campo ambíguo.

:class:`EmbeddingSpace` leva essa separação um nível adiante: dois vetores
só podem ser comparados ou agregados quando declaram o mesmo modelo,
checkpoint, dimensão, modalidade e política de normalização. A #193 exige
que essa checagem seja explícita, e não uma convenção implícita entre
produtores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.identifiers import validate_identifier


# Sinaliza que dois espaços de embedding incompatíveis seriam comparados ou
# agregados. Existe como ValueError (como todas as violações de invariante
# deste módulo) para que a fronteira falhe cedo, em vez de produzir uma
# similaridade sem significado.
class IncompatibleEmbeddingSpaceError(ValueError):
    """Dois embeddings de espaços diferentes seriam comparados ou agregados."""


# Distingue de qual pipeline de encoding um vetor veio. Existe porque
# ``dimension`` e ``model_id`` iguais não bastam: um encoder visual e um
# encoder alinhado a texto podem coincidir nesses campos e ainda assim
# viver em espaços sem relação.
class EmbeddingModality(StrEnum):
    """A modalidade do encoder que produziu um vetor."""

    VISUAL_DENSE = "visual_dense"
    LANGUAGE_ALIGNED = "language_aligned"


# Identifica o espaço vetorial em que um embedding vive. Existe para que a
# compatibilidade entre slots de evidência multi-contexto (#193) seja uma
# verificação explícita, feita por :meth:`require_compatible`, em vez de uma
# suposição de quem consome os vetores.
@dataclass(frozen=True)
class EmbeddingSpace:
    """A identidade completa do espaço vetorial de um embedding."""

    model_id: str
    checkpoint: str
    dimension: int
    modality: EmbeddingModality
    normalized: bool = True

    # Valida a proveniência mínima e a dimensão declarada, já que um espaço
    # sem checkpoint não é reproduzível e um espaço sem dimensão positiva
    # não é comparável.
    def __post_init__(self) -> None:
        """Rejeita espaços sem proveniência ou com dimensão inválida."""
        if not self.model_id or not self.checkpoint:
            raise ValueError("EmbeddingSpace requires model_id and checkpoint provenance.")
        if self.dimension <= 0:
            raise ValueError("EmbeddingSpace.dimension must be positive.")

    # Responde se dois espaços são o mesmo, sem levantar exception. Usada
    # por consumidores que precisam filtrar slots compatíveis em vez de
    # falhar no primeiro incompatível.
    def is_compatible_with(self, other: EmbeddingSpace) -> bool:
        """Indica se ``other`` é exatamente o mesmo espaço vetorial."""
        return self == other

    # Impõe a compatibilidade, levantando um erro nomeado quando os espaços
    # divergem. Usada antes de qualquer comparação ou pooling entre
    # embeddings de slots de evidência diferentes (#193/#194).
    def require_compatible(self, other: EmbeddingSpace) -> None:
        """Levanta se ``other`` não for o mesmo espaço vetorial.

        Argumentos:
            other: o espaço do outro embedding envolvido na operação.
        Levanta:
            IncompatibleEmbeddingSpaceError: se os espaços divergirem.
        """
        if not self.is_compatible_with(other):
            raise IncompatibleEmbeddingSpaceError(
                f"Embedding spaces are not comparable: {self.describe()} vs {other.describe()}."
            )

    # Produz uma descrição curta e estável do espaço. Existe para mensagens
    # de erro e para os reports de benchmark/avaliação, que precisam nomear
    # o espaço sem imprimir a dataclass inteira.
    def describe(self) -> str:
        """Uma descrição curta e estável do espaço, usada em erros e reports."""
        suffix = "l2" if self.normalized else "raw"
        return f"{self.modality.value}:{self.model_id}@{self.checkpoint}/{self.dimension}/{suffix}"


# Valida que um vetor de embedding não está vazio e não contém NaN/Inf.
# Existe como helper compartilhado entre VisualEmbedding e LanguageEmbedding
# para não duplicar a mesma checagem nos dois __post_init__.
def _validate_vector(vector: tuple[float, ...], *, field_name: str) -> None:
    if not vector:
        raise ValueError(f"{field_name} must not be empty.")
    if any(math.isnan(component) or math.isinf(component) for component in vector):
        raise ValueError(f"{field_name} must be finite (no NaN/Inf).")


# Representa um embedding de região pooled a partir de features visuais densas.
# Existe para manter o espaço vetorial visual (#161/#162) separado do espaço
# alinhado a linguagem, evitando comparações sem sentido entre os dois.
@dataclass(frozen=True)
class VisualEmbedding:
    """Um embedding de região agregado (pooled) a partir de features visuais densas (#161, #162)."""

    embedding_id: str
    region_id: str
    vector: tuple[float, ...]
    dimension: int
    pooling_method: str
    feature_resolution: str
    model_id: str
    normalized: bool

    # Valida identificadores, o vetor em si, e que a dimensão declarada
    # bate com o tamanho real do vetor.
    def __post_init__(self) -> None:
        validate_identifier(self.embedding_id, field="embedding_id")
        validate_identifier(self.region_id, field="region_id")
        _validate_vector(self.vector, field_name="VisualEmbedding.vector")
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"VisualEmbedding dimension mismatch: declared {self.dimension}, "
                f"got vector of length {len(self.vector)}."
            )


# Representa um embedding de região em um espaço alinhado a texto. Existe
# separado de VisualEmbedding porque é produzido por um encoder de linguagem
# diferente e carrega proveniência de modelo/checkpoint própria.
@dataclass(frozen=True)
class LanguageEmbedding:
    """Um embedding de região em um espaço alinhado a texto (#163)."""

    embedding_id: str
    region_id: str
    vector: tuple[float, ...]
    dimension: int
    model_id: str
    checkpoint: str
    normalized: bool
    dtype: str = "float32"

    # Valida identificadores, o vetor, a dimensão declarada, e exige
    # proveniência de modelo/checkpoint (obrigatória para embeddings de
    # linguagem, ao contrário dos visuais).
    def __post_init__(self) -> None:
        validate_identifier(self.embedding_id, field="embedding_id")
        validate_identifier(self.region_id, field="region_id")
        _validate_vector(self.vector, field_name="LanguageEmbedding.vector")
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"LanguageEmbedding dimension mismatch: declared {self.dimension}, "
                f"got vector of length {len(self.vector)}."
            )
        if not self.model_id or not self.checkpoint:
            raise ValueError("LanguageEmbedding requires model_id and checkpoint provenance.")
