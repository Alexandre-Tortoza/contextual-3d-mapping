"""Contracts de embedding de região.

Issues: #162 (pooling visual com mask-aware), #163 (embedding alinhado a linguagem).

Embeddings visuais e alinhados a linguagem são deliberadamente dois tipos
separados: eles vivem em espaços vetoriais diferentes (e possivelmente
incompatíveis) e nunca devem ser confundidos por trás de um campo ambíguo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from visual_perception.domain.identifiers import validate_identifier


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
