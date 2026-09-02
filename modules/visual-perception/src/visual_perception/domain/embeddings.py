"""Region embedding contracts.

Issues: #162 (mask-aware visual pooling), #163 (language-aligned embedding).

Visual and language-aligned embeddings are deliberately two separate types:
they live in different (and possibly incompatible) vector spaces and must
never be confused behind one ambiguous field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from visual_perception.domain.identifiers import validate_identifier


def _validate_vector(vector: tuple[float, ...], *, field_name: str) -> None:
    if not vector:
        raise ValueError(f"{field_name} must not be empty.")
    if any(math.isnan(component) or math.isinf(component) for component in vector):
        raise ValueError(f"{field_name} must be finite (no NaN/Inf).")


@dataclass(frozen=True)
class VisualEmbedding:
    """A region embedding pooled from dense visual features (#161, #162)."""

    embedding_id: str
    region_id: str
    vector: tuple[float, ...]
    dimension: int
    pooling_method: str
    feature_resolution: str
    model_id: str
    normalized: bool

    def __post_init__(self) -> None:
        validate_identifier(self.embedding_id, field="embedding_id")
        validate_identifier(self.region_id, field="region_id")
        _validate_vector(self.vector, field_name="VisualEmbedding.vector")
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"VisualEmbedding dimension mismatch: declared {self.dimension}, "
                f"got vector of length {len(self.vector)}."
            )


@dataclass(frozen=True)
class LanguageEmbedding:
    """A region embedding in a text-aligned space (#163)."""

    embedding_id: str
    region_id: str
    vector: tuple[float, ...]
    dimension: int
    model_id: str
    checkpoint: str
    normalized: bool
    dtype: str = "float32"

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
