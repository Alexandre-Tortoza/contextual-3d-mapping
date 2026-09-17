"""Fronteira pública do módulo semantic-fusion."""

from .fusion import (
    fuse_language_embeddings,
    fuse_point_contributions,
    fuse_point_features,
    measure_visual_coherence,
)
from .geometric_support import (
    GeometricSemanticPoint,
    GeometricSupport,
    GeometryConsistencyConfig,
    GeometryConsistencyPolicy,
    SemanticNature,
    measure_geometric_support,
)
from .models import (
    FusedLanguageEmbedding,
    FusedPointContext,
    FusedVisualEmbedding,
    LanguageEmbeddingReference,
    PointVisualFeature,
    SemanticContribution,
    SpatialNeighbourhood,
    VisualCoherence,
    VisualCoherencePolicy,
)
from .spatial import LabelledPoint, measure_spatial_support

__all__ = [
    "FusedPointContext",
    "LabelledPoint",
    "SemanticContribution",
    "SpatialNeighbourhood",
    "fuse_point_contributions",
    "fuse_language_embeddings",
    "fuse_point_features",
    "measure_spatial_support",
    "measure_visual_coherence",
    "PointVisualFeature",
    "FusedVisualEmbedding",
    "FusedLanguageEmbedding",
    "LanguageEmbeddingReference",
    "VisualCoherence",
    "VisualCoherencePolicy",
    "GeometricSemanticPoint",
    "GeometricSupport",
    "GeometryConsistencyConfig",
    "GeometryConsistencyPolicy",
    "SemanticNature",
    "measure_geometric_support",
]
