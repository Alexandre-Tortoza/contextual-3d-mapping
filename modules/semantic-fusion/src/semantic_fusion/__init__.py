"""Fronteira pública do módulo semantic-fusion."""

from .fusion import fuse_point_contributions
from .models import FusedPointContext, SemanticContribution, SpatialNeighbourhood
from .spatial import LabelledPoint, measure_spatial_support

__all__ = [
    "FusedPointContext",
    "LabelledPoint",
    "SemanticContribution",
    "SpatialNeighbourhood",
    "fuse_point_contributions",
    "measure_spatial_support",
]
