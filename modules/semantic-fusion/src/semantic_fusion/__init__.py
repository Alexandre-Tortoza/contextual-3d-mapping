"""Fronteira pública do módulo semantic-fusion."""

from .fusion import fuse_point_contributions
from .models import FusedPointContext, SemanticContribution

__all__ = [
    "FusedPointContext",
    "SemanticContribution",
    "fuse_point_contributions",
]
