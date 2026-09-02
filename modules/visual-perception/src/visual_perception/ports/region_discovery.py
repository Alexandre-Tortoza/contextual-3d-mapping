"""Replaceable region discovery boundary.

Issue: #158.

Implementations discover image regions independently from semantic
reasoning: they return geometry and a geometric confidence only, never a
semantic label. This boundary must stay satisfiable by a GPU-free fake (see
``infrastructure/fakes/fake_region_discoverer.py``) and must never leak
backend-specific tensor/model types.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal


class RegionDiscoverer(Protocol):
    """Discovers class-agnostic, promptable, or dense region proposals."""

    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Return zero, one, or many proposals in ``image``-local coordinates."""
        ...
