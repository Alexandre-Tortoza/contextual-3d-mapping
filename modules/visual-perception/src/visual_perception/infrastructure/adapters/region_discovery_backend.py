"""Real region discovery backend adapter.

Issue: #186. Blocked on #174 (benchmark-driven backend selection) and a
GPU-equipped environment; see ``docs/architecture.md`` "Real backends".
"""

from __future__ import annotations

from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.errors import BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal


class RealRegionDiscoveryAdapter:
    """Satisfies :class:`~visual_perception.ports.region_discovery.RegionDiscoverer`.

    Not implemented yet: no backend has been benchmark-selected (#174), and
    this environment has no GPU to run one. Use
    ``infrastructure.fakes.fake_region_discoverer.FakeRegionDiscoverer`` for
    tests and development.
    """

    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        raise BackendUnavailableError(
            f"Region discovery backend {config.backend!r} is not implemented in this "
            "environment (no GPU, no benchmark-selected checkpoint). See issue #186."
        )
