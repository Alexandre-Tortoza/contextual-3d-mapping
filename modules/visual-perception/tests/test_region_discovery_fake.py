"""Region discovery boundary tests using the GPU-free fake (#158)."""

from __future__ import annotations

from fixtures import blank_payload, payload_with_blobs
from visual_perception.config import RegionDiscoveryConfig
from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer


def test_zero_regions_in_blank_image() -> None:
    proposals = FakeRegionDiscoverer().discover(blank_payload(), RegionDiscoveryConfig())
    assert proposals == ()


def test_one_region_for_one_blob() -> None:
    payload = payload_with_blobs(blobs=((4, 4, 10, 10, (200, 30, 30)),))
    proposals = FakeRegionDiscoverer().discover(payload, RegionDiscoveryConfig())
    assert len(proposals) == 1
    assert proposals[0].mask.image_width == payload.width


def test_multiple_regions_for_multiple_blobs() -> None:
    payload = payload_with_blobs(
        blobs=((2, 2, 6, 6, (200, 30, 30)), (20, 20, 26, 26, (30, 200, 30)))
    )
    proposals = FakeRegionDiscoverer().discover(payload, RegionDiscoveryConfig())
    assert len(proposals) == 2


def test_proposal_geometry_follows_invariants() -> None:
    payload = payload_with_blobs(blobs=((4, 4, 10, 10, (200, 30, 30)),))
    proposal = FakeRegionDiscoverer().discover(payload, RegionDiscoveryConfig())[0]
    box = proposal.mask.bounding_box()
    assert box == proposal.box
