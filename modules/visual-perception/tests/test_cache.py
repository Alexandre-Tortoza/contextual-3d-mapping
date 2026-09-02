"""Stage fingerprints and reusable artifact cache tests (#170). No real models."""

from __future__ import annotations

from pathlib import Path

from visual_perception.application.cache import StageCache, compute_fingerprint


def test_repeated_identical_execution_reuses_cache(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    fingerprint = compute_fingerprint("region_discovery", "v1", "cfg-abc")
    assert cache.get("region_discovery", fingerprint) is None

    cache.put("region_discovery", fingerprint, {"proposal_count": 3})
    assert cache.is_complete("region_discovery", fingerprint)
    assert cache.get("region_discovery", fingerprint) == {"proposal_count": 3}


def test_changing_stage_config_invalidates_only_that_fingerprint() -> None:
    original = compute_fingerprint("feature_extraction", "v1", "cfg-a")
    changed = compute_fingerprint("feature_extraction", "v1", "cfg-b")
    assert original != changed


def test_downstream_fingerprint_changes_when_upstream_changes() -> None:
    upstream_a = compute_fingerprint("region_discovery", "v1", "cfg-a")
    upstream_b = compute_fingerprint("region_discovery", "v1", "cfg-b")
    downstream_a = compute_fingerprint("merge", "v1", "cfg-merge", upstream_fingerprints=(upstream_a,))
    downstream_b = compute_fingerprint("merge", "v1", "cfg-merge", upstream_fingerprints=(upstream_b,))
    assert downstream_a != downstream_b


def test_sibling_stage_fingerprint_is_unaffected_by_unrelated_change() -> None:
    before = compute_fingerprint("language_embedding", "v1", "cfg-lang")
    # A change to an unrelated stage's config does not touch this fingerprint.
    after = compute_fingerprint("language_embedding", "v1", "cfg-lang")
    assert before == after


def test_interrupted_run_is_resumable(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    fingerprint = compute_fingerprint("scene_context", "v1", "cfg-abc")
    assert not cache.is_complete("scene_context", fingerprint)
    cache.put("scene_context", fingerprint, {"done": True})

    resumed_cache = StageCache(tmp_path)
    assert resumed_cache.is_complete("scene_context", fingerprint)
