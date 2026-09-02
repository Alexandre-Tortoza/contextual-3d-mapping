"""Stage fingerprints and a reusable, resumable artifact cache.

Issue: #170.

A stage's fingerprint is chained: it hashes the stage's own version and
configuration together with every upstream stage's fingerprint. Changing one
stage's configuration therefore changes only that stage's fingerprint and
every fingerprint computed from it (its downstream dependents), while
sibling stages that do not consume its output keep a valid cache entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Bump when the on-disk artifact record shape changes, to reject stale
#: cache entries written by an incompatible module version.
CACHE_SCHEMA_VERSION = 1


def compute_fingerprint(
    stage_name: str,
    stage_version: str,
    config_fingerprint: str,
    upstream_fingerprints: tuple[str, ...] = (),
) -> str:
    """Compute one stage's cache fingerprint."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage_name": stage_name,
        "stage_version": stage_version,
        "config_fingerprint": config_fingerprint,
        "upstream_fingerprints": list(upstream_fingerprints),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class StageCache:
    """A resumable, disk-backed cache of completed stage results."""

    def __init__(self, cache_directory: Path) -> None:
        self.cache_directory = cache_directory

    def get(self, stage_name: str, fingerprint: str) -> dict[str, Any] | None:
        """Return the cached artifact record, or ``None`` on a cache miss."""
        path = self._record_path(stage_name, fingerprint)
        if not path.exists():
            return None
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        artifacts: dict[str, Any] = record["artifacts"]
        return artifacts

    def put(self, stage_name: str, fingerprint: str, artifacts: dict[str, Any]) -> None:
        """Durably record that ``stage_name`` completed for ``fingerprint``."""
        path = self._record_path(stage_name, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": CACHE_SCHEMA_VERSION, "artifacts": artifacts}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def is_complete(self, stage_name: str, fingerprint: str) -> bool:
        return self.get(stage_name, fingerprint) is not None

    def _record_path(self, stage_name: str, fingerprint: str) -> Path:
        return self.cache_directory / stage_name / f"{fingerprint}.json"
