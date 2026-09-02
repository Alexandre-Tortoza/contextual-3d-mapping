"""In-memory :class:`EvidencePersistencePort` fake for integration tests (#180)."""

from __future__ import annotations

from typing import Any


class InMemoryEvidenceStore:
    """A trivial store used to test persistence integration without real storage."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def put(self, artifact_id: str, payload: dict[str, Any]) -> str:
        reference = f"memory://{artifact_id}"
        self._records[reference] = payload
        return reference

    def get(self, reference: str) -> dict[str, Any]:
        return self._records[reference]
