"""Module-level diagnostics for backend and stage failures.

Issues: #171 (surface OOM/backend failures explicitly), #186-#189 (translate
backend failures to module diagnostics without leaking backend exceptions).
"""

from __future__ import annotations


class VisualPerceptionError(Exception):
    """Base class for all explicit visual-perception failures."""


class BackendUnavailableError(VisualPerceptionError):
    """A configured backend has no real implementation available yet.

    Raised by the stub adapters in ``infrastructure/adapters/`` (#186-#189)
    until a GPU-equipped environment implements them (#190). Never raised by
    the fakes used in tests.
    """


class BackendExecutionError(VisualPerceptionError):
    """A real backend failed during inference (including out-of-memory).

    Carries no backend-specific exception type or tensor object, per the
    module's public-boundary rule.
    """


class RegionInterpretationFailure(VisualPerceptionError):
    """One region's semantic interpretation failed in isolation (#165).

    Isolated per-region so that one failure does not invalidate the rest of
    the observation.
    """

    def __init__(self, region_id: str, reason: str) -> None:
        super().__init__(f"Region {region_id!r} interpretation failed: {reason}")
        self.region_id = region_id
        self.reason = reason
