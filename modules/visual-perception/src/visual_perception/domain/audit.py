"""Quality audit result contracts.

Issue: #168. Audit logic itself lives in ``application/quality_audit.py``;
this module only defines the structured, deterministic report shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AuditSeverity(StrEnum):
    """How serious an audit finding is.

    ``ERROR`` means the observation is internally inconsistent (e.g. a
    dangling reference) and should not be treated as canonical. ``WARNING``
    flags something worth surfacing (e.g. a contradiction) without
    invalidating the observation: contradictions are reported, not deleted.
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class AuditIssue:
    """One finding produced by the quality auditor."""

    severity: AuditSeverity
    code: str
    message: str
    region_id: str | None = None
    relation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("AuditIssue.code must not be empty.")
        if not self.message:
            raise ValueError("AuditIssue.message must not be empty.")


@dataclass(frozen=True)
class AuditResult:
    """The structured outcome of auditing one VisualObservation."""

    observation_id: str
    issues: tuple[AuditIssue, ...]

    @property
    def passed(self) -> bool:
        """An observation passes when it has no ERROR-severity issue.

        WARNING-severity issues (e.g. contradictory claims) do not fail the
        audit: they remain represented and visible.
        """
        return not any(issue.severity is AuditSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is AuditSeverity.ERROR)

    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is AuditSeverity.WARNING)
