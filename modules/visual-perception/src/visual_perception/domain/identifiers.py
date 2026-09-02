"""Stable identity primitives used across visual-perception contracts.

Issue: #155 (stable region identity rules).
"""

from __future__ import annotations

import hashlib
import re

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def validate_identifier(value: str, *, field: str) -> str:
    """Reject identifiers that are empty or contain ambiguous characters."""
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ValueError(
            f"{field} must be a non-empty identifier matching {_ID_PATTERN.pattern!r}, "
            f"got {value!r}."
        )
    return value


def derive_region_id(observation_id: str, contributing_proposal_ids: tuple[str, ...]) -> str:
    """Derive a stable, deterministic region id from its merge provenance.

    Same observation and same (order-independent) set of contributing proposals
    always produce the same id, satisfying issue #160's stability requirement.
    """
    if not contributing_proposal_ids:
        raise ValueError("A region must be derived from at least one contributing proposal.")
    canonical = "|".join(sorted(contributing_proposal_ids))
    digest = hashlib.sha256(f"{observation_id}:{canonical}".encode()).hexdigest()[:16]
    return f"region-{digest}"
