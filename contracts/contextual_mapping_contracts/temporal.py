"""Shared temporal values.

Timestamps use integer nanoseconds on a named clock. Integer storage avoids
platform-dependent floating-point rounding during synchronization and replay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Timestamp:
    """A non-negative instant expressed as nanoseconds on ``clock_id``."""

    nanoseconds: int
    clock_id: str

    def __post_init__(self) -> None:
        if isinstance(self.nanoseconds, bool) or not isinstance(self.nanoseconds, int):
            raise TypeError("nanoseconds must be an integer.")
        if self.nanoseconds < 0:
            raise ValueError("nanoseconds must be non-negative.")
        if not self.clock_id.strip():
            raise ValueError("clock_id must not be empty.")
