"""Shared spatial identity values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class FrameId:
    """Stable coordinate-frame name; transform semantics remain adapter-owned."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("frame id must not be empty.")

    def __str__(self) -> str:
        return self.value
