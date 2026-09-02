"""Small helpers shared across application stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def fingerprint_of(config: Any) -> str:
    """A stable hash of any (nested) dataclass configuration.

    Used to populate ``ModelProvenance.config_fingerprint`` for every stage
    (#156, #164, #165, #167) without duplicating hashing logic.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"fingerprint_of requires a dataclass instance, got {type(config)!r}.")
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
