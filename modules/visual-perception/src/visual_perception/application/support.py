"""Pequenos helpers compartilhados entre as etapas da application."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


# Gera um hash estável de qualquer configuração em dataclass (aninhada ou
# não). Existe para popular ``ModelProvenance.config_fingerprint`` em toda
# etapa (#156, #164, #165, #167) sem duplicar lógica de hashing; chamada por
# scene_context.py, region_semantics.py e relation_generation.py.
def fingerprint_of(config: Any) -> str:
    """Um hash estável de qualquer configuração em dataclass (aninhada).

    Usado para popular ``ModelProvenance.config_fingerprint`` em toda etapa
    (#156, #164, #165, #167) sem duplicar lógica de hashing.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"fingerprint_of requires a dataclass instance, got {type(config)!r}.")
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
