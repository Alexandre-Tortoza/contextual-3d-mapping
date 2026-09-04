"""Primitivas de identidade estável usadas nos contracts de visual-perception.

Issue: #155 (regras de identidade estável de região).
"""

from __future__ import annotations

import hashlib
import re

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


# Valida o formato de um identificador de string, rejeitando valores vazios
# ou com caracteres ambíguos. Existe como validação compartilhada usada por
# todo dataclass do módulo que carrega um id (VisualEmbedding, RegionProposal,
# CandidateRelation, etc.).
def validate_identifier(value: str, *, field: str) -> str:
    """Rejeita identificadores vazios ou que contenham caracteres ambíguos."""
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ValueError(
            f"{field} must be a non-empty identifier matching {_ID_PATTERN.pattern!r}, "
            f"got {value!r}."
        )
    return value


# Deriva um id de região estável e determinístico a partir de sua
# proveniência de merge. Existe para que o mesmo conjunto de propostas
# contribuintes sempre gere o mesmo region_id, permitindo re-execução
# determinística e deduplicação; usada por region_merge.
def derive_region_id(observation_id: str, contributing_proposal_ids: tuple[str, ...]) -> str:
    """Deriva um id de região estável e determinístico a partir de sua proveniência de merge.

    A mesma observação e o mesmo conjunto (independente de ordem) de
    propostas contribuintes sempre produzem o mesmo id, satisfazendo o
    requisito de estabilidade da issue #160.
    """
    if not contributing_proposal_ids:
        raise ValueError("A region must be derived from at least one contributing proposal.")
    canonical = "|".join(sorted(contributing_proposal_ids))
    digest = hashlib.sha256(f"{observation_id}:{canonical}".encode()).hexdigest()[:16]
    return f"region-{digest}"
