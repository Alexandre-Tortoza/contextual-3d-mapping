"""Primitivas de identidade estável usadas nos contracts de visual-perception.

Issue: #155 (regras de identidade estável de região).
"""

from __future__ import annotations

import hashlib
import json
import re

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

#: Quantos dígitos hexadecimais do SHA-256 entram num identificador derivado:
#: 16 dígitos são 64 bits, e a chance de colisão entre ``n`` identificadores da
#: mesma família é da ordem de ``n² / 2⁶⁵`` — desprezível para os milhares de
#: regiões e entidades de um run.
_DIGEST_HEX_LENGTH = 16


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


# Deriva um identificador estável a partir de uma observação e de um conjunto
# de membros. Existe como a única regra de derivação do módulo: o texto
# canônico é uma lista JSON, então nenhum caractere dentro de um componente
# consegue imitar um separador — antes, ``("a:b", ("c",))`` e
# ``("a", ("b:c",))`` produziam o mesmo id. Usada por derive_region_id e por
# derive_entity_id.
def derive_stable_identifier(prefix: str, observation_id: str, members: tuple[str, ...]) -> str:
    """Deriva ``<prefix>-<digest>`` determinístico, independente da ordem dos membros.

    Argumentos:
        prefix: família do identificador, como ``region`` ou ``entity``.
        observation_id: identidade da observação que contém os membros.
        members: identificadores dos membros; a ordem não altera o resultado.
    Retorna:
        o identificador derivado.
    Levanta:
        ValueError: se ``observation_id`` for vazio ou se não houver membros.
    """
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError(f"observation_id must be a non-empty string, got {observation_id!r}.")
    if not members:
        raise ValueError(f"A {prefix} identifier must be derived from at least one member.")
    canonical = json.dumps([observation_id, sorted(members)], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:_DIGEST_HEX_LENGTH]
    return f"{prefix}-{digest}"


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
    return derive_stable_identifier("region", observation_id, contributing_proposal_ids)
