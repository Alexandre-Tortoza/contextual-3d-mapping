"""Contracts de claims semânticos, confiança, evidência e proveniência.

Issue: #156.

A interpretação semântica é representada como um conjunto de *claims*
auditáveis, em vez de um único label e um único score. Múltiplos claims do
mesmo tipo, até contraditórios entre si, podem coexistir (ex: duas hipóteses
de label para uma região ambígua); nada é sobrescrito silenciosamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.references import ModelProvenance, SourceArtifactReference


# Categoriza o tipo de um claim semântico. Existe para distinguir os
# diferentes tipos de interpretação (label, atributo, condição, etc.) sem
# misturá-los com a confiança geométrica da região, que é um conceito
# separado.
class ClaimKind(StrEnum):
    """A categoria de um claim semântico.

    A confiança geométrica (qualidade da máscara/box) é um conceito
    separado, rastreado em ``ObservedRegion.geometric_confidence`` (ver
    ``domain/regions.py``), nunca misturado com a confiança semântica de um
    claim.
    """

    LABEL = "label"
    ATTRIBUTE = "attribute"
    CONDITION = "condition"
    MATERIAL = "material"
    HAZARD = "hazard"
    SCENE_TYPE = "scene_type"
    SCENE_DESCRIPTION = "scene_description"


# Categoriza o que uma região *é*, independentemente do label livre que o
# reasoner multimodal escreveu. Existe porque "chão" e "porta" exigem
# tratamento diferente do resto do pipeline: classes *stuff* (não-contáveis)
# são derivadas da geometria do LiDAR, enquanto *thing* são objetos contáveis
# vindos da percepção visual.
class RegionKind(StrEnum):
    """A natureza de uma região, separada do seu label em linguagem natural.

    ``UNKNOWN`` existe para que "o produtor não informou" nunca seja
    confundido com "o produtor informou ``thing``" — a mesma distinção que
    ``SemanticClaim.confidence = None`` faz para a confiança.
    """

    THING = "thing"
    STUFF = "stuff"
    PART = "part"
    UNKNOWN = "unknown"


# Representa um valor de confiança em [0, 1] atribuído a uma fonte. Existe
# como o formato compartilhado de confiança usado por claims e relações,
# sempre amarrado a qual fonte a produziu.
@dataclass(frozen=True)
class ConfidenceScore:
    """Um valor de confiança em ``[0, 1]`` atribuído a uma fonte."""

    value: float
    source: str

    # Valida que o valor está em [0, 1] e que a fonte não está vazia.
    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"ConfidenceScore.value must be in [0, 1], got {self.value}.")
        if not self.source:
            raise ValueError("ConfidenceScore.source must not be empty.")


# Aponta para a evidência bruta que sustenta um claim (crop, prompt/resposta,
# etc.). Existe para tornar cada claim auditável até sua evidência de
# origem, reutilizando o formato SourceArtifactReference compartilhado em
# vez de introduzir um tipo local quase idêntico.
@dataclass(frozen=True)
class Evidence:
    """Um ponteiro para a evidência bruta que sustenta um claim (crop, prompt/resposta, etc.).

    Reutiliza o formato compartilhado ``SourceArtifactReference``
    (uri/media_type/digest) também para os artifacts de evidência derivados
    próprios deste módulo, em vez de introduzir um tipo local quase
    idêntico.
    """

    description: str
    artifact: SourceArtifactReference | None = None

    # Exige uma descrição não vazia, já que uma Evidence sem descrição não
    # seria auditável por um humano.
    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("Evidence.description must not be empty.")


# Representa uma unidade auditável de interpretação semântica. Existe como o
# átomo do modelo de claims: todo label/atributo/condição inferido é um
# SemanticClaim com sua própria confiança, evidência e proveniência.
@dataclass(frozen=True)
class SemanticClaim:
    """Uma unidade auditável de interpretação semântica.

    ``confidence`` é ``None`` quando o produtor não forneceu score. Isso é
    diferente de ``ConfidenceScore(0.0)``, que é um score baixo efetivamente
    informado. Nenhum consumidor deve converter a ausência em número: a
    política de ordenação está em :func:`most_confident_claim`.
    """

    kind: ClaimKind
    value: str
    confidence: ConfidenceScore | None
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance

    # Exige um valor não vazio e ao menos uma Evidence, para que todo claim
    # seja auditável até sua origem.
    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("SemanticClaim.value must not be empty.")
        if not self.evidence:
            raise ValueError(
                f"SemanticClaim({self.kind.value!r}) must reference at least one Evidence."
            )


# Retorna os claims de um dado kind que discordam entre si (mais de um valor
# distinto). Existe para que o auditor de qualidade (#168) sinalize
# contradições em vez de resolvê-las silenciosamente.
def contradicting_claims(claims: tuple[SemanticClaim, ...], kind: ClaimKind) -> tuple[SemanticClaim, ...]:
    """Retorna os claims de ``kind`` que discordam entre si (mais de um valor distinto).

    Usada pelo auditor de qualidade (#168) para sinalizar, e não resolver
    silenciosamente, contradições.
    """
    matching = tuple(claim for claim in claims if claim.kind is kind)
    distinct_values = {claim.value for claim in matching}
    return matching if len(distinct_values) > 1 else ()


# Aplica a política de ordenação de claims não pontuadas: uma claim com score
# sempre vence uma sem score, mesmo com confiança baixíssima; se nenhuma foi
# pontuada, não há decisão a tomar. Existe como a Specification única dessa regra
# para que merge, filtro e qualquer futuro consumidor não a repliquem divergindo.
def most_confident_claim(claims: tuple[SemanticClaim, ...]) -> SemanticClaim | None:
    """Retorna a claim pontuada mais confiante, ou ``None`` se nenhuma foi pontuada.

    Uma claim sem score permanece acessível a quem auditar a região, mas nunca
    decide: ausência de confiança não é convertida em número, nem tratada como
    confiança máxima (a regressão que originou este contrato).

    Argumentos:
        claims: as claims candidatas, tipicamente todas de um mesmo kind.
    Retorna:
        a claim de maior confiança entre as pontuadas, ou ``None``.
    """
    scored = [(claim.confidence.value, claim) for claim in claims if claim.confidence is not None]
    if not scored:
        return None
    return max(scored, key=lambda scored_claim: scored_claim[0])[1]
