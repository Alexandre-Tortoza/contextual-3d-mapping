"""Contracts de claims semânticos, confiança, evidência e proveniência.

Issues: #156 (claims auditáveis), #195 (suporte semântico calibrado).

A interpretação semântica é representada como um conjunto de *claims*
auditáveis, em vez de um único label e um único score. Múltiplos claims do
mesmo tipo, até contraditórios entre si, podem coexistir (ex: duas hipóteses
de label para uma região ambígua); nada é sobrescrito silenciosamente.

A #195 acrescenta :attr:`SemanticClaim.support`: um registro estruturado
(:class:`~visual_perception.domain.semantic_support.SemanticSupport`) que
separa score bruto, score calibrado, abstenção e falha de calibração. Score
bruto e calibrado nunca se misturam: ``confidence`` é sempre o que o
produtor informou, e o valor calibrado só existe dentro de ``support``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.confidence import ConfidenceScore
from visual_perception.domain.references import ModelProvenance, SourceArtifactReference
from visual_perception.domain.semantic_support import SemanticSupport

__all__ = [
    "ClaimKind",
    "ConfidenceScore",
    "Evidence",
    "RegionKind",
    "SemanticClaim",
    "UNSCORED_CLAIM_KINDS",
    "calibrated_confidence_of",
    "contradicting_claims",
    "most_confident_claim",
    "most_supported_claim",
]


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


#: Kinds gerados livremente por um modelo, para os quais a #195 proíbe
#: confiança bruta. Um texto descritivo aberto não tem score que signifique
#: algo antes de ser calibrado contra evidência: um número aqui seria
#: sempre uma certeza inventada. Esses claims podem receber confiança
#: através de ``support.calibrated_confidence`` (#196), nunca de
#: ``confidence``.
UNSCORED_CLAIM_KINDS = frozenset(
    {
        ClaimKind.ATTRIBUTE,
        ClaimKind.CONDITION,
        ClaimKind.MATERIAL,
        ClaimKind.HAZARD,
        ClaimKind.SCENE_DESCRIPTION,
    }
)


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

    ``raw_response_json`` preserva a resposta literal do produtor, para que
    uma claim calibrada continue auditável até o texto que a originou
    (#195). Fica ausente quando o produtor não é um modelo (ex: relações
    geométricas).
    """

    description: str
    artifact: SourceArtifactReference | None = None
    raw_response_json: str | None = None

    # Exige uma descrição não vazia, já que uma Evidence sem descrição não
    # seria auditável por um humano, e valida que a resposta bruta
    # preservada é de fato um objeto JSON.
    def __post_init__(self) -> None:
        """Rejeita descrição vazia e resposta bruta que não seja um objeto JSON."""
        if not self.description:
            raise ValueError("Evidence.description must not be empty.")
        if self.raw_response_json is not None:
            try:
                decoded = json.loads(self.raw_response_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Evidence.raw_response_json must be valid JSON: {error}.") from error
            if not isinstance(decoded, dict):
                raise ValueError("Evidence.raw_response_json must decode to a JSON object.")


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

    ``support`` é o registro estruturado de calibração (#195). Ele nunca
    substitui ``confidence``: os dois coexistem para que bruto e calibrado
    permaneçam distinguíveis na saída pública.
    """

    kind: ClaimKind
    value: str
    confidence: ConfidenceScore | None
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance
    support: SemanticSupport | None = None

    # Exige um valor não vazio e ao menos uma Evidence, para que todo claim
    # seja auditável até sua origem, e proíbe confiança bruta nos kinds
    # livremente gerados por modelo (#195).
    def __post_init__(self) -> None:
        """Valida valor, evidência e a proibição de confiança bruta descritiva."""
        if not self.value:
            raise ValueError("SemanticClaim.value must not be empty.")
        if not self.evidence:
            raise ValueError(
                f"SemanticClaim({self.kind.value!r}) must reference at least one Evidence."
            )
        if self.kind in UNSCORED_CLAIM_KINDS and self.confidence is not None:
            raise ValueError(
                f"SemanticClaim({self.kind.value!r}) must not carry a raw confidence: freely generated "
                "claims are scored only through a calibrated SemanticSupport (see issue #195)."
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
    """Retorna a claim *bruta* mais confiante, ou ``None`` se nenhuma foi pontuada.

    Esta função enxerga apenas ``confidence`` — o que o produtor informou.
    Ela nunca considera scores calibrados: para esses, use
    :func:`most_supported_claim`. Manter as duas seleções separadas é o que
    impede que bruto e calibrado se confundam na saída pública (#195).

    Argumentos:
        claims: as claims candidatas, tipicamente todas de um mesmo kind.
    Retorna:
        a claim de maior confiança bruta entre as pontuadas, ou ``None``.
    """
    scored = [(claim.confidence.value, claim) for claim in claims if claim.confidence is not None]
    if not scored:
        return None
    return max(scored, key=lambda scored_claim: scored_claim[0])[1]


# Extrai o score calibrado de uma claim, se existir. Existe para que os
# consumidores não precisem conhecer a estrutura interna de SemanticSupport
# só para ler o valor calibrado.
def calibrated_confidence_of(claim: SemanticClaim) -> ConfidenceScore | None:
    """Retorna o score calibrado de ``claim``, ou ``None`` se ela não foi calibrada."""
    if claim.support is None or not claim.support.is_calibrated:
        return None
    return claim.support.calibrated_confidence


# Seleciona a claim mais confiável entre as que a calibração declarou
# válidas. Existe como a contraparte calibrada de most_confident_claim:
# claims abstidas, falhas ou apenas brutas nunca decidem aqui, para que uma
# decisão calibrada jamais seja tomada com base em score não verificado.
def most_supported_claim(claims: tuple[SemanticClaim, ...]) -> SemanticClaim | None:
    """Retorna a claim calibrada mais confiante, ou ``None`` se nenhuma foi calibrada.

    Argumentos:
        claims: as claims candidatas, tipicamente todas de um mesmo kind.
    Retorna:
        a claim de maior confiança calibrada, ou ``None`` quando nenhuma
        claim tem suporte no estado ``calibrated``.
    """
    scored = []
    for claim in claims:
        calibrated = calibrated_confidence_of(claim)
        if calibrated is not None:
            scored.append((calibrated.value, claim))
    if not scored:
        return None
    return max(scored, key=lambda scored_claim: scored_claim[0])[1]
