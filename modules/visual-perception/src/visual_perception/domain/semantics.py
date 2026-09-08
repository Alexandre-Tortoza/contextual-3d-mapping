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
    "HypothesisRole",
    "IDENTITY_CLAIM_KINDS",
    "RegionKind",
    "SemanticClaim",
    "UNSCORED_CLAIM_KINDS",
    "calibrated_confidence_of",
    "most_confident_claim",
    "most_supported_claim",
    "normalize_claim_value",
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
    #: Preservado para desserializar observações anteriores à #202. O contract
    #: de cena não produz mais prosa livre: ela era o canal por onde um objeto
    #: alucinado ("there is a suitcase in the foreground", que era o próprio
    #: rig) alcançava o prompt de cada região.
    SCENE_DESCRIPTION = "scene_description"
    ENVIRONMENT = "environment"
    LAYOUT = "layout"
    LIGHTING = "lighting"
    VISIBILITY = "visibility"
    NAVIGABILITY = "navigability"


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
        ClaimKind.ENVIRONMENT,
        ClaimKind.LAYOUT,
        ClaimKind.LIGHTING,
        ClaimKind.VISIBILITY,
        ClaimKind.NAVIGABILITY,
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


# Distingue a hipótese que o produtor elegeu como principal das concorrentes
# que ele registrou junto. Existe porque, até a #202, primary e alternative
# viravam claims LABEL irmãos indistinguíveis: "quem é o primary" era
# recuperado por posição na tupla em ``domain/regions.py`` e por score em
# ``semantic_merge``, e as duas políticas discordavam.
class HypothesisRole(StrEnum):
    """O papel de uma hipótese de identidade dentro de uma região.

    ``PRIMARY`` é a interpretação que o produtor elegeu; ``ALTERNATIVE`` é
    uma hipótese concorrente que ele quis registrar. Uma alternative nunca
    substitui o primary por ter score maior: quem decide é o produtor, e o
    papel preserva essa decisão de forma auditável.
    """

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"


#: Kinds que expressam a *identidade* do sujeito, e por isso carregam papel de
#: hipótese, ``category`` e ``RegionKind``. Os demais kinds descrevem
#: propriedades que coexistem com a identidade, não competem com ela.
IDENTITY_CLAIM_KINDS = frozenset({ClaimKind.LABEL})


# Normaliza o texto de um claim apenas o suficiente para deduplicação trivial.
# Existe para que o dedupe de hipóteses e a contagem de labels no diagnóstico
# usem exatamente a mesma regra; deliberadamente **não** é uma ontologia:
# ``ceiling light`` e ``ceiling light fixture`` continuam distintos.
def normalize_claim_value(value: str) -> str:
    """Devolve ``value`` em caixa baixa e com espaços colapsados.

    Argumentos:
        value: o texto do claim, como o produtor o escreveu.
    Retorna:
        a forma normalizada usada só para comparar igualdade trivial.
    """
    return " ".join(value.split()).lower()


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

    ``role``, ``category`` e ``region_kind`` só existem em claims de
    identidade (:data:`IDENTITY_CLAIM_KINDS`) e completam o contract que a
    #202 exige que sobreviva até a serialização: qual hipótese o produtor
    elegeu, qual categoria mais estável ele atribuiu, e qual a natureza da
    região independentemente do label aberto.
    """

    kind: ClaimKind
    value: str
    confidence: ConfidenceScore | None
    evidence: tuple[Evidence, ...]
    provenance: ModelProvenance
    support: SemanticSupport | None = None
    role: HypothesisRole | None = None
    category: str | None = None
    region_kind: RegionKind | None = None

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
        self._validate_identity_fields()

    # Separa a validação dos campos de identidade porque ela é condicional ao
    # kind: um claim de identidade precisa declarar seu papel, e um claim
    # descritivo não pode fingir carregar identidade.
    def _validate_identity_fields(self) -> None:
        """Valida ``role``, ``category`` e ``region_kind`` contra o kind do claim."""
        if self.kind in IDENTITY_CLAIM_KINDS:
            if self.role is None:
                raise ValueError(
                    f"SemanticClaim({self.kind.value!r}) must declare a role: primary and alternative "
                    "hypotheses stay distinguishable by contract, never by tuple position (see issue #202)."
                )
        else:
            for field_name in ("role", "category", "region_kind"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"SemanticClaim({self.kind.value!r}) must not carry {field_name}: identity fields "
                        f"belong only to {sorted(kind.value for kind in IDENTITY_CLAIM_KINDS)}."
                    )
        if self.category is not None and not self.category.strip():
            raise ValueError("SemanticClaim.category must be a non-empty string or None.")


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
