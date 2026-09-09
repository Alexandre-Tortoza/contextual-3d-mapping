"""Política de exclusividade mútua entre claims semânticos.

Issues: #202 (política de slot exclusivo), #215 (alternativa não é
contradição). Existe como a Specification única de "estes dois claims se
excluem", consumida tanto pela derivação de ``contradiction_support``
(``application/semantic_calibration.py``) quanto pela auditoria de qualidade
(``application/quality_audit.py``). Duas implementações da mesma regra já
divergiram uma vez neste módulo, e esta é a que decide se uma contradição é
real.

A política anterior tratava como contraditório qualquer par de claims do mesmo
``ClaimKind`` com texto diferente. Medido em ``corridor-02-002``, isso fazia
``smooth`` contradizer a ``description`` da mesma região e marcava as 60
regiões do frame com ``contradiction_support`` inflado. Contradição depende do
*slot* ser mutuamente exclusivo, não de os valores serem diferentes: uma
parede pode ser branca, lisa e danificada ao mesmo tempo, e um objeto composto
pode ter mais de um material.

A correção da #202 resolveu metade do problema e deixou a outra metade de pé.
Medido no run ``20260908T131207Z``: **163 de 165 regiões** saíram com o warning
``contradictory_claims``, e ``contradiction_support`` teve média **1,000** em
quatro dos cinco frames. A causa era estrutural: o prompt pede uma lista
``alternatives``, o modelo devolve quase sempre exatamente uma, e o par
primária/alternativa era contado como contradição.

Uma alternativa **não é** uma contradição. É a incerteza que o próprio produtor
declarou, na mesma resposta, sobre a mesma evidência — o oposto de duas fontes
independentes discordando. Confundir as duas coisas destruía justamente o sinal
que deveria dirigir o refinamento: ``select_refinement_targets`` filtrava por
``contradictory_claims`` e, com isso, selecionaria 163 das 165 regiões.

A política atual é, portanto, mais estreita e mais informativa: dentro de um
kind de identidade, **apenas hipóteses afirmadas** (:data:`ASSERTED_HYPOTHESIS_ROLES`)
competem entre si. A ambiguidade entre uma primária e suas alternativas continua
representada — por ``HypothesisSupportSignal`` com status ``indistinguishable``,
que é medida, e não uma consequência de existir uma lista.
"""

from __future__ import annotations

from visual_perception.domain.semantics import (
    ASSERTED_HYPOTHESIS_ROLES,
    IDENTITY_CLAIM_KINDS,
    ClaimKind,
    SemanticClaim,
    normalize_claim_value,
)

__all__ = [
    "EXCLUSIVE_CLAIM_KINDS",
    "MUTUALLY_EXCLUSIVE_STATES",
    "claims_compete",
    "competing_claims",
    "contradicting_claims",
]


#: Kinds cujo slot admite um único valor verdadeiro por sujeito. Identidade é
#: exclusiva por definição (a região não é uma parede *e* uma porta), e uma
#: cena tem um tipo só. Todo kind fora deste conjunto descreve uma propriedade
#: que coexiste com as demais.
EXCLUSIVE_CLAIM_KINDS = frozenset(IDENTITY_CLAIM_KINDS | {ClaimKind.SCENE_TYPE})

#: Grupos de estados declarados como mutuamente incompatíveis. Deliberadamente
#: pequeno e explícito: ``condition`` é texto aberto, e derivar exclusividade
#: de "os valores são diferentes" transformaria cada condição nova em
#: contradição. Um valor fora de qualquer grupo coexiste com os demais.
MUTUALLY_EXCLUSIVE_STATES: tuple[frozenset[str], ...] = (
    frozenset({"open", "closed"}),
    frozenset({"on", "off"}),
    frozenset({"wet", "dry"}),
)


# Decide se dois claims disputam o mesmo slot semântico e portanto não podem
# ser verdadeiros ao mesmo tempo. Existe como o único lugar que responde essa
# pergunta; chamada por competing_claims, por derive_support_inputs e pela
# auditoria de qualidade.
def claims_compete(claim: SemanticClaim, other: SemanticClaim) -> bool:
    """Indica se ``claim`` e ``other`` são hipóteses mutuamente exclusivas.

    Dois claims só competem quando ocupam o mesmo slot exclusivo e afirmam
    valores diferentes. Kinds diferentes descrevem eixos diferentes e nunca
    competem entre si.

    Em kinds de identidade há uma condição adicional: os dois claims precisam
    **afirmar** a identidade da região. Uma ``ALTERNATIVE`` é dúvida declarada
    pelo próprio produtor e uma ``RECONCILED`` descende da primária; nenhuma
    das duas é uma segunda observação independente, e tratá-las como
    contradição tornava o sinal constante (ver a docstring do módulo).

    Argumentos:
        claim: o claim avaliado.
        other: o claim candidato a contradizê-lo.
    Retorna:
        ``True`` quando os dois não podem ser verdadeiros simultaneamente.
    """
    if claim is other or claim.kind is not other.kind:
        return False
    value, other_value = normalize_claim_value(claim.value), normalize_claim_value(other.value)
    if value == other_value:
        return False
    if claim.kind in IDENTITY_CLAIM_KINDS:
        return (
            claim.role in ASSERTED_HYPOTHESIS_ROLES and other.role in ASSERTED_HYPOTHESIS_ROLES
        )
    if claim.kind in EXCLUSIVE_CLAIM_KINDS:
        return True
    if claim.kind is ClaimKind.CONDITION:
        return any(
            value in group and other_value in group for group in MUTUALLY_EXCLUSIVE_STATES
        )
    return False


# Lista, dentro de um conjunto de claims, os que competem com um claim dado.
# Existe para que os consumidores não repitam o laço de comparação nem a regra
# de "quem é irmão"; chamada pela calibração e pela auditoria.
def competing_claims(
    claim: SemanticClaim, claims: tuple[SemanticClaim, ...]
) -> tuple[SemanticClaim, ...]:
    """Retorna os claims de ``claims`` que contradizem ``claim``.

    Argumentos:
        claim: o claim avaliado.
        claims: o conjunto candidato, tipicamente todos os claims do sujeito.
    Retorna:
        os claims mutuamente exclusivos com ``claim``, na ordem original.
    """
    return tuple(other for other in claims if claims_compete(claim, other))


# Retorna os claims de um dado kind que se contradizem entre si. Existe para
# que o auditor de qualidade (#168) sinalize contradições em vez de resolvê-las
# silenciosamente, usando exatamente a mesma política de exclusividade que a
# calibração — antes da #202 eram duas regras separadas, ambas erradas do mesmo
# jeito. Vive aqui, e não em ``domain/semantics.py``, porque depende de
# ``claims_compete`` e o caminho inverso criaria um ciclo de import.
def contradicting_claims(
    claims: tuple[SemanticClaim, ...], kind: ClaimKind
) -> tuple[SemanticClaim, ...]:
    """Retorna os claims de ``kind`` que são mutuamente exclusivos entre si.

    Um kind não exclusivo nunca produz contradição, por mais valores
    diferentes que carregue: três atributos distintos descrevem o mesmo
    sujeito, não hipóteses concorrentes.

    Argumentos:
        claims: os claims do sujeito auditado.
        kind: o kind cujo slot será inspecionado.
    Retorna:
        os claims envolvidos em ao menos uma contradição real, na ordem
        original, ou uma tupla vazia quando não há nenhuma.
    """
    matching = tuple(claim for claim in claims if claim.kind is kind)
    return tuple(claim for claim in matching if competing_claims(claim, matching))
