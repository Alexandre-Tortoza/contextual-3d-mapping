"""Política determinística do que é evidência contextual publicável.

Este módulo responde uma pergunta só, e é a pergunta que mudou o escopo do
`visual-perception`:

```text
antes:  "que superfícies existem nesta imagem?"
depois: "que informação visualmente observável merece virar evidência
         contextual em um mapa 3D?"
```

Uma parede, um piso ou um teto genérico não é evidência contextual. Medido no
run de referência ``20260910T115810Z`` (três frames vinculantes, 150 hipóteses
de identidade afirmadas): 52 ``wall``, 42 ``ceiling``, 10 ``plain wall``, 6
``floor``, 6 ``ceiling tiles``, 5 ``wooden panel``. Cerca de 85% do output
público descrevia superfície estrutural genérica, que satura o overlay e domina
``label_counts`` sem discriminar nada.

A informação que um mapa contextual precisa preservar está **sobre** essas
superfícies — rachadura, graffiti, texto de aviso, tinta descascada, mancha,
corrosão, dano estrutural —, e não na superfície em si.

A regra, então:

    Uma região é publicada como observação contextual **a menos que** sua
    identidade afirmada seja *apenas* uma superfície estrutural genérica.

O que este módulo deliberadamente **não** faz, e por quê:

- **não substitui label.** Uma região que o produtor não conseguiu decidir
  entre ``wall`` e ``door`` sai do output público; ela nunca vira ``door``
  porque ``wall`` deixou de ser publicável. A política existe para reduzir
  falso positivo, não para trocar de classe;
- **não usa posição na imagem.** Nada de "embaixo é piso, em cima é teto":
  estes frames são fisheye, e o projeto também mira UAV e orientação arbitrária
  de câmera. A semântica estrutural 3D pertence a ``sensor-association`` e
  ``semantic-fusion``, com LiDAR, normais e gravidade;
- **não consulta ``category``.** ``category`` é uma generalização, e uma região
  cujo label é ``door`` com categoria ``wall`` está afirmando uma porta. Deixar
  a generalização suprimir a afirmação específica seria inverter o contract;
- **não consulta ``RegionKind``.** O reasoner erra a natureza em 73,3% das
  regiões (ver ``structural_consistency``); uma política apoiada nele herdaria
  esse erro;
- **não apaga nada.** A região suprimida continua na observação, em
  ``VisualObservation.structural_context``, e continua alimentando reconciliação,
  relações, audit e diagnóstico. O que muda é o que o módulo **publica**.
"""

from __future__ import annotations

from enum import StrEnum

from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import (
    ASSERTED_HYPOTHESIS_ROLES,
    ClaimKind,
    SemanticClaim,
)
from visual_perception.domain.structural_consistency import (
    INHERENTLY_STUFF_HEAD_NOUNS,
    NON_DISCRIMINATIVE_MODIFIERS,
    concept_tokens,
)

__all__ = [
    "GENERIC_STRUCTURAL_HEAD_NOUNS",
    "ContextualEvidenceVerdict",
    "RegionSuppressionReason",
    "adjectival_stem",
    "asserted_identity_claims",
    "contextual_evidence_claim",
    "contextual_evidence_verdict",
    "host_surface_concept",
    "material_roots",
]


#: Núcleos nominais de superfície estrutural genérica, para fins de
#: **publicação**. Parte de :data:`INHERENTLY_STUFF_HEAD_NOUNS`, que é o
#: invariante mínimo de ``RegionKind`` usado pelo audit, e acrescenta os núcleos
#: arquitetônicos genéricos que os runs reais produziram (``wooden panel``,
#: ``wooden plank``, ``ceiling panel``, ``baseboard``, ``molding``).
#:
#: Os dois conjuntos existem separados de propósito. O do audit precisa
#: permanecer minúsculo porque ampliá-lo mascararia o erro do reasoner no
#: momento em que ele passou a ser mensurável. Este responde outra pergunta —
#: "isto vale como evidência contextual?" — e um conceito a mais aqui não apaga
#: nenhuma medida: a região continua no contexto estrutural.
GENERIC_STRUCTURAL_HEAD_NOUNS = INHERENTLY_STUFF_HEAD_NOUNS | frozenset(
    {
        "baseboard",
        "molding",
        "panel",
        "panelling",
        "partition",
        "plank",
    }
)

#: Sufixos adjetivais que ligam um modificador ao material que ele nomeia:
#: ``wooden`` → ``wood``, ``metallic`` → ``metal``. Existe pelo mesmo motivo que
#: ``singularize``, e com a mesma ambição curta: a regra que precisamos é essa e
#: mais nada. Uma biblioteca de lematização traria uma fronteira externa inteira
#: para resolver um caso de três linhas.
_ADJECTIVAL_SUFFIXES = ("en", "ic", "ish", "y")

#: Tamanho mínimo de um radical derivado. Abaixo disso a redução deixa de ser
#: uma forma da mesma palavra e passa a ser um fragmento que casaria por acaso.
_MIN_STEM_LENGTH = 3


# Enumera o que a política consegue afirmar sobre uma região. Existe para que
# "a regra não tem base para julgar" seja um desfecho nomeado, e não a ausência
# de um resultado: uma região sem nenhuma identidade afirmada não é uma
# superfície genérica, é uma região que ninguém interpretou.
class ContextualEvidenceVerdict(StrEnum):
    """O que a política de publicação afirma sobre uma região."""

    #: A identidade afirmada nomeia algo além de superfície estrutural genérica.
    CONTEXT_BEARING = "context_bearing"
    #: Toda identidade afirmada é apenas superfície estrutural genérica.
    GENERIC_STRUCTURAL_SURFACE = "generic_structural_surface"
    #: A região não carrega identidade afirmada; não há base para julgá-la.
    UNINTERPRETED = "uninterpreted"


# Enumera por que uma região não foi publicada como observação contextual.
# Vocabulário fechado, no molde de ``ProposalRejectionReason``: o motivo do
# descarte precisa ter identidade estável para ser contável no diagnóstico, em
# vez de uma string livre por chamador.
class RegionSuppressionReason(StrEnum):
    """Por que uma região ficou no contexto estrutural em vez de ser publicada."""

    GENERIC_STRUCTURAL_SURFACE = "generic_structural_surface"


# Reduz um modificador à raiz do material que ele nomeia. Existe para que
# ``wooden floor`` seja reconhecido como piso genérico quando a própria região
# já declarou ``material: wood`` — o modificador não acrescenta evidência, ele
# repete um canal que o contract já tem. Chamada por _is_generic_structural.
def adjectival_stem(word: str) -> str:
    """Retorna ``word`` sem o sufixo adjetival, ou a própria palavra.

    A redução é aproximada de propósito, e pode encurtar demais uma palavra que
    apenas termina como um adjetivo (``plastic`` → ``plast``). Isso é inofensivo
    porque o radical só é consultado **depois** da comparação literal com os
    materiais declarados: uma palavra que já bate exatamente nunca chega aqui.

    Argumentos:
        word: uma palavra já normalizada e singularizada.
    Retorna:
        o radical aproximado, ou ``word`` quando a redução não se aplica ou
        produziria um fragmento curto demais para casar com significado.
    """
    for suffix in _ADJECTIVAL_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_LENGTH:
            return _undouble(word[: -len(suffix)])
    return word


# Desfaz a consoante dobrada que a sufixação adjetival introduz: ``metallic``
# reduz para ``metall``, e o material que ele nomeia é ``metal``. Separada para
# que a regra de sufixo e a de ortografia fiquem legíveis uma de cada vez.
def _undouble(stem: str) -> str:
    """Retorna ``stem`` sem a consoante final duplicada."""
    if len(stem) > _MIN_STEM_LENGTH and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        return stem[:-1]
    return stem


# Lista as hipóteses de identidade **afirmadas** de uma região. Existe separada
# porque a política precisa julgar todas elas, e não só a primeira: um passe de
# refinamento acrescenta uma segunda claim PRIMARY, e julgar apenas a original
# suprimiria uma região que o refinamento já tinha reinterpretado. Uma
# ALTERNATIVE nunca entra: ela é a dúvida que o produtor registrou, e promovê-la
# a razão para publicar seria exatamente a substituição de label que a política
# proíbe.
def asserted_identity_claims(region: ObservedRegion) -> tuple[SemanticClaim, ...]:
    """Retorna as claims de identidade com papel afirmado, na ordem original.

    Argumentos:
        region: a região cujas claims serão inspecionadas.
    Retorna:
        as claims de kind ``LABEL`` cujo papel está em
        :data:`ASSERTED_HYPOTHESIS_ROLES`.
    """
    return tuple(
        claim
        for claim in region.claims
        if claim.kind is ClaimKind.LABEL and claim.role in ASSERTED_HYPOTHESIS_ROLES
    )


# Reúne os materiais que a própria região declarou, já tokenizados. Existe
# porque o desconto de modificador é feito contra o que o produtor arquivou em
# ``material``, e não contra uma lista de adjetivos que este módulo inventasse.
#
# NOTE: deliberadamente **não** usa ``ClaimKind.ATTRIBUTE``. O VLM às vezes
# arquiva ``cracked``/``damaged`` em ``attributes`` em vez de ``condition``, e
# descontar atributos engoliria exatamente a evidência que esta política existe
# para preservar.
def material_roots(region: ObservedRegion) -> frozenset[str]:
    """Retorna os tokens dos materiais declarados pela região.

    Argumentos:
        region: a região cujas claims de material serão lidas.
    Retorna:
        o conjunto de tokens singularizados, vazio quando a região não declarou
        material.
    """
    return frozenset(
        token
        for claim in region.claims
        if claim.kind is ClaimKind.MATERIAL
        for token in concept_tokens(claim.value)
    )


# Decide se um conceito descreve apenas superfície estrutural genérica. É a
# regra central da política, isolada como função para que o veredito por região
# e a derivação do host surface leiam exatamente a mesma decisão.
def _is_generic_structural(
    concept: str, material: frozenset[str], head_nouns: frozenset[str]
) -> bool:
    """Indica se ``concept`` não afirma nada além de superfície estrutural."""
    tokens = concept_tokens(concept)
    if not tokens or tokens[-1] not in head_nouns:
        return False
    for token in tokens[:-1]:
        if token in head_nouns or token in NON_DISCRIMINATIVE_MODIFIERS:
            continue
        if token in material or adjectival_stem(token) in material:
            continue
        return False
    return True


# Ponto de entrada da política: responde se uma região é evidência contextual.
# Chamada por ``application/contextual_publication.py``, o único estágio que
# aplica a decisão. Ela é pública para que uma ferramenta de inspeção possa
# perguntar "por que esta região saiu?" sem reimplementar a regra — duas cópias
# da mesma política divergem em silêncio, como já aconteceu neste módulo.
def contextual_evidence_verdict(
    region: ObservedRegion, *, extra_head_nouns: frozenset[str] = frozenset()
) -> ContextualEvidenceVerdict:
    """Retorna o veredito de publicação da região.

    Uma região só é ``GENERIC_STRUCTURAL_SURFACE`` quando **todas** as suas
    hipóteses afirmadas o são. Em caso de desacordo entre elas, a região é
    publicada: a política reduz falso positivo, e suprimir uma região que
    alguma afirmação diz ser outra coisa seria criar um.

    Argumentos:
        region: a região já interpretada, calibrada e reconciliada.
        extra_head_nouns: núcleos estruturais adicionais vindos da configuração
            da composição, para ambientes cujo vocabulário este módulo não
            precisa conhecer de antemão.
    Retorna:
        o :class:`ContextualEvidenceVerdict` correspondente.
    """
    asserted = asserted_identity_claims(region)
    if not asserted:
        return ContextualEvidenceVerdict.UNINTERPRETED
    material = material_roots(region)
    head_nouns = GENERIC_STRUCTURAL_HEAD_NOUNS | extra_head_nouns
    if all(_is_generic_structural(claim.value, material, head_nouns) for claim in asserted):
        return ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE
    return ContextualEvidenceVerdict.CONTEXT_BEARING


# Devolve a hipótese que **carrega** a evidência contextual da região: a
# primeira afirmada que não é superfície estrutural genérica. Existe porque a
# região publicada pode carregar mais de uma hipótese afirmada — um passe de
# refinamento acrescenta a segunda —, e quem apresenta a região precisa mostrar
# aquela que a tornou publicável. Sem isto, uma região publicada por causa de
# ``red wall`` apareceria rotulada ``wall``, que é exatamente o label que a
# política existe para tirar da frente.
#
# Ela **não** substitui ``primary_label_claim``: aquela responde "o que o
# produtor elegeu", e continua sendo a política única de quem conta labels. Esta
# responde "o que sustenta a publicação desta região". As duas perguntas são
# diferentes, e a diferença entre elas é justamente o que o refinamento mudou.
def contextual_evidence_claim(
    region: ObservedRegion, *, extra_head_nouns: frozenset[str] = frozenset()
) -> SemanticClaim | None:
    """Retorna a hipótese afirmada que carrega a evidência contextual, ou ``None``.

    Argumentos:
        region: a região inspecionada.
        extra_head_nouns: núcleos estruturais adicionais vindos da configuração.
    Retorna:
        a primeira claim afirmada cujo conceito não é superfície estrutural
        genérica; ``None`` quando a região não tem nenhuma.
    """
    material = material_roots(region)
    head_nouns = GENERIC_STRUCTURAL_HEAD_NOUNS | extra_head_nouns
    for claim in asserted_identity_claims(region):
        if not _is_generic_structural(claim.value, material, head_nouns):
            return claim
    return None


# Deriva a superfície que hospeda a evidência, quando a própria identidade
# afirmada a nomeia. ``cracked wall`` e ``graffiti on wall`` dizem, além do
# dano, sobre o que ele está.
#
# A derivação vem **apenas** do texto que o produtor escreveu. Ela nunca vem de
# contenção geométrica com uma região vizinha: isso seria lavar geometria como
# semântica, e a associação entre uma evidência 2D e uma superfície física é
# trabalho de ``sensor-association``/``semantic-fusion``, com geometria 3D.
# Ausência de resultado significa host desconhecido, e é um desfecho legítimo.
def host_surface_concept(
    region: ObservedRegion, *, extra_head_nouns: frozenset[str] = frozenset()
) -> str | None:
    """Retorna a superfície estrutural nomeada pela identidade da região, ou ``None``.

    Argumentos:
        region: a região publicada cuja identidade será inspecionada.
        extra_head_nouns: núcleos estruturais adicionais vindos da configuração.
    Retorna:
        o conceito da superfície hospedeira quando **todas** as hipóteses
        afirmadas nomeiam exatamente a mesma; ``None`` quando nenhuma nomeia,
        quando elas discordam, ou quando o conceito cita mais de uma superfície.
    """
    asserted = asserted_identity_claims(region)
    if not asserted:
        return None
    head_nouns = GENERIC_STRUCTURAL_HEAD_NOUNS | extra_head_nouns
    hosts: set[str] = set()
    for claim in asserted:
        cited = {token for token in concept_tokens(claim.value) if token in head_nouns}
        if len(cited) != 1:
            return None
        hosts |= cited
    return hosts.pop() if len(hosts) == 1 else None
