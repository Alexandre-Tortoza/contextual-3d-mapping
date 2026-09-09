"""Política determinística de coerência entre o conceito e a natureza de uma região.

Issues: #216 (política determinística), #204 (razão de refinamento),
#205 (reconciliação intra-frame).

Uma parede é *stuff* — matéria contínua e não contável — em qualquer cena. Um
reasoner que responde ``label: "wall", kind: "thing"`` está errado, e o erro é
verificável sem nenhum modelo. Medido nos cinco frames de referência do run
``20260908T131207Z``: **121 de 165 regiões** (73,3%) carregam um ``RegionKind``
que contradiz o próprio conceito que elas declaram.

Este módulo existe para tornar essa contradição **contável e auditável**, e é
deliberadamente construído para não poder virar outra coisa:

- ele **não** reescreve a saída do produtor. Ele emite um veredito
  (:class:`StructuralVerdict`), e quem decide o que fazer com ele é a
  reconciliação, que registra a interpretação reconciliada *ao lado* da
  original;
- ele **não** é uma tabela ``label -> kind``. Uma tabela dessas corrigiria o
  reasoner exatamente no momento em que o erro dele passou a ser mensurável.
  O que existe aqui é um conjunto pequeno e explícito de **núcleos nominais**
  cuja natureza é inequívoca, e a regra de casamento é o núcleo à direita do
  composto — não "qualquer palavra da frase".

A diferença entre as duas coisas é medível: ``ceiling light fixture`` tem
``ceiling`` como palavra e ``fixture`` como núcleo. Casar por palavra
classificaria uma luminária como superfície contínua; casar pelo núcleo a deixa
corretamente indeterminada.
"""

from __future__ import annotations

from enum import StrEnum

from visual_perception.domain.semantics import RegionKind, SemanticClaim, normalize_claim_value

__all__ = [
    "INHERENTLY_STUFF_HEAD_NOUNS",
    "StructuralVerdict",
    "expected_region_kind",
    "head_noun",
    "region_kind_verdict",
    "singularize",
]


#: Núcleos nominais cuja natureza é inequívoca: superfícies e materiais
#: contínuos, não contáveis, em qualquer cena. A lista é curta de propósito.
#: Ampliá-la até o número de contradições ficar bonito seria mascarar o erro do
#: reasoner; o caminho é melhorar a evidência e medir de novo.
INHERENTLY_STUFF_HEAD_NOUNS = frozenset(
    {
        "carpet",
        "ceiling",
        "floor",
        "flooring",
        "grass",
        "ground",
        "pavement",
        "road",
        "sky",
        "surface",
        "tile",
        "wall",
    }
)

#: Sufixos que a singularização ingênua estragaria: ``glass`` não é plural de
#: ``glas``, e ``chassis`` não é plural de ``chassi``.
_PROTECTED_PLURAL_SUFFIXES = ("ss", "us", "is", "as", "os")


# Enumera o que a regra determinística consegue afirmar sobre a natureza
# declarada de uma região. Existe para que "a regra não sabe" seja um desfecho
# nomeado, e não a ausência de um resultado: um conceito fora do conjunto
# inequívoco é indeterminado, não correto.
class StructuralVerdict(StrEnum):
    """O que a coerência estrutural afirma sobre o ``RegionKind`` declarado."""

    #: O conceito é inequívoco e o kind declarado bate com ele.
    SUPPORTS = "supports"
    #: O conceito é inequívoco e o kind declarado o contradiz.
    CONTRADICTS = "contradicts"
    #: O conceito não está no conjunto inequívoco, ou o kind não foi declarado.
    UNDETERMINED = "undetermined"


# Reduz um substantivo à sua forma singular apenas o suficiente para que
# ``trees``/``tree`` e ``tiles``/``tile`` sejam o mesmo núcleo. Existe aqui, e
# não como dependência de NLP, porque a regra que precisamos é essa e mais nada:
# uma biblioteca de lematização traria uma fronteira externa inteira para
# resolver um caso de três linhas.
def singularize(word: str) -> str:
    """Retorna ``word`` sem o ``s`` de plural, preservando sufixos protegidos.

    Argumentos:
        word: uma palavra já normalizada (caixa baixa, sem espaços extras).
    Retorna:
        a forma singular aproximada, ou a própria palavra quando a regra não
        se aplica.
    """
    if len(word) <= 3 or not word.endswith("s"):
        return word
    if word.endswith(_PROTECTED_PLURAL_SUFFIXES):
        return word
    if word.endswith("ies") and len(word) > 4:
        return f"{word[:-3]}y"
    return word[:-1]


# Extrai o núcleo nominal de um conceito composto. Existe porque compostos
# nominais em inglês são de núcleo final (``ceiling light fixture`` é um tipo de
# ``fixture``), e é essa propriedade que impede a regra de confundir uma
# luminária com o teto. Chamada por expected_region_kind.
def head_noun(concept: str) -> str:
    """Retorna o núcleo nominal singular de ``concept``, ou string vazia.

    Argumentos:
        concept: o conceito livre escrito pelo produtor.
    Retorna:
        a última palavra do composto, singularizada; string vazia quando o
        conceito não tem nenhuma palavra utilizável.
    """
    tokens = [token for token in normalize_claim_value(concept).replace(",", " ").split() if token]
    if not tokens:
        return ""
    return singularize(tokens[-1])


# Responde qual natureza um conceito exige, quando ela é inequívoca. Existe
# separada do veredito para que a reconciliação possa propor a interpretação
# corrigida sem reimplementar o casamento por núcleo. Chamada por
# region_kind_verdict e pela reconciliação intra-frame.
def expected_region_kind(claim: SemanticClaim) -> RegionKind | None:
    """Retorna a natureza exigida pelo conceito da claim, ou ``None`` se ambígua.

    O ``value`` da claim é consultado primeiro, porque é o que o produtor
    afirmou sobre o sujeito. A ``category`` só é consultada quando o label é
    indeterminado: ela é uma generalização, e uma generalização não deve
    prevalecer sobre a afirmação específica.

    Argumentos:
        claim: a claim de identidade avaliada.
    Retorna:
        ``RegionKind.STUFF`` quando o conceito é inequivocamente contínuo, ou
        ``None`` quando a regra não tem nada a dizer.
    """
    for concept in (claim.value, claim.category):
        if not concept:
            continue
        if head_noun(concept) in INHERENTLY_STUFF_HEAD_NOUNS:
            return RegionKind.STUFF
    return None


# Compara a natureza exigida pelo conceito com a que o produtor declarou.
# Existe como a Specification única dessa comparação, consumida pela auditoria
# de qualidade, pela derivação de suporte e pela reconciliação — três
# consumidores que já divergiram uma vez neste módulo quando cada um teve a
# sua cópia da regra.
def region_kind_verdict(claim: SemanticClaim) -> StructuralVerdict:
    """Retorna o veredito estrutural sobre o ``region_kind`` declarado na claim.

    Argumentos:
        claim: a claim de identidade avaliada.
    Retorna:
        o :class:`StructuralVerdict` correspondente; ``UNDETERMINED`` sempre
        que o conceito for ambíguo ou o produtor não tiver declarado o kind.
    """
    if claim.region_kind is None or claim.region_kind is RegionKind.UNKNOWN:
        return StructuralVerdict.UNDETERMINED
    expected = expected_region_kind(claim)
    if expected is None:
        return StructuralVerdict.UNDETERMINED
    return (
        StructuralVerdict.SUPPORTS
        if claim.region_kind is expected
        else StructuralVerdict.CONTRADICTS
    )
