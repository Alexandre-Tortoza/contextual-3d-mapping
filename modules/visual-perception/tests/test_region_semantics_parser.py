"""Testes da fronteira pura de parsing da resposta de região do VLM.

``parse_region_interpretation`` é determinística e sem I/O: não conhece
``ObservedRegion``, ``ImagePayload``, ``ModelProvenance`` nem ``Evidence``. É isso
que permite testar o contrato de resposta sem mocks.

A regressão que motivou esta fronteira: o parser anterior devolvia confiança
``1.0`` sempre que o VLM omitia o score, fazendo todo claim sair com confiança
máxima artificial (ver ``benchmarks/results/samples/20260904T112627Z/``).
"""

from __future__ import annotations

from typing import Any

import pytest

from visual_perception.application.region_semantics import (
    InvalidInterpretation,
    parse_region_interpretation,
)
from visual_perception.domain.semantics import RegionKind


# Constrói uma resposta válida no contrato novo, com overrides pontuais; espelha o
# helper ``response()`` do documento de especificação do passo.
def response(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "category": "obstacle",
        "label": "collapsed wooden shelving",
        "kind": "thing",
        "confidence": 0.71,
        "alternatives": [{"label": "furniture", "confidence": 0.44}],
    }
    base.update(overrides)
    return base


# --- confiança ---------------------------------------------------------------


# Caso central do passo: score ausente vira ausência explícita, nunca certeza.
def test_missing_label_confidence_is_unscored() -> None:
    assert parse_region_interpretation(response(confidence=None)).primary.confidence is None


# A chave ausente por completo é tratada igual à chave presente com None.
def test_absent_confidence_key_is_unscored() -> None:
    payload = response()
    del payload["confidence"]
    assert parse_region_interpretation(payload).primary.confidence is None


# O outro lado: quando o VLM informa o score, ele é preservado sem alteração.
def test_explicit_label_confidence_is_preserved() -> None:
    result = parse_region_interpretation(response(confidence=0.71))
    assert result.primary.confidence is not None
    assert result.primary.confidence.value == pytest.approx(0.71)


# Regressão direta: o fallback antigo devolvia ("floor", 1.0) para uma resposta
# que era só uma string.
def test_bare_string_response_is_unscored_not_certain() -> None:
    result = parse_region_interpretation("floor")
    assert result.primary.value == "floor"
    assert result.primary.confidence is None


# Uma string nua não tem como declarar kind nem category.
def test_bare_string_response_has_unknown_kind_and_no_category() -> None:
    result = parse_region_interpretation("floor")
    assert result.kind is RegionKind.UNKNOWN
    assert result.category is None


# Um score fora de [0, 1] é resposta malformada, não algo a normalizar em silêncio.
@pytest.mark.parametrize("value", [1.4, -0.2])
def test_confidence_out_of_range_is_rejected(value: float) -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation(response(confidence=value))


# Um score não numérico também é malformado.
def test_non_numeric_confidence_is_rejected() -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation(response(confidence="alta"))


# --- category / label / kind -------------------------------------------------


# Os três campos são semanticamente distintos: category é taxonomia reduzida,
# label é open-vocabulary livre. Um não substitui o outro.
def test_category_is_distinct_from_open_label() -> None:
    result = parse_region_interpretation(response())
    assert result.category == "obstacle"
    assert result.label == "collapsed wooden shelving"


# Não há taxonomia canônica versionada no módulo, então category é string livre:
# um valor fora de qualquer lista conhecida é preservado, não rejeitado.
def test_unknown_category_is_preserved_as_free_string() -> None:
    assert parse_region_interpretation(response(category="entulho-estranho")).category == (
        "entulho-estranho"
    )


# category ausente é ausência, não um valor inventado.
def test_missing_category_is_none() -> None:
    assert parse_region_interpretation(response(category=None)).category is None


# O kind declarado pelo VLM é preservado.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("thing", RegionKind.THING), ("stuff", RegionKind.STUFF), ("part", RegionKind.PART)],
)
def test_region_kind_is_preserved(raw: str, expected: RegionKind) -> None:
    assert parse_region_interpretation(response(kind=raw)).kind is expected


# Mesma regra da confiança aplicada ao kind: omissão vira UNKNOWN, jamais THING.
def test_missing_kind_is_unknown_not_thing() -> None:
    assert parse_region_interpretation(response(kind=None)).kind is RegionKind.UNKNOWN


# Um kind fora do enum é malformado — diferente de omitido.
def test_unrecognized_kind_is_rejected() -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation(response(kind="surface-ish"))


# Sem label não há interpretação: é o único campo obrigatório.
@pytest.mark.parametrize("value", [None, "", 42])
def test_missing_label_is_rejected(value: Any) -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation(response(label=value))


# --- alternatives ------------------------------------------------------------


# A hipótese primária é a declarada em 'label', mesmo quando uma alternativa tem
# confiança maior — o parser não reordena a decisão do modelo.
def test_alternatives_do_not_replace_primary_label() -> None:
    result = parse_region_interpretation(
        response(
            label="collapsed wooden shelving",
            confidence=0.31,
            alternatives=[{"label": "furniture", "confidence": 0.92}],
        )
    )
    assert result.label == "collapsed wooden shelving"
    assert result.alternatives[0].value == "furniture"


# A mesma regra de confiança ausente vale para as alternativas.
def test_alternative_without_confidence_is_unscored() -> None:
    result = parse_region_interpretation(response(alternatives=[{"label": "furniture"}]))
    assert result.alternatives[0].confidence is None


# Ausência de alternativas é tupla vazia, não None.
@pytest.mark.parametrize("value", [None, []])
def test_missing_alternatives_is_empty_tuple(value: Any) -> None:
    assert parse_region_interpretation(response(alternatives=value)).alternatives == ()


# Uma alternativa malformada invalida a resposta inteira, em vez de ser descartada
# em silêncio.
def test_malformed_alternative_is_rejected() -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation(response(alternatives=[{"confidence": 0.4}]))


# --- formato legado ----------------------------------------------------------


# O dict legado {"labels": [...]} é rejeitado: aceitá-lo manteria o fake emitindo
# um schema que a produção não usa mais, e a suíte deixaria de cobrir o contrato
# real (decisão do passo).
def test_legacy_labels_dict_is_rejected() -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation({"labels": [{"value": "door", "confidence": 0.6}]})


# Uma resposta vazia (o que o adapter produz quando o VLM devolve JSON inválido)
# é malformada.
def test_empty_response_is_rejected() -> None:
    with pytest.raises(InvalidInterpretation):
        parse_region_interpretation({})


# Um exemplo de formato copiado não é uma interpretação. Medido em `0fda5cf`:
# com um exemplo concreto no prompt, 26 das 27 regiões rotuladas "door" em
# corridor-02-000 reproduziam a assinatura inteira do exemplo. Um label ainda
# em forma de placeholder é o caso detectável dessa falha, e precisa falhar
# localmente em vez de entrar como claim.
def test_a_label_that_echoes_the_prompt_placeholder_is_rejected() -> None:
    """Um label em forma de placeholder é recusado em vez de virar claim."""
    with pytest.raises(InvalidInterpretation, match="echoes the prompt placeholder"):
        parse_region_interpretation({"label": "<one noun naming the subject>", "kind": "thing"})


# A alternativa é evidência opcional: um placeholder ecoado nela é descartado,
# e não derruba um label primário perfeitamente válido.
def test_a_placeholder_alternative_is_dropped_without_losing_the_primary_label() -> None:
    """Uma alternativa em forma de placeholder é descartada, preservando o primário."""
    interpretation = parse_region_interpretation(
        {
            "label": "ceiling vent",
            "kind": "thing",
            "alternatives": [
                {"label": "<competing noun>", "confidence": 0.2},
                {"label": "ceiling fan", "confidence": 0.3},
            ],
        }
    )
    assert interpretation.label == "ceiling vent"
    assert [hypothesis.value for hypothesis in interpretation.alternatives] == ["ceiling fan"]


# A guarda reconhece o formato, não uma lista de textos: um label legítimo que
# apenas contenha sinais de comparação continua aceito.
def test_a_legitimate_label_containing_angle_brackets_is_still_accepted() -> None:
    """Um label real não é confundido com placeholder por conter '<' no meio."""
    interpretation = parse_region_interpretation({"label": "sign <exit>", "kind": "thing"})
    assert interpretation.label == "sign <exit>"
