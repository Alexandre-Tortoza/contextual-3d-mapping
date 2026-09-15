"""Testes da derivação de identificadores estáveis (#242)."""

from __future__ import annotations

import itertools

import pytest

from visual_perception.domain.contextual_entities import derive_entity_id
from visual_perception.domain.identifiers import derive_region_id, derive_stable_identifier


# Regressão do defeito que motivou a #242: com o texto canônico
# ``observation_id:membro|membro``, um ``:`` dentro da observação ou de um
# membro — caractere permitido pelo padrão de identificador — produzia o mesmo
# id para entradas diferentes.
@pytest.mark.parametrize("derive", [derive_region_id, derive_entity_id])
def test_separator_characters_inside_components_do_not_collide(derive) -> None:
    """Entradas que só diferem pela posição de um separador geram ids diferentes."""
    assert derive("a:b", ("c",)) != derive("a", ("b:c",))
    assert derive("a", ("b|c",)) != derive("a", ("b", "c"))


# Varre combinações pequenas de componentes feitos só de separadores e letras,
# que é onde uma codificação ambígua falharia, e exige que entradas distintas
# nunca compartilhem id.
def test_distinct_inputs_never_share_an_identifier() -> None:
    """Toda entrada distinta de um alfabeto de separadores produz um id distinto."""
    alphabet = ("a", ":", "|", "a:", ":a", "a|b", '"', ",")
    seen: dict[str, tuple[str, tuple[str, ...]]] = {}
    for observation_id in alphabet:
        for size in (1, 2):
            for members in itertools.combinations(alphabet, size):
                key = (observation_id, tuple(sorted(members)))
                identifier = derive_stable_identifier("region", observation_id, members)
                assert seen.setdefault(identifier, key) == key


# A estabilidade que a #160 exige continua valendo: a ordem dos membros não
# altera o identificador.
def test_member_order_does_not_change_the_identifier() -> None:
    """O mesmo conjunto em qualquer ordem produz o mesmo id."""
    assert derive_region_id("frame-1", ("p-2", "p-1", "p-3")) == derive_region_id(
        "frame-1", ("p-3", "p-1", "p-2")
    )


# A família do identificador faz parte dele, e entradas inválidas falham com
# mensagem acionável em vez de gerar um hash de string vazia.
def test_prefix_is_preserved_and_invalid_inputs_fail() -> None:
    """O prefixo aparece no id; observação vazia e conjunto vazio são rejeitados."""
    assert derive_region_id("frame-1", ("p-1",)).startswith("region-")
    assert derive_entity_id("frame-1", ("region-1",)).startswith("entity-")
    with pytest.raises(ValueError, match="observation_id"):
        derive_stable_identifier("region", "", ("p-1",))
    with pytest.raises(ValueError, match="at least one"):
        derive_region_id("frame-1", ())
