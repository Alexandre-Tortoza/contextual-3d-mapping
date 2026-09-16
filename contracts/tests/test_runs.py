"""Testes do contract compartilhado de identidade de run."""

from datetime import date

import pytest

from contextual_mapping_contracts import RunId, next_run_id


# Confirma que o formato canônico completo é aceito e que o número
# sequencial fica acessível sem reanalisar a string.
def test_run_id_accepts_canonical_format() -> None:
    """Aceita ``AAAA-MM-DD-run-NNN-nome`` e expõe o número sequencial."""
    run_id = RunId("2026-09-16-run-007-corridor-02")

    assert str(run_id) == "2026-09-16-run-007-corridor-02"
    assert run_id.sequence == 7


@pytest.mark.parametrize(
    "value",
    [
        "run-007-corridor-02",
        "2026-09-16-corridor-02",
        "2026-09-16-run-7-corridor-02",
        "2026-09-16-run-007-Corridor-02",
        "2026-09-16-run-007-",
        "",
    ],
)
# Rejeita qualquer variação fora do formato canônico antes que alcance
# disco ou URL publicada, para que todas as capacidades concordem com a
# mesma forma textual.
def test_run_id_rejects_malformed_values(value: str) -> None:
    """Rejeita ids malformados."""
    with pytest.raises(ValueError, match="run id inválido"):
        RunId(value)


# Garante que o catálogo vazio produz a primeira run do dia.
def test_next_run_id_starts_at_one_for_empty_history() -> None:
    """Gera ``run-001`` quando não há runs anteriores."""
    result = next_run_id([], today=date(2026, 9, 16), name="corridor-02")

    assert str(result) == "2026-09-16-run-001-corridor-02"


# Confirma que a numeração avança a partir da maior run encontrada, mesmo
# com lacunas, e ignora entradas que não casam com o padrão numerado.
def test_next_run_id_increments_past_highest_existing_and_ignores_unparseable() -> None:
    """Incrementa a partir da maior run numerada existente."""
    existing = [
        "2026-09-01-run-003-corridor-02",
        "2026-09-10-run-005-corridor-02",
        "manual-publish",
        "20260916T044509Z-corridor-02",
    ]

    result = next_run_id(existing, today=date(2026, 9, 16), name="corridor-02")

    assert str(result) == "2026-09-16-run-006-corridor-02"


# Confirma que a data embutida vem do parâmetro injetado, não do relógio
# real, para que o contract permaneça testável de forma determinística.
def test_next_run_id_uses_injected_today() -> None:
    """Usa a data injetada, não ``datetime.now()``."""
    result = next_run_id([], today=date(2020, 1, 1), name="probe")

    assert str(result) == "2020-01-01-run-001-probe"
