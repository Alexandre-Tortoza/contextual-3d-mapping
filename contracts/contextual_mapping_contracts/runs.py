"""Identidades compartilhadas de execuções (runs) persistidas em disco."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-run-(\d{3,})-([a-z0-9][a-z0-9._-]*)$")
_EXISTING_NUMBER = re.compile(r"(?:\d{4}-\d{2}-\d{2}-)?run-(\d+)-")

# Nomes de subpasta compartilhados dentro de uma run, para que nenhuma
# capacidade reinvente sufixos diferentes para o mesmo conceito (assets de
# preview e artifacts de debug ficam sempre dentro da própria pasta da run).
ASSETS_DIRNAME = "assets"
DEBUG_DIRNAME = "debug"


# Identifica uma run persistida (composição contextual, benchmark ou
# experimento) no formato canônico ``AAAA-MM-DD-run-NNN-nome``. Existe para
# que apps/cli, apps/mapping-runtime, apps/map-explorer, os benchmarks de
# visual-perception e os experiments concordem com exatamente a mesma forma
# textual, em vez de cada capacidade reimplementar seu próprio esquema.
@dataclass(frozen=True, order=True)
class RunId:
    """Identidade estável de uma run no formato ``AAAA-MM-DD-run-NNN-nome``.

    Argumentos:
        value: identificador completo já formatado.
    Levanta:
        ValueError: se ``value`` não casar com o formato canônico.
    """

    value: str

    # Rejeita qualquer id fora do formato canônico antes que alcance disco
    # ou URL publicada, evitando divergência silenciosa entre capacidades.
    def __post_init__(self) -> None:
        """Valida o formato completo do identificador."""
        if not _PATTERN.match(self.value):
            raise ValueError(f"run id inválido: {self.value!r}")

    # Fornece a forma textual usada em paths, manifests e URLs públicas.
    def __str__(self) -> str:
        """Retorna a representação textual estável da run."""
        return self.value

    # Expõe o número sequencial já validado para quem precisa ordenar ou
    # decidir o próximo número sem reanalisar a string.
    @property
    def sequence(self) -> int:
        """Retorna o número sequencial embutido na identidade."""
        match = _PATTERN.match(self.value)
        assert match is not None  # garantido por __post_init__
        return int(match.group(2))


# Calcula o próximo identificador monotônico a partir das runs já
# conhecidas. Existe para que toda capacidade que cria runs compartilhe a
# mesma numeração e o mesmo texto de data, em vez de reimplementar
# strftime + regex localmente em cada script. É pura (recebe ``today`` em
# vez de ler o relógio) para que o comportamento seja testável sem mock.
def next_run_id(existing_ids: Iterable[str], *, today: date, name: str) -> RunId:
    """Retorna o próximo ``RunId`` dado o catálogo de identidades existentes.

    Argumentos:
        existing_ids: identidades já usadas (locais e/ou publicadas), em
            qualquer formato — entradas que não casam com o padrão numerado
            são ignoradas na contagem, não rejeitadas.
        today: data a embutir no identificador.
        name: nome legível já validado (segmento, benchmark ou experimento).
    Retorna:
        identidade não usada, sequencialmente após a maior encontrada.
    """
    numbers = [int(match.group(1)) for match in map(_EXISTING_NUMBER.match, existing_ids) if match]
    return RunId(f"{today:%Y-%m-%d}-run-{max(numbers, default=0) + 1:03d}-{name}")
