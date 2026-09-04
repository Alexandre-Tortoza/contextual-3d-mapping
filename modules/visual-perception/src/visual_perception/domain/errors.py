"""Diagnósticos de nível de módulo para falhas de backend e de estágio.

Issues: #171 (expor falhas de OOM/backend explicitamente), #186-#189 (traduzir
falhas de backend para diagnósticos de módulo sem vazar exceptions de backend).
"""

from __future__ import annotations


# Classe base para toda falha explícita de visual-perception. Existe para que
# o código consumidor possa capturar qualquer erro do módulo com um único
# except, sem depender de tipos de exception específicos de backend.
class VisualPerceptionError(Exception):
    """Classe base para todas as falhas explícitas de visual-perception."""


# Sinaliza que um backend configurado ainda não tem implementação real
# disponível. Existe para diferenciar "não implementado ainda" de uma falha
# real de execução (BackendExecutionError).
class BackendUnavailableError(VisualPerceptionError):
    """Um backend configurado ainda não tem implementação real disponível.

    Levantada pelos adapters stub em ``infrastructure/adapters/`` (#186-#189)
    até que um ambiente equipado com GPU os implemente (#190). Nunca é
    levantada pelos fakes usados em testes.
    """


# Sinaliza que um backend real falhou durante a inferência. Existe para
# encapsular qualquer falha de backend (incluindo out-of-memory) atrás de um
# único tipo de exception estável do módulo, sem vazar o tipo de exception ou
# objeto de tensor específico do backend.
class BackendExecutionError(VisualPerceptionError):
    """Um backend real falhou durante a inferência (incluindo out-of-memory).

    Não carrega nenhum tipo de exception ou objeto de tensor específico de
    backend, conforme a regra de fronteira pública do módulo.
    """


# Sinaliza que a interpretação semântica de uma região específica falhou,
# isolada das demais. Existe para que uma falha pontual (#165) não invalide
# a observação inteira.
class RegionInterpretationFailure(VisualPerceptionError):
    """A interpretação semântica de uma região falhou isoladamente (#165).

    Isolada por região para que uma falha não invalide o restante da
    observação.
    """

    # Guarda o id da região e o motivo da falha, para que o chamador consiga
    # decidir se descarta só essa região ou propaga o erro.
    def __init__(self, region_id: str, reason: str) -> None:
        super().__init__(f"Region {region_id!r} interpretation failed: {reason}")
        self.region_id = region_id
        self.reason = reason
