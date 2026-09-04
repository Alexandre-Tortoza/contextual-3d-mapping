"""Contracts do resultado da auditoria de qualidade.

Issue: #168. A lógica de auditoria em si vive em ``application/quality_audit.py``;
este módulo só define o formato estruturado e determinístico do relatório.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


# Define os dois níveis de gravidade que um achado de auditoria pode ter.
# Existe para que o auditor consiga distinguir problemas que invalidam uma
# observação (ERROR) de contradições que devem ser sinalizadas mas
# preservadas (WARNING); usada por AuditIssue.severity.
class AuditSeverity(StrEnum):
    """Quão grave é um achado de auditoria.

    ``ERROR`` significa que a observação é internamente inconsistente (ex: uma
    referência pendurada) e não deve ser tratada como canônica. ``WARNING``
    sinaliza algo que vale a pena destacar (ex: uma contradição) sem
    invalidar a observação: contradições são reportadas, não apagadas.
    """

    ERROR = "error"
    WARNING = "warning"


# Representa um único achado produzido pelo auditor de qualidade. Existe para
# que cada problema carregue sua própria severidade, código e mensagem,
# permitindo agregação em AuditResult.issues.
@dataclass(frozen=True)
class AuditIssue:
    """Um achado produzido pelo auditor de qualidade."""

    severity: AuditSeverity
    code: str
    message: str
    region_id: str | None = None
    relation_id: str | None = None

    # Garante que todo AuditIssue carregue um código e uma mensagem
    # utilizáveis, falhando cedo se algum dos dois estiver vazio.
    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("AuditIssue.code must not be empty.")
        if not self.message:
            raise ValueError("AuditIssue.message must not be empty.")


# Agrupa todos os achados de auditar uma VisualObservation. Existe para dar
# ao chamador um único objeto de onde derivar passou/falhou e as listas de
# erros/warnings, em vez de inspecionar issues manualmente.
@dataclass(frozen=True)
class AuditResult:
    """O resultado estruturado de auditar uma VisualObservation."""

    observation_id: str
    issues: tuple[AuditIssue, ...]

    # Indica se a observação pode ser tratada como canônica. Usada por quem
    # consome o resultado da auditoria para decidir se aceita a observação.
    @property
    def passed(self) -> bool:
        """Uma observação passa quando não tem nenhum issue de severidade ERROR.

        Issues de severidade WARNING (ex: claims contraditórios) não reprovam
        a auditoria: eles continuam representados e visíveis.
        """
        return not any(issue.severity is AuditSeverity.ERROR for issue in self.issues)

    # Filtra apenas os issues de severidade ERROR, para quem precisa tratar
    # só as falhas que invalidam a observação.
    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is AuditSeverity.ERROR)

    # Filtra apenas os issues de severidade WARNING, para quem precisa
    # exibir ou registrar contradições sem tratá-las como falha.
    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is AuditSeverity.WARNING)
