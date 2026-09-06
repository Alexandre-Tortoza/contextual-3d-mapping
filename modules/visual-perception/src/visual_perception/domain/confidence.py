"""Contract compartilhado de valor de confiança.

Issues: #156 (claims auditáveis), #195 (suporte semântico calibrado).

Extraído de ``domain/semantics.py`` para que
``domain/semantic_support.py`` possa referenciar o mesmo formato de
confiança sem criar um ciclo de import: o suporte semântico é usado *por*
uma claim, então ele não pode depender do módulo que define a claim.
``domain.semantics`` continua reexportando :class:`ConfidenceScore`, que
permanece o único formato público de confiança do módulo.
"""

from __future__ import annotations

from dataclasses import dataclass


# Representa um valor de confiança em [0, 1] atribuído a uma fonte. Existe
# como o formato compartilhado de confiança usado por claims, relações e
# suporte calibrado, sempre amarrado a qual fonte a produziu.
@dataclass(frozen=True)
class ConfidenceScore:
    """Um valor de confiança em ``[0, 1]`` atribuído a uma fonte."""

    value: float
    source: str

    # Valida que o valor é um número real em [0, 1] e que a fonte não está
    # vazia. ``bool`` é rejeitado explicitamente porque ``True`` passaria
    # como ``1.0`` e transformaria um flag em certeza máxima — exatamente a
    # confusão que a #195 existe para impedir.
    def __post_init__(self) -> None:
        """Rejeita valores fora de ``[0, 1]``, booleanos e fontes vazias."""
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ValueError(f"ConfidenceScore.value must be a real number, got {self.value!r}.")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"ConfidenceScore.value must be in [0, 1], got {self.value}.")
        if not self.source:
            raise ValueError("ConfidenceScore.source must not be empty.")
