"""Conceitos visuais concretos propostos para grounding em nível de cena (#277).

Um conceito é uma hipótese de busca, e não uma afirmação sobre o mundo: ele diz ao
grounding o que procurar. A interpretação de cada região continua sendo do reasoner
de região.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from visual_perception.domain.references import ModelProvenance


# Separa o que a cena contém (entidades) do que ela sinaliza (dano, obstrução,
# placa); o teto de conceitos prioriza os sinais contextuais.
class SceneConceptKind(StrEnum):
    """Tipo de conceito proposto pela descoberta de cena."""

    ENTITY = "entity"
    CONTEXTUAL_FEATURE = "contextual_feature"


# Um conceito normalizado pronto para virar prompt de grounding.
@dataclass(frozen=True)
class SceneConcept:
    """Frase nominal curta a ser procurada no frame, com confiança opcional."""

    text: str
    kind: SceneConceptKind
    confidence: float | None = None

    # Recusa conceito vazio ou confiança fora de [0, 1] na fronteira do domínio.
    def __post_init__(self) -> None:
        """Valida texto e confiança."""
        if not self.text or self.text != self.text.strip():
            raise ValueError("SceneConcept.text must be non-empty and stripped.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SceneConcept.confidence must be within [0, 1].")


# Resultado auditável da descoberta de cena de um frame: o que foi mantido, o que
# foi descartado e por quê, com a resposta bruta e a proveniência do modelo.
@dataclass(frozen=True)
class SceneConceptSet:
    """Conceitos mantidos, descartados e a proveniência da descoberta de cena."""

    concepts: tuple[SceneConcept, ...]
    discarded: tuple[tuple[str, str], ...]
    raw_response_json: str
    provenance: ModelProvenance
