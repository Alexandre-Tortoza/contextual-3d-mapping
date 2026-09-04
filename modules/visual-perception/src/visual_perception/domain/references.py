"""Referências de nível de repositório de que este módulo depende, mais a proveniência de modelo própria.

``ObservationReference`` e ``SourceArtifactReference`` vêm de
`contextual_mapping_contracts` (#99/#100): identidade, timing, frame e
referências de source-artifact estáveis são um conceito de nível de
repositório, não de posse de visual-perception.

``ModelProvenance`` é deliberadamente mantido local e distinto de
``contextual_mapping_contracts.Provenance``: o tipo compartilhado liga um
item derivado de volta às *observações de origem* de onde ele veio, enquanto
este tipo registra *qual modelo/estágio/configuração* produziu um claim ou
uma relação semântica (checkpoint, versão de prompt, fingerprint de config).
Eles respondem perguntas diferentes e não devem ser confundidos.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextual_mapping_contracts import ObservationReference, SourceArtifactReference

__all__ = ["ModelProvenance", "ObservationReference", "SourceArtifactReference"]


# Registra qual modelo/estágio/configuração produziu um claim, embedding ou
# relação derivados. Existe separado do Provenance compartilhado do
# repositório porque responde "quem processou" em vez de "de onde veio o
# dado bruto".
@dataclass(frozen=True)
class ModelProvenance:
    """Qual modelo/estágio/configuração produziu um claim, embedding ou relação derivados."""

    stage: str
    producer: str
    config_fingerprint: str
    model_id: str | None = None
    checkpoint: str | None = None
    prompt_version: str | None = None

    # Exige que stage, producer e config_fingerprint estejam presentes,
    # já que proveniência incompleta tornaria um claim não auditável.
    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must not be empty.")
        if not self.producer:
            raise ValueError("producer must not be empty.")
        if not self.config_fingerprint:
            raise ValueError("config_fingerprint must not be empty.")
