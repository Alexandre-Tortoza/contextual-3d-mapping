"""Referências de observação de origem e proveniência, independentes de implementação."""

from __future__ import annotations

from dataclasses import dataclass

from .spatial import FrameId
from .temporal import Timestamp


# Helper de validação compartilhado por todos os dataclasses deste módulo.
# Existe para não repetir a mesma checagem de "campo string não vazio" em
# cada __post_init__ abaixo.
def _require(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty.")


# Referência lógica a um dado de origem mantido fora do repositório (ex: um
# arquivo de imagem, nuvem de pontos ou bag). Existe porque contracts não
# devem carregar o payload bruto, só uma referência estável a ele.
@dataclass(frozen=True)
class SourceArtifactReference:
    """Referência lógica a um dado de origem mantido fora do repositório."""

    uri: str
    media_type: str
    digest: str | None = None

    def __post_init__(self) -> None:
        _require(self.uri, "uri")
        _require(self.media_type, "media_type")
        if self.digest is not None:
            _require(self.digest, "digest")


# Identidade estável de uma observação de origem (uma leitura de sensor
# específica), com os metadados necessários para interpretá-la. Existe para
# que módulos consumidores (fusão, adapters de dataset, proveniência) se
# refiram sempre à mesma observação sem duplicar sua interpretação.
@dataclass(frozen=True)
class ObservationReference:
    """Identidade estável e metadados de interpretação de uma observação de origem."""

    observation_id: str
    dataset_id: str
    sequence_id: str
    sensor_id: str
    sequence_index: int
    timestamp: Timestamp
    frame_id: FrameId
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for field in ("observation_id", "dataset_id", "sequence_id", "sensor_id"):
            _require(getattr(self, field), field)
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative.")
        if self.calibration_id is not None:
            _require(self.calibration_id, "calibration_id")


# Conjunto completo dos contribuidores de origem de um item derivado (ex:
# uma detecção semântica fundida a partir de várias observações). Existe
# para preservar proveniência um-para-muitos e muitos-para-um sem nunca
# sobrescrever contribuidores anteriores.
@dataclass(frozen=True)
class Provenance:
    """Conjunto completo e não-destrutivo dos contribuidores de origem de um item derivado."""

    producer: str
    observations: tuple[ObservationReference, ...]
    source_artifacts: tuple[SourceArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        _require(self.producer, "producer")
        if not self.observations:
            raise ValueError("provenance must contain at least one observation.")
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("provenance observation ids must be unique.")
