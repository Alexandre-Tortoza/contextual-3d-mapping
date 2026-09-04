"""Manifests versionados que descrevem sequências multimodais externas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .paths import validate_dataset_name

SensorKind = Literal["rgb", "lidar", "imu", "pose"]
SUPPORTED_SCHEMA_VERSION = "1.0"


# Helper de validação compartilhado por todos os dataclasses deste módulo,
# para não repetir a checagem de "campo string não vazio" em cada
# __post_init__.
def _require(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty.")


# Descreve como converter entre dois frames de coordenadas (calibração
# extrínseca/intrínseca). Existe para que sensores referenciem sua
# calibração por id em vez de embutir o transform diretamente no manifest.
@dataclass(frozen=True)
class CalibrationManifest:
    calibration_id: str
    artifact_uri: str
    source_frame: str
    target_frame: str

    def __post_init__(self) -> None:
        for field in ("calibration_id", "artifact_uri", "source_frame", "target_frame"):
            _require(getattr(self, field), field)


# Descreve uma fonte de sensor dentro de uma sequência: onde encontrar seus
# dados, em que frame/clock ela vive e qual calibração usar. Existe como a
# unidade que os adapters de dataset leem para saber quais sensores
# processar e como interpretá-los.
@dataclass(frozen=True)
class SensorSourceManifest:
    sensor_id: str
    kind: SensorKind
    artifact_uri: str
    media_type: str
    frame_id: str
    clock_id: str
    calibration_id: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        for field in ("sensor_id", "artifact_uri", "media_type", "frame_id", "clock_id"):
            _require(getattr(self, field), field)
        if self.kind not in {"rgb", "lidar", "imu", "pose"}:
            raise ValueError(f"unsupported sensor kind: {self.kind!r}.")


# Uma sequência gravada dentro de um dataset (ex: uma corrida/percurso),
# com seus sensores e calibrações. Valida no __post_init__ que cada sensor
# referencia uma calibração conhecida da própria sequência, para detectar
# manifests inconsistentes cedo, antes de qualquer adapter tentar usá-los.
@dataclass(frozen=True)
class SequenceManifest:
    sequence_id: str
    sensors: tuple[SensorSourceManifest, ...]
    calibrations: tuple[CalibrationManifest, ...] = ()
    split: str | None = None

    def __post_init__(self) -> None:
        _require(self.sequence_id, "sequence_id")
        if not self.sensors:
            raise ValueError("a sequence must declare at least one sensor source.")
        sensor_ids = [source.sensor_id for source in self.sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("sensor ids must be unique within a sequence.")
        calibration_ids = [item.calibration_id for item in self.calibrations]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise ValueError("calibration ids must be unique within a sequence.")
        available = set(calibration_ids)
        for source in self.sensors:
            if source.kind != "pose" and source.calibration_id is None:
                raise ValueError(f"sensor {source.sensor_id!r} requires calibration_id.")
            if source.calibration_id is not None and source.calibration_id not in available:
                raise ValueError(
                    f"sensor {source.sensor_id!r} references unknown calibration "
                    f"{source.calibration_id!r}."
                )


# Raiz do manifest de um dataset: sua identidade, versão de schema e as
# sequências que ele contém. É o ponto de entrada que `SyntheticDatasetAdapter`
# e outros adapters concretos recebem para saber o que está disponível e
# validar observações contra ele.
@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sequences: tuple[SequenceManifest, ...]
    schema_version: str = SUPPORTED_SCHEMA_VERSION
    source_uri: str | None = None

    def __post_init__(self) -> None:
        _require(self.dataset_id, "dataset_id")
        validate_dataset_name(self.dataset_id)
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SUPPORTED_SCHEMA_VERSION!r}."
            )
        if not self.sequences:
            raise ValueError("a dataset must declare at least one sequence.")
        ids = [sequence.sequence_id for sequence in self.sequences]
        if len(ids) != len(set(ids)):
            raise ValueError("sequence ids must be unique within a dataset.")

    # Busca uma sequência pelo id; usada por adapters e por
    # `SyntheticDatasetAdapter.observations` para validar que a sequência
    # pedida existe antes de iterar suas observações.
    def sequence(self, sequence_id: str) -> SequenceManifest:
        for sequence in self.sequences:
            if sequence.sequence_id == sequence_id:
                return sequence
        raise KeyError(f"unknown sequence {sequence_id!r}.")
