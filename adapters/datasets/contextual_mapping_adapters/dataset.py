"""Fronteira pública para normalizar datasets multimodais externos."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from contextual_mapping_contracts import ObservationReference, SourceArtifactReference
from contextual_mapping_datasets.manifest import DatasetManifest, SensorKind


# Observação canônica, independente de payload, emitida por todo adapter de
# dataset. Existe para que módulos consumidores (sincronização, fusão,
# pipelines downstream) lidem com um único formato de observação, não com o
# schema bruto de cada dataset.
@dataclass(frozen=True)
class CanonicalObservation:
    """Observação independente de payload emitida por todo adapter de dataset."""

    kind: SensorKind
    reference: ObservationReference
    artifact: SourceArtifactReference
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"rgb", "lidar", "imu", "pose"}:
            raise ValueError(f"unsupported observation kind: {self.kind!r}.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


# Protocol público que todo adapter de dataset concreto deve satisfazer.
# Existe como o contract mínimo (manifest + observações por sequência) que
# permite trocar o dataset de origem sem alterar o código que o consome.
@runtime_checkable
class MultimodalDatasetAdapter(Protocol):
    @property
    def manifest(self) -> DatasetManifest: ...

    def observations(self, sequence_id: str) -> Iterator[CanonicalObservation]: ...


# Adapter de referência, em memória, usado por testes de contract e por
# composições que não precisam de um dataset real em disco. Valida no
# construtor que cada observação é consistente com o manifest fornecido
# (sequência, sensor, clock, frame, calibração conhecidos).
class SyntheticDatasetAdapter:
    """Adapter de referência em memória para testes de contract e composições."""

    def __init__(self, manifest: DatasetManifest, observations: Iterable[CanonicalObservation]) -> None:
        self._manifest = manifest
        known_sequences = {sequence.sequence_id for sequence in manifest.sequences}
        known_sensors = {(sequence.sequence_id, sensor.sensor_id): sensor for sequence in manifest.sequences for sensor in sequence.sensors}
        normalized = tuple(observations)
        seen_ids: set[str] = set()
        for observation in normalized:
            reference = observation.reference
            if reference.dataset_id != manifest.dataset_id:
                raise ValueError(f"observation {reference.observation_id!r} belongs to dataset {reference.dataset_id!r}, expected {manifest.dataset_id!r}.")
            if reference.sequence_id not in known_sequences:
                raise ValueError(f"unknown sequence {reference.sequence_id!r}.")
            source = known_sensors.get((reference.sequence_id, reference.sensor_id))
            if source is None:
                raise ValueError(f"unknown sensor {reference.sensor_id!r}.")
            if source.kind != observation.kind:
                raise ValueError(f"sensor {reference.sensor_id!r} is {source.kind!r}, not {observation.kind!r}.")
            if reference.timestamp.clock_id != source.clock_id:
                raise ValueError(f"clock mismatch for sensor {reference.sensor_id!r}.")
            if str(reference.frame_id) != source.frame_id:
                raise ValueError(f"frame mismatch for sensor {reference.sensor_id!r}.")
            if reference.calibration_id != source.calibration_id:
                raise ValueError(f"calibration mismatch for sensor {reference.sensor_id!r}.")
            if reference.observation_id in seen_ids:
                raise ValueError(f"duplicate observation id {reference.observation_id!r}.")
            seen_ids.add(reference.observation_id)
        self._observations = normalized

    # Expõe o manifest validado do dataset, satisfazendo o Protocol
    # MultimodalDatasetAdapter.
    @property
    def manifest(self) -> DatasetManifest:
        return self._manifest

    # Retorna as observações de uma sequência em ordem determinística.
    # Existe para que consumidores (ex: synchronize) sempre iterem as
    # observações na mesma ordem, independentemente da ordem de inserção.
    def observations(self, sequence_id: str) -> Iterator[CanonicalObservation]:
        self._manifest.sequence(sequence_id)
        selected = (item for item in self._observations if item.reference.sequence_id == sequence_id)
        yield from sorted(selected, key=_observation_order)


# Chave de ordenação determinística das observações: por timestamp, depois
# sensor, sequência e id, para desempatar observações simultâneas de forma
# estável.
def _observation_order(observation: CanonicalObservation) -> tuple[int, str, int, str]:
    reference = observation.reference
    return (reference.timestamp.nanoseconds, reference.sensor_id, reference.sequence_index, reference.observation_id)
