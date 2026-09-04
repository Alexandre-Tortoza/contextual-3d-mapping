"""Sincronização determinística por timestamp mais próximo entre observações canônicas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from contextual_mapping_datasets.manifest import SensorKind
from .dataset import CanonicalObservation


# Configuração de uma rodada de sincronização: qual tipo de sensor serve de
# âncora, quais tipos são esperados em cada grupo e a tolerância de tempo
# aceita. Existe para que `synchronize` seja parametrizável sem argumentos
# soltos e para validar a configuração uma única vez, antes de agrupar.
@dataclass(frozen=True)
class SynchronizationConfig:
    anchor_kind: SensorKind
    expected_kinds: tuple[SensorKind, ...]
    tolerance_ns: int

    def __post_init__(self) -> None:
        supported = {"rgb", "lidar", "imu", "pose"}
        if self.anchor_kind not in supported:
            raise ValueError(f"unsupported anchor kind: {self.anchor_kind!r}.")
        if isinstance(self.tolerance_ns, bool) or not isinstance(self.tolerance_ns, int):
            raise TypeError("tolerance_ns must be an integer.")
        if self.tolerance_ns < 0:
            raise ValueError("tolerance_ns must be non-negative.")
        if not self.expected_kinds:
            raise ValueError("expected_kinds must not be empty.")
        if len(self.expected_kinds) != len(set(self.expected_kinds)):
            raise ValueError("expected_kinds must not contain duplicates.")
        if self.anchor_kind not in self.expected_kinds:
            raise ValueError("anchor_kind must be included in expected_kinds.")
        unknown = set(self.expected_kinds) - supported
        if unknown:
            raise ValueError(f"unsupported expected kinds: {sorted(unknown)!r}.")


# Um grupo de observações sincronizadas em torno de uma âncora, com os
# tipos que não puderam ser casados dentro da tolerância marcados como
# ausentes. Existe para que consumidores downstream (fusão) recebam grupos
# explícitos em vez de terem que casar timestamps eles mesmos.
@dataclass(frozen=True)
class SynchronizedObservationGroup:
    anchor: CanonicalObservation
    observations: tuple[CanonicalObservation, ...]
    missing_kinds: tuple[SensorKind, ...]

    # Busca a observação de um tipo específico dentro do grupo, ou None se
    # esse tipo ficou ausente (fora da tolerância).
    def observation(self, kind: SensorKind) -> CanonicalObservation | None:
        return next((item for item in self.observations if item.kind == kind), None)


# Agrupa observações canônicas de uma única sequência/clock em torno de cada
# observação âncora, casando o candidato não utilizado mais próximo de cada
# tipo esperado dentro da tolerância configurada. Usada pelos adapters de
# dataset para produzir grupos multimodais sincronizados sem reamostrar ou
# interpolar as observações originais.
def synchronize(observations: Iterable[CanonicalObservation], config: SynchronizationConfig) -> tuple[SynchronizedObservationGroup, ...]:
    """Agrupa uma sequência/clock sem alterar ou reutilizar observações de origem."""
    items = tuple(observations)
    if not items:
        return ()
    if len({item.reference.sequence_id for item in items}) != 1:
        raise ValueError("synchronization input must contain exactly one sequence.")
    if len({item.reference.timestamp.clock_id for item in items}) != 1:
        raise ValueError("synchronization input must use exactly one clock_id.")
    by_kind: dict[SensorKind, list[CanonicalObservation]] = {kind: [] for kind in config.expected_kinds}
    for item in items:
        if item.kind in by_kind:
            by_kind[item.kind].append(item)
    for candidates in by_kind.values():
        candidates.sort(key=_candidate_order)
    used: set[str] = set()
    groups: list[SynchronizedObservationGroup] = []
    for anchor in by_kind[config.anchor_kind]:
        matched = [anchor]
        missing: list[SensorKind] = []
        for kind in config.expected_kinds:
            if kind == config.anchor_kind:
                continue
            candidate = _nearest_unused(anchor, by_kind[kind], used, config.tolerance_ns)
            if candidate is None:
                missing.append(kind)
            else:
                matched.append(candidate)
                used.add(candidate.reference.observation_id)
        matched.sort(key=lambda item: config.expected_kinds.index(item.kind))
        groups.append(SynchronizedObservationGroup(anchor, tuple(matched), tuple(missing)))
    return tuple(groups)


# Encontra, entre os candidatos ainda não usados, o mais próximo em tempo
# da âncora dentro da tolerância; desempate por _candidate_order. Existe
# para isolar a lógica de casamento por proximidade usada em cada tipo
# esperado dentro de `synchronize`.
def _nearest_unused(anchor: CanonicalObservation, candidates: list[CanonicalObservation], used: set[str], tolerance_ns: int) -> CanonicalObservation | None:
    anchor_ns = anchor.reference.timestamp.nanoseconds
    eligible = [item for item in candidates if item.reference.observation_id not in used and abs(item.reference.timestamp.nanoseconds - anchor_ns) <= tolerance_ns]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (abs(item.reference.timestamp.nanoseconds - anchor_ns), *_candidate_order(item)))


# Chave de ordenação/desempate determinística de candidatos: por timestamp,
# depois índice de sequência e id de observação.
def _candidate_order(observation: CanonicalObservation) -> tuple[int, int, str]:
    reference = observation.reference
    return (reference.timestamp.nanoseconds, reference.sequence_index, reference.observation_id)
