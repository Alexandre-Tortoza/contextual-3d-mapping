"""Sequential model lifecycle and memory diagnostics.

Issue: #171.

Heavyweight adapters are created just before their stage runs and released
immediately after, so the canonical pipeline never needs every model
resident at once. Peak memory is recorded per stage as a CPU-RSS proxy in
this GPU-free environment; real peak-VRAM measurement is part of the
real-hardware validation in #190.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from visual_perception.domain.errors import BackendExecutionError


@dataclass(frozen=True)
class StageMetrics:
    """Timing and memory diagnostics recorded for one stage execution."""

    stage_name: str
    load_time_s: float
    inference_time_s: float
    peak_memory_bytes: int


class ModelLifecycleManager:
    """Creates, uses, and releases one heavyweight adapter per stage."""

    def __init__(self) -> None:
        self._metrics: list[StageMetrics] = []

    @property
    def metrics(self) -> tuple[StageMetrics, ...]:
        return tuple(self._metrics)

    @contextmanager
    def stage[T](self, stage_name: str, factory: Callable[[], T]) -> Iterator[T]:
        """Load ``factory()``, yield it, then record diagnostics and release it.

        OOM surfaces as :class:`BackendExecutionError`; it never triggers a
        silent substitution of a different backend or configuration.
        """
        load_start = time.monotonic()
        try:
            model = factory()
        except MemoryError as error:
            raise BackendExecutionError(
                f"Out of memory while loading stage {stage_name!r}."
            ) from error
        load_time = time.monotonic() - load_start

        infer_start = time.monotonic()
        try:
            yield model
        except MemoryError as error:
            raise BackendExecutionError(
                f"Out of memory while running stage {stage_name!r}."
            ) from error
        finally:
            inference_time = time.monotonic() - infer_start
            self._metrics.append(
                StageMetrics(
                    stage_name=stage_name,
                    load_time_s=load_time,
                    inference_time_s=inference_time,
                    peak_memory_bytes=_peak_memory_bytes(),
                )
            )
            del model


def _peak_memory_bytes() -> int:
    # ru_maxrss is KB on Linux, bytes on macOS; this module only needs to be
    # monotonic and comparable across stages within one run.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
