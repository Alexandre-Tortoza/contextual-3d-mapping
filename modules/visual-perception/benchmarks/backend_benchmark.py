"""Benchmark perception backends under the reference GPU budget.

Issue: #174. **Not yet run**: this environment has no GPU (see
``docs/architecture.md`` "Real backends"), so no candidate has been
benchmarked and no default backend has been selected yet. The harness below
is ready to run once real adapters (#186-#189) exist on GPU hardware; until
then every stage's ``config.py`` default stays ``backend="fake"`` and this
issue remains open.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable

from visual_perception.application.execution_profile import BackendCandidate


def benchmark_candidate[T](
    name: str,
    factory: Callable[[], T],
    run_once: Callable[[T], float],
    *,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> BackendCandidate:
    """Measure one candidate's load time, latency, peak RSS, and quality.

    ``run_once`` executes the candidate on one representative input and
    returns a task-relevant quality score in ``[0, 1]``; the caller supplies
    it because quality scoring is dataset-specific (see ``harness.py``).

    Peak memory here is CPU-RSS, a placeholder for peak VRAM: real GPU
    measurement requires the hardware this environment does not have (#190).
    """
    model = factory()
    for _ in range(warmup_runs):
        run_once(model)

    quality_scores = []
    start = time.monotonic()
    for _ in range(measured_runs):
        quality_scores.append(run_once(model))
    latency_s = (time.monotonic() - start) / max(measured_runs, 1)

    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return BackendCandidate(
        name=name,
        quality_score=sum(quality_scores) / len(quality_scores),
        peak_vram_gb=peak_bytes / (1024**3),
        latency_s=latency_s,
    )
