"""Faz benchmark dos backends de percepção sob o orçamento de GPU de referência.

Issue: #174, executado na RTX 3060 8GB de referência (veja
``docs/model-backends.md``). ``benchmark_candidate`` mede o pico real de VRAM
CUDA quando torch com uma GPU visível está disponível, e cai para CPU-RSS
caso contrário (ex: ao fazer benchmark de um candidato CPU-only, ou em um
ambiente de dev GPU-free), para que este módulo continue importável sem os
extras ``ml``.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable

from visual_perception.application.execution_profile import BackendCandidate

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# Mede o pico de uso de memória do processo atual: usa o pico real de VRAM CUDA
# quando torch com GPU está disponível, senão cai para o pico de RSS da CPU (via
# resource.getrusage) como proxy. Usada por benchmark_candidate para reportar
# peak_vram_gb de cada candidato.
def _peak_memory_gb() -> float:
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return peak_bytes / (1024**3)


# Executa uma rodada de warmup e medição para um candidato de backend (definido
# pela tripla factory/run_once), medindo latência média e pico de memória.
# Existe para dar ao harness de seleção de backend (#174,
# application/execution_profile.py) um jeito uniforme de comparar candidatos
# heterogêneos (SAM, DINOv2, CLIP, Qwen-VL, ...) sob o mesmo protocolo de
# medição. Chamada pelos módulos em benchmarks/candidates/ e por
# run_backend_benchmark.py.
def benchmark_candidate[T](
    name: str,
    factory: Callable[[], T],
    run_once: Callable[[T], float],
    *,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> BackendCandidate:
    """Mede o tempo de load, a latência, o pico de VRAM e a qualidade de um candidato.

    ``run_once`` executa o candidato em uma entrada representativa e retorna um
    score de qualidade relevante para a tarefa em ``[0, 1]``; quem chama fornece
    esse score porque a avaliação de qualidade é específica do dataset (veja
    ``harness.py``).

    O pico de VRAM cobre o load do modelo mais toda execução de warmup/medição,
    combinando com a forma como um chamador real experimentaria o pior caso de
    uso de memória.
    """
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = factory()
    for _ in range(warmup_runs):
        run_once(model)

    quality_scores = []
    start = time.monotonic()
    for _ in range(measured_runs):
        quality_scores.append(run_once(model))
    latency_s = (time.monotonic() - start) / max(measured_runs, 1)

    return BackendCandidate(
        name=name,
        quality_score=sum(quality_scores) / len(quality_scores),
        peak_vram_gb=_peak_memory_gb(),
        latency_s=latency_s,
    )
