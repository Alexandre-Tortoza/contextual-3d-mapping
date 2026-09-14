"""Faz benchmark dos backends de percepção sob o orçamento de GPU de referência.

Issue: #174, executado na RTX 3060 8GB de referência (veja
``docs/model-backends.md``). ``benchmark_candidate`` mede o pico real de VRAM
CUDA de cada candidato e se recusa a rodar sem CUDA: o orçamento que decide a
seleção é de VRAM, e todo candidato de ``candidates/`` roda na GPU. Até a #249
havia um fallback para o RSS do processo, que é monotônico — como todos os
candidatos rodam no mesmo processo, o pico de um candidato herdava o do
anterior e a seleção podia rejeitar um backend pelo consumo de outro. O módulo
continua importável sem os extras ``ml``.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

from visual_perception.application.execution_profile import BackendCandidate

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# Exige CUDA antes de qualquer medição. Existe porque um pico de VRAM que não
# pode ser medido não pode ser comparado ao orçamento: reportar outra grandeza
# com o mesmo nome era o que tornava a seleção dependente da ordem dos
# candidatos.
def _require_cuda() -> None:
    """Levanta quando não há GPU CUDA para medir VRAM."""
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(
            "benchmark_candidate measures peak CUDA memory against a VRAM budget; "
            "no CUDA device is available."
        )


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
    _require_cuda()
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
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Libera o modelo antes de devolver, para que o próximo candidato comece a
    # medir sem nenhuma alocação herdada deste.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return BackendCandidate(
        name=name,
        quality_score=sum(quality_scores) / len(quality_scores),
        peak_vram_gb=peak_vram_gb,
        latency_s=latency_s,
    )
