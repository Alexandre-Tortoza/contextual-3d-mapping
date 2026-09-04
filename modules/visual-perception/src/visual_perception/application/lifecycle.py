"""Ciclo de vida sequencial de modelo e diagnóstico de memória.

Issue: #171.

Adapters pesados são criados pouco antes de seu stage rodar e liberados
imediatamente depois, para que o pipeline canônico nunca precise manter
todo modelo residente ao mesmo tempo. O pico de memória é VRAM CUDA real
(#190, referência RTX 3060) quando torch com uma GPU visível está
disponível, e cai para um proxy de CPU-RSS caso contrário (ex: um
ambiente de dev sem GPU usando apenas os fakes).
"""

from __future__ import annotations

import dataclasses
import gc
import resource
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from visual_perception.domain.errors import BackendExecutionError

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# torch.cuda.OutOfMemoryError is a RuntimeError subclass, not a MemoryError,
# so it needs its own catch clause alongside MemoryError below.
_OOM_EXCEPTIONS: tuple[type[BaseException], ...] = (
    (MemoryError, torch.cuda.OutOfMemoryError) if torch is not None else (MemoryError,)
)


# Agrupa os diagnósticos de tempo e memória registrados para a execução de
# um stage, produzidos por ModelLifecycleManager.stage e consumidos para
# análise de performance/benchmark.
@dataclass(frozen=True)
class StageMetrics:
    """Diagnósticos de tempo e memória registrados para a execução de um stage."""

    stage_name: str
    load_time_s: float
    inference_time_s: float
    peak_memory_bytes: int


# Cria, usa e libera um adapter pesado por stage, garantindo que apenas um
# modelo fique residente em memória por vez e centralizando a coleta de
# métricas de ciclo de vida usadas pelo pipeline canônico e por benchmarks.
class ModelLifecycleManager:
    """Cria, usa e libera um adapter pesado por stage."""

    # Inicializa a lista de métricas coletadas ao longo da execução deste
    # manager, e o slot de modelo ativo usado por get_or_load/release_active.
    def __init__(self) -> None:
        self._metrics: list[StageMetrics] = []
        self._active_key: str | None = None
        self._active_model: object | None = None

    # Expõe, de forma somente leitura, as métricas acumuladas de todos os
    # stages já executados por este manager.
    @property
    def metrics(self) -> tuple[StageMetrics, ...]:
        return tuple(self._metrics)

    # Context manager central do ciclo de vida: carrega o modelo do stage,
    # cede o controle para o chamador executar a inferência, e garante a
    # liberação do modelo e o registro de métricas mesmo em caso de erro.
    @contextmanager
    def stage[T](self, stage_name: str, factory: Callable[[], T]) -> Iterator[T]:
        """Carrega ``factory()``, cede o modelo, e então registra diagnósticos e o libera.

        Um OOM aparece como :class:`BackendExecutionError`; ele nunca
        dispara uma substituição silenciosa por outro backend ou
        configuração.
        """
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        try:
            model = factory()
        except _OOM_EXCEPTIONS as error:
            raise BackendExecutionError(
                f"Out of memory while loading stage {stage_name!r}."
            ) from error
        load_time = time.monotonic() - load_start

        infer_start = time.monotonic()
        try:
            yield model
        except _OOM_EXCEPTIONS as error:
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
            # gc.collect() antes de empty_cache(): modelos carregados com
            # device_map/accelerate (ex: 4-bit) criam hooks com referências
            # cíclicas que o refcounting sozinho não coleta — sem isso, a
            # VRAM real não é liberada mesmo depois do del (visto na prática
            # no #190: OOM a partir do 2º frame de uma sequência).
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Retorna o modelo já residente para ``key`` sem recarregar (caminho
    # rápido para chamadas repetidas do mesmo stage, ex: uma região por
    # chamada de multimodal reasoning), ou libera o modelo de uma ``key``
    # diferente e carrega o novo. Ao contrário de ``stage``, o modelo
    # permanece residente entre chamadas até uma ``key`` diferente ser
    # pedida — é o que permite que os 4 adapters reais compartilhem este
    # manager (ver ``infrastructure/adapters/factory.py``) e nunca fiquem
    # todos residentes ao mesmo tempo, mesmo com ``PerceptionPorts`` sendo
    # construído com os 4 já instanciados.
    def get_or_load[T](self, key: str, factory: Callable[[], T]) -> T:
        """Retorna o modelo residente para ``key``, trocando o modelo ativo se necessário."""
        if self._active_key == key:
            return self._active_model  # type: ignore[return-value]
        self.release_active()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        try:
            model = factory()
        except _OOM_EXCEPTIONS as error:
            raise BackendExecutionError(f"Out of memory while loading {key!r}.") from error
        self._metrics.append(
            StageMetrics(
                stage_name=key,
                load_time_s=time.monotonic() - load_start,
                inference_time_s=0.0,
                peak_memory_bytes=_peak_memory_bytes(),
            )
        )
        self._active_key = key
        self._active_model = model
        return model

    # Libera o modelo atualmente residente (se houver), esvaziando o cache
    # CUDA para que o próximo ``get_or_load`` com uma ``key`` diferente
    # tenha o orçamento de VRAM de volta. Antes de liberar, atualiza a
    # métrica da key ativa com o pico real observado desde o load (que
    # pode ter subido durante as chamadas de inferência feitas depois do
    # load, ex: múltiplas regiões usando o mesmo VLM). Chamada
    # internamente por ``get_or_load`` antes de trocar de modelo, e pode
    # ser chamada explicitamente ao final de um pipeline para não deixar
    # nada residente.
    def release_active(self) -> None:
        """Libera o modelo residente, se houver, e esvazia o cache CUDA."""
        if self._active_model is not None:
            if self._metrics and self._metrics[-1].stage_name == self._active_key:
                current_peak = _peak_memory_bytes()
                last = self._metrics[-1]
                if current_peak > last.peak_memory_bytes:
                    self._metrics[-1] = dataclasses.replace(last, peak_memory_bytes=current_peak)
            del self._active_model
            self._active_model = None
            self._active_key = None
            # Ver comentário equivalente em `stage`: quebra referências
            # cíclicas de hooks device_map/accelerate antes de liberar o
            # cache CUDA.
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


# Lê o pico de memória: VRAM CUDA real (torch.cuda.max_memory_allocated,
# resetado a cada troca de modelo em get_or_load/stage) quando torch com GPU
# visível está disponível, ou RSS de CPU como proxy caso contrário; helper
# interno usado por ModelLifecycleManager para preencher StageMetrics.
def _peak_memory_bytes() -> int:
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    # NOTE: ru_maxrss é KB no Linux e bytes no macOS; o proxy de CPU só
    # precisa ser monotônico e comparável entre stages dentro de uma mesma
    # execução.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
