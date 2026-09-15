"""Ciclo de vida de modelo residente e diagnóstico de memória.

Issue: #171.

Adapters pesados permanecem residentes entre chamadas e entre stages
enquanto couberem na VRAM disponível; a liberação (LRU) só acontece de
forma reativa, quando um load novo esgota a memória de fato. O pico de
memória é VRAM CUDA real (#190, referência RTX 3060) quando torch com uma
GPU visível está disponível, e cai para um proxy de CPU-RSS caso
contrário (ex: um ambiente de dev sem GPU usando apenas os fakes).
"""

from __future__ import annotations

import dataclasses
import gc
import resource
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from visual_perception.domain.errors import BackendExecutionError

try:
    torch: Any = import_module("torch")
except ModuleNotFoundError:
    torch = None

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
    memory_kind: str = "cpu_rss"
    peak_vram_bytes: int | None = None
    peak_cpu_rss_bytes: int = 0


# Cria, usa e mantém residentes múltiplos adapters pesados, liberando o
# menos recentemente usado apenas quando a VRAM real esgota, e centraliza
# a coleta de métricas de ciclo de vida usadas pelo pipeline canônico e
# por benchmarks.
class ModelLifecycleManager:
    """Cria e mantém residentes adapters pesados, com eviction reativa a OOM."""

    # Inicializa a lista de métricas coletadas ao longo da execução deste
    # manager, e o cache de modelos residentes usado por get_or_load
    # (ordenado por recência de uso: o primeiro item é o candidato a LRU).
    def __init__(self) -> None:
        self._metrics: list[StageMetrics] = []
        self._resident: dict[str, object] = {}

    # Expõe, de forma somente leitura, as métricas acumuladas de todos os
    # stages já executados por este manager.
    @property
    def metrics(self) -> tuple[StageMetrics, ...]:
        return tuple(self._metrics)

    # Expõe, de forma somente leitura, as keys atualmente residentes em
    # ordem de recência (menos recente primeiro), para diagnóstico e testes.
    @property
    def resident_keys(self) -> tuple[str, ...]:
        return tuple(self._resident.keys())

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
                    memory_kind=_memory_kind(),
                    peak_vram_bytes=_peak_vram_bytes(),
                    peak_cpu_rss_bytes=_peak_cpu_rss_bytes(),
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
    # chamada de multimodal reasoning, ou stages diferentes intercalados
    # no mesmo frame). Modelos residentes de outras keys não são tocados:
    # tudo que já carregou fica residente até um load novo esgotar a VRAM
    # de verdade, quando então o menos recentemente usado é liberado (ver
    # ``_evict_lru``) e o load é repetido.
    def get_or_load[T](self, key: str, factory: Callable[[], T]) -> T:
        """Retorna o modelo residente para ``key``, carregando (com eviction sob OOM) se necessário."""
        if key in self._resident:
            model = self._resident.pop(key)
            self._resident[key] = model  # move para o fim: agora é o mais recente
            return model  # type: ignore[return-value]
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        model = self._load_with_eviction(key, factory)
        self._metrics.append(
            StageMetrics(
                stage_name=key,
                load_time_s=time.monotonic() - load_start,
                inference_time_s=0.0,
                peak_memory_bytes=_peak_memory_bytes(),
                memory_kind=_memory_kind(),
                peak_vram_bytes=_peak_vram_bytes(),
                peak_cpu_rss_bytes=_peak_cpu_rss_bytes(),
            )
        )
        self._resident[key] = model
        return model

    # Tenta ``factory()``; se faltar memória de verdade e ainda houver
    # modelos residentes, libera o menos recentemente usado e tenta de
    # novo. Só propaga o OOM quando não sobra mais nada para liberar.
    def _load_with_eviction[T](self, key: str, factory: Callable[[], T]) -> T:
        """Carrega via ``factory``, liberando residentes por LRU sob OOM real."""
        while True:
            try:
                return factory()
            except _OOM_EXCEPTIONS as error:
                if not self._resident:
                    raise BackendExecutionError(
                        f"Out of memory while loading {key!r}."
                    ) from error
                self._evict(next(iter(self._resident)))

    # Executa ``call`` (uma inferência sobre um modelo já residente para
    # ``key``), liberando outros residentes por LRU quando a VRAM esgota de
    # verdade durante a chamada — não só durante o load. Sem isso, um OOM em
    # inferência (ex: generate() do VLM, ou o forward do gerador de
    # máscaras) ficava permanente: o modelo alvo já está em cache,
    # get_or_load não é chamado de novo, e nenhuma eviction reativa era
    # acionada (visto na prática: uma cena com SAM+VLM residentes no limite
    # da 3060 falhava em todo frame subsequente, sem recuperação).
    def call_with_eviction[T](self, key: str, call: Callable[[], T]) -> T:
        """Executa ``call``, liberando outros residentes por LRU sob OOM real."""
        while True:
            try:
                result = call()
            except _OOM_EXCEPTIONS as error:
                candidates = [candidate for candidate in self._resident if candidate != key]
                if not candidates:
                    raise BackendExecutionError(
                        f"Out of memory while running {key!r}."
                    ) from error
                self._evict(candidates[0])
                continue
            # Uma inferência com muitas máscaras/tokens intermediários (ex: o
            # automatic mask generation do SAM) deixa blocos "reserved mas não
            # alocados" na allocator cache do modelo residente — pesos ficam,
            # só o scratch transitório sobra. Sem devolver isso aqui, o
            # próximo modelo a carregar (ex: o VLM, na mesma volta do
            # pipeline) enxerga menos memória livre do que existe de verdade
            # (visto na prática: SAM sozinho reserva ~2.8GB a mais do que usa
            # depois de gerar máscaras, e isso bastava para o Qwen estourar).
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result

    # Libera uma key residente específica, atualizando sua métrica com o
    # pico real observado desde o load (pode ter subido durante chamadas
    # de inferência feitas depois do load, ex: múltiplas regiões usando o
    # mesmo VLM) antes de descartar o modelo.
    def _evict(self, key: str) -> None:
        """Libera o modelo residente de ``key`` e esvazia o cache CUDA."""
        model = self._resident.pop(key)
        for index in range(len(self._metrics) - 1, -1, -1):
            if self._metrics[index].stage_name == key:
                current_peak = _peak_memory_bytes()
                last = self._metrics[index]
                self._metrics[index] = dataclasses.replace(
                    last,
                    peak_memory_bytes=max(current_peak, last.peak_memory_bytes),
                    peak_vram_bytes=_peak_vram_bytes(),
                    peak_cpu_rss_bytes=_peak_cpu_rss_bytes(),
                )
                break
        del model
        # gc.collect() antes de empty_cache(): modelos carregados com
        # device_map/accelerate (ex: 4-bit) criam hooks com referências
        # cíclicas que o refcounting sozinho não coleta — sem isso, a
        # VRAM real não é liberada mesmo depois do del (visto na prática
        # no #190: OOM a partir do 2º frame de uma sequência).
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Libera todos os modelos residentes. Chamada por adapters ou
    # benchmarks que precisam de um estado limpo para uma medição
    # isolada; não é mais chamada internamente por ``get_or_load``, já
    # que a eviction agora é reativa e granular (``_evict``/LRU).
    def release_all(self) -> None:
        """Libera todos os modelos residentes e esvazia o cache CUDA."""
        for key in tuple(self._resident.keys()):
            self._evict(key)


# Lê o pico de memória: VRAM CUDA real (torch.cuda.max_memory_allocated,
# resetado a cada troca de modelo em get_or_load/stage) quando torch com GPU
# visível está disponível, ou RSS de CPU como proxy caso contrário; helper
# interno usado por ModelLifecycleManager para preencher StageMetrics.
def _peak_memory_bytes() -> int:
    """Mantém a medida legada, acompanhada de seu tipo explícito em StageMetrics."""
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return _peak_cpu_rss_bytes()


# Identifica a unidade de observação para evitar interpretar RSS como VRAM.
def _memory_kind() -> str:
    """Retorna o tipo da medida legada de pico de memória."""
    return "cuda_allocated" if torch is not None and torch.cuda.is_available() else "cpu_rss"


# Mede somente a memória CUDA; ausência de GPU permanece explícita.
def _peak_vram_bytes() -> int | None:
    """Retorna bytes alocados em CUDA ou None quando essa medição não existe."""
    if torch is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


# Converte o high-water mark de RSS do processo para bytes na plataforma atual.
def _peak_cpu_rss_bytes() -> int:
    """Retorna o pico acumulado de RSS do processo, separado da VRAM."""
    multiplier = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier
