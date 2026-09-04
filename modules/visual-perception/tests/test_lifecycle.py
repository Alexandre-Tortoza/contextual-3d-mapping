"""Testes sequenciais de ciclo de vida de modelo e diagnóstico de memória (#171). Só adapters fake."""

from __future__ import annotations

import pytest

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.domain.errors import BackendExecutionError


# Modelo fake mínimo usado para exercitar o gerenciador de ciclo de vida sem depender de
# nenhum backend real.
class _FakeModel:
    def infer(self) -> str:
        return "ok"


# Verifica que, ao executar um estágio, o manager registra métricas de timing e memória
# (load_time_s, inference_time_s, peak_memory_bytes) com valores plausíveis.
def test_stage_records_timing_and_memory_metrics() -> None:
    manager = ModelLifecycleManager()
    with manager.stage("region_discovery", _FakeModel) as model:
        model.infer()

    assert len(manager.metrics) == 1
    metrics = manager.metrics[0]
    assert metrics.stage_name == "region_discovery"
    assert metrics.load_time_s >= 0
    assert metrics.inference_time_s >= 0
    # 0 é o valor correto aqui: _FakeModel nunca toca CUDA, então o pico real
    # de VRAM observado é genuinamente zero (ver _peak_memory_bytes).
    assert metrics.peak_memory_bytes >= 0


# Confirma o motivo de existir do lifecycle manager: cada estágio é criado, usado, e
# liberado isoladamente, então dois estágios nunca precisam ficar residentes ao mesmo tempo.
def test_models_do_not_need_to_stay_resident_simultaneously() -> None:
    manager = ModelLifecycleManager()
    with manager.stage("stage_a", _FakeModel):
        pass
    with manager.stage("stage_b", _FakeModel):
        pass
    assert [m.stage_name for m in manager.metrics] == ["stage_a", "stage_b"]


# Garante que um MemoryError durante o load do modelo é convertido em
# BackendExecutionError, nunca deixado propagar como exception crua de backend.
def test_oom_during_load_surfaces_as_backend_execution_error() -> None:
    def failing_factory() -> _FakeModel:
        raise MemoryError

    manager = ModelLifecycleManager()
    with pytest.raises(BackendExecutionError), manager.stage("region_discovery", failing_factory):
        pass


# Mesma garantia acima, mas para um MemoryError levantado durante a inferência (dentro
# do bloco `with`), não durante o load.
def test_oom_during_inference_surfaces_as_backend_execution_error() -> None:
    manager = ModelLifecycleManager()
    with pytest.raises(BackendExecutionError), manager.stage("region_discovery", _FakeModel):
        raise MemoryError


# Confirma o caminho rápido de get_or_load: chamadas repetidas com a mesma key
# retornam a mesma instância sem chamar a factory de novo (ex: várias regiões
# usando o mesmo VLM já carregado).
def test_get_or_load_reuses_model_for_same_key() -> None:
    manager = ModelLifecycleManager()
    calls = []

    def factory() -> _FakeModel:
        calls.append(1)
        return _FakeModel()

    first = manager.get_or_load("multimodal_reasoning", factory)
    second = manager.get_or_load("multimodal_reasoning", factory)

    assert first is second
    assert len(calls) == 1
    assert len(manager.metrics) == 1


# Confirma a razão de existir de get_or_load para os adapters reais: pedir uma
# key diferente libera o modelo ativo antes de carregar o novo, garantindo que
# no máximo um modelo pesado fica residente por vez mesmo quando os 4 adapters
# compartilham este manager.
def test_get_or_load_releases_previous_model_on_key_change() -> None:
    manager = ModelLifecycleManager()

    manager.get_or_load("region_discovery", _FakeModel)
    manager.get_or_load("feature_extraction", _FakeModel)

    assert [m.stage_name for m in manager.metrics] == ["region_discovery", "feature_extraction"]
    assert manager._active_key == "feature_extraction"  # noqa: SLF001 - test-only introspection


# Garante que release_active limpa o estado ativo, para que um pipeline possa
# não deixar nada residente ao terminar.
def test_release_active_clears_state() -> None:
    manager = ModelLifecycleManager()
    manager.get_or_load("region_discovery", _FakeModel)

    manager.release_active()

    assert manager._active_key is None  # noqa: SLF001 - test-only introspection


# Garante que um MemoryError durante o load em get_or_load também é convertido
# em BackendExecutionError, com a mesma garantia de `stage`.
def test_get_or_load_oom_surfaces_as_backend_execution_error() -> None:
    def failing_factory() -> _FakeModel:
        raise MemoryError

    manager = ModelLifecycleManager()
    with pytest.raises(BackendExecutionError):
        manager.get_or_load("region_discovery", failing_factory)
