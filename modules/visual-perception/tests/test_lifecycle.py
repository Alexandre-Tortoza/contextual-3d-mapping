"""Sequential model lifecycle and memory diagnostics tests (#171). Fake adapters only."""

from __future__ import annotations

import pytest

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.domain.errors import BackendExecutionError


class _FakeModel:
    def infer(self) -> str:
        return "ok"


def test_stage_records_timing_and_memory_metrics() -> None:
    manager = ModelLifecycleManager()
    with manager.stage("region_discovery", _FakeModel) as model:
        model.infer()

    assert len(manager.metrics) == 1
    metrics = manager.metrics[0]
    assert metrics.stage_name == "region_discovery"
    assert metrics.load_time_s >= 0
    assert metrics.inference_time_s >= 0
    assert metrics.peak_memory_bytes > 0


def test_models_do_not_need_to_stay_resident_simultaneously() -> None:
    manager = ModelLifecycleManager()
    with manager.stage("stage_a", _FakeModel):
        pass
    with manager.stage("stage_b", _FakeModel):
        pass
    assert [m.stage_name for m in manager.metrics] == ["stage_a", "stage_b"]


def test_oom_during_load_surfaces_as_backend_execution_error() -> None:
    def failing_factory() -> _FakeModel:
        raise MemoryError

    manager = ModelLifecycleManager()
    with pytest.raises(BackendExecutionError), manager.stage("region_discovery", failing_factory):
        pass


def test_oom_during_inference_surfaces_as_backend_execution_error() -> None:
    manager = ModelLifecycleManager()
    with pytest.raises(BackendExecutionError), manager.stage("region_discovery", _FakeModel):
        raise MemoryError
