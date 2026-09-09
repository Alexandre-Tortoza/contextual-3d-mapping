"""Testes do contract público e do bridge de processo FAST-LIO."""

from __future__ import annotations

import sys

import pytest
from contextual_mapping_contracts import FrameId, ObservationReference, Timestamp

from state_estimation import ImuObservation, LidarObservation
from state_estimation.infrastructure.fast_lio import (
    FastLioBridgeError,
    FastLioProcessAdapter,
    FastLioProcessConfig,
)


# Cria referências temporalmente compatíveis usadas pelos testes do adapter.
def _reference(observation_id: str, sensor_id: str, timestamp_ns: int) -> ObservationReference:
    """Cria uma referência sintética no clock ROS de teste."""
    return ObservationReference(
        observation_id,
        "dataset",
        "sequence",
        sensor_id,
        0,
        Timestamp(timestamp_ns, "ros"),
        FrameId(sensor_id),
    )


# Garante que o protocolo envia IMU completa e usa os pontos deskewed da
# resposta, em vez de republicar silenciosamente o scan bruto.
def test_process_uses_complete_bridge_response() -> None:
    """Converte pose e pontos deskewed retornados pelo processo fixture."""
    bridge = """
import json, sys
request = json.load(sys.stdin)
assert request["schema_version"] == 1
assert request["imu_samples"][0]["angular_velocity_rad_s"] == [0.1, 0.2, 0.3]
json.dump({"translation_m": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1], "points_m": [[9, 8, 7]]}, sys.stdout)
"""
    lidar = LidarObservation(_reference("lidar-1", "lidar", 20), ((1.0, 2.0, 3.0),))
    imu = ImuObservation(_reference("imu-1", "imu", 10), (0.1, 0.2, 0.3), (0.0, 0.0, 9.8))
    adapter = FastLioProcessAdapter(FastLioProcessConfig((sys.executable, "-c", bridge), FrameId("map")))

    estimate, corrected = adapter.process(lidar, (imu,))

    assert estimate.pose.transform.translation_m == (1.0, 2.0, 3.0)
    assert corrected.observation.points_m == ((9, 8, 7),)
    assert tuple(item.observation_id for item in corrected.provenance.observations) == (
        "lidar-1",
        "imu-1",
    )


# Confirma que uma resposta incompleta vira diagnóstico estável do adapter.
def test_process_wraps_invalid_bridge_payload() -> None:
    """Traduz payload sem pontos deskewed para FastLioBridgeError."""
    adapter = FastLioProcessAdapter(
        FastLioProcessConfig(
            (sys.executable, "-c", "print('{}')"),
            FrameId("map"),
        )
    )
    lidar = LidarObservation(_reference("lidar-1", "lidar", 20), ((1.0, 2.0, 3.0),))

    with pytest.raises(FastLioBridgeError, match="bridge failed"):
        adapter.process(lidar, ())


# Protege a semântica da janela IMU, que precisa chegar ordenada ao estimator.
def test_process_rejects_unordered_imu_samples() -> None:
    """Rejeita amostras IMU fora da ordem temporal."""
    lidar = LidarObservation(_reference("lidar-1", "lidar", 30), ((1.0, 2.0, 3.0),))
    later = ImuObservation(_reference("imu-2", "imu", 20), (0.0, 0.0, 0.0), (0.0, 0.0, 9.8))
    earlier = ImuObservation(_reference("imu-1", "imu", 10), (0.0, 0.0, 0.0), (0.0, 0.0, 9.8))
    adapter = FastLioProcessAdapter(
        FastLioProcessConfig((sys.executable, "-c", "print('{}')"), FrameId("map"))
    )

    with pytest.raises(ValueError, match="ordered"):
        adapter.process(lidar, (later, earlier))
