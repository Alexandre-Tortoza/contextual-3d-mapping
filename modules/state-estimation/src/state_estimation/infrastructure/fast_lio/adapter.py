"""Adapter de processo externo para FAST-LIO."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from math import isfinite

from contextual_mapping_contracts import (
    FrameId,
    ObservationReference,
    Pose,
    Provenance,
    RigidTransform,
)

from ...models import ImuObservation, LidarObservation, MotionCorrectedLidarFrame, StateEstimate


# Declara o comando externo e frames esperados sem transportar detalhes de
# ROS para o domain. Compose fornece as implementações ROS 1 e ROS 2 deste
# protocolo de processo.
@dataclass(frozen=True)
class FastLioProcessConfig:
    """Configuração do bridge JSON-lines para processo FAST-LIO externo."""

    command: tuple[str, ...]
    map_frame: FrameId
    timeout_seconds: float = 30.0

    # Rejeita configurações que nunca poderiam iniciar ou limitar o bridge.
    def __post_init__(self) -> None:
        """Valida comando e timeout do processo externo."""
        if not self.command:
            raise ValueError("command must not be empty.")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


# Traduz falhas de processo, protocolo e payload em um erro estável do adapter.
# Existe para que o runtime não precise interpretar exceptions de subprocess ou JSON.
class FastLioBridgeError(RuntimeError):
    """Falha acionável na comunicação com o bridge FAST-LIO."""


# Serializa a identidade completa de uma observação para preservar frames,
# clocks e proveniência através da fronteira de processo.
def _reference_record(reference: ObservationReference) -> dict[str, object]:
    """Converte uma referência de observação para o protocolo JSON.

    Argumentos:
        reference: referência pública compartilhada a serializar.
    Retorna:
        registro JSON compatível com o bridge externo.
    """
    return {
        "observation_id": reference.observation_id,
        "dataset_id": reference.dataset_id,
        "sequence_id": reference.sequence_id,
        "sensor_id": reference.sensor_id,
        "sequence_index": reference.sequence_index,
        "timestamp_ns": reference.timestamp.nanoseconds,
        "clock_id": reference.timestamp.clock_id,
        "frame_id": str(reference.frame_id),
        "calibration_id": reference.calibration_id,
    }


# Encapsula o bridge de runtime externo. Existe para tornar FAST-LIO trocável
# e testável com um processo fixture, preservando contracts públicos locais.
class FastLioProcessAdapter:
    """Adapter StateEstimator que delega a um bridge FAST-LIO JSON-lines."""

    # Mantém somente configuração imutável; o estado do estimator pertence ao
    # processo externo selecionado pela aplicação.
    def __init__(self, config: FastLioProcessConfig) -> None:
        """Inicializa o adapter com o comando do bridge externo.

        Argumentos:
            config: comando, frame de mapa e timeout do bridge.
        """
        self._config = config

    # Envia o scan e as amostras IMU completas e converte a resposta deskewed.
    def process(
        self, lidar: LidarObservation, imu_samples: tuple[ImuObservation, ...]
    ) -> tuple[StateEstimate, MotionCorrectedLidarFrame]:
        """Executa o bridge e converte sua saída no contract público.

        Argumentos:
            lidar: scan LiDAR canônico.
            imu_samples: janela IMU ordenada associada ao scan.
        Retorna:
            pose estimada e nuvem deskewed retornadas pelo bridge.
        Levanta:
            FastLioBridgeError: quando o processo ou o protocolo falha.
        """
        if any(
            sample.reference.timestamp.clock_id != lidar.reference.timestamp.clock_id
            for sample in imu_samples
        ):
            raise ValueError("LiDAR and IMU observations must use the same clock_id.")
        if tuple(sample.reference.timestamp.nanoseconds for sample in imu_samples) != tuple(
            sorted(sample.reference.timestamp.nanoseconds for sample in imu_samples)
        ):
            raise ValueError("imu_samples must be ordered by timestamp.")
        request = {
            "schema_version": 1,
            "lidar": {"reference": _reference_record(lidar.reference), "points_m": lidar.points_m},
            "imu_samples": [
                {
                    "reference": _reference_record(sample.reference),
                    "angular_velocity_rad_s": sample.angular_velocity_rad_s,
                    "linear_acceleration_m_s2": sample.linear_acceleration_m_s2,
                }
                for sample in imu_samples
            ],
        }
        try:
            completed = subprocess.run(
                self._config.command,
                input=json.dumps(request) + "\n",
                capture_output=True,
                check=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )
            response = json.loads(completed.stdout)
            transform = RigidTransform(
                lidar.reference.frame_id,
                self._config.map_frame,
                tuple(response["translation_m"]),
                tuple(response["rotation_xyzw"]),
            )
            corrected = LidarObservation(
                lidar.reference, tuple(tuple(point) for point in response["points_m"])
            )
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise FastLioBridgeError(f"FAST-LIO bridge failed: {error}") from error
        contributors = {lidar.reference.observation_id: lidar.reference}
        contributors.update({sample.reference.observation_id: sample.reference for sample in imu_samples})
        provenance = Provenance("fast-lio", tuple(contributors.values()))
        pose = Pose(transform, lidar.reference.timestamp.nanoseconds)
        estimate = StateEstimate(pose, lidar.reference, provenance)
        return estimate, MotionCorrectedLidarFrame(corrected, pose, lidar.reference.frame_id, provenance)

    # O bridge atual é stateless por invocação; satisfaz o port explicitamente.
    def reset(self) -> None:
        """Não mantém estado local; o bridge recebe uma execução por scan."""
        return
