"""Adapter de processo externo para FAST-LIO."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Sequence

from contextual_mapping_contracts import FrameId, Pose, Provenance, RigidTransform

from ...models import ImuObservation, LidarObservation, MotionCorrectedLidarFrame, StateEstimate


# Declara o comando externo e frames esperados sem transportar detalhes de
# ROS para o domain. Compose fornece as implementações ROS 1 e ROS 2 deste
# protocolo de processo.
@dataclass(frozen=True)
class FastLioProcessConfig:
    """Configuração do bridge JSON-lines para processo FAST-LIO externo."""

    command: tuple[str, ...]
    map_frame: FrameId

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must not be empty.")


# Encapsula o bridge de runtime externo. Existe para tornar FAST-LIO trocável
# e testável com um processo fixture, preservando contracts públicos locais.
class FastLioProcessAdapter:
    """Adapter StateEstimator que delega a um bridge FAST-LIO JSON-lines."""

    def __init__(self, config: FastLioProcessConfig) -> None:
        self._config = config

    def process(
        self, lidar: LidarObservation, imu_samples: tuple[ImuObservation, ...]
    ) -> tuple[StateEstimate, MotionCorrectedLidarFrame]:
        """Executa o bridge e converte sua saída no contract público."""
        request = {
            "lidar": {"id": lidar.reference.observation_id, "points_m": lidar.points_m},
            "imu_count": len(imu_samples),
        }
        completed = subprocess.run(
            self._config.command,
            input=json.dumps(request),
            capture_output=True,
            check=True,
            text=True,
        )
        response = json.loads(completed.stdout)
        transform = RigidTransform(
            lidar.reference.frame_id,
            self._config.map_frame,
            tuple(response["translation_m"]),
            tuple(response["rotation_xyzw"]),
        )
        provenance = Provenance("fast-lio", (lidar.reference,))
        pose = Pose(transform, lidar.reference.timestamp.nanoseconds)
        estimate = StateEstimate(pose, lidar.reference, provenance)
        return estimate, MotionCorrectedLidarFrame(lidar, pose, lidar.reference.frame_id, provenance)

    def reset(self) -> None:
        """Não mantém estado local; o bridge recebe uma execução por scan."""
        return None
