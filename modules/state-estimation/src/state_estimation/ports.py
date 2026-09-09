"""Ports públicos para estimators substituíveis."""

from __future__ import annotations

from typing import Protocol

from .models import ImuObservation, LidarObservation, MotionCorrectedLidarFrame, StateEstimate


# Define a capacidade estável consumida pelo runtime sem acoplá-lo ao FAST-LIO.
# É satisfeita por adapters reais e fakes de teste.
class StateEstimator(Protocol):
    """Port de estimação de estado LiDAR-inercial."""

    def process(
        self, lidar: LidarObservation, imu_samples: tuple[ImuObservation, ...]
    ) -> tuple[StateEstimate, MotionCorrectedLidarFrame]:
        """Processa um scan LiDAR e amostras IMU temporalmente compatíveis."""

    def reset(self) -> None:
        """Redefine o estado de runtime do estimator."""
