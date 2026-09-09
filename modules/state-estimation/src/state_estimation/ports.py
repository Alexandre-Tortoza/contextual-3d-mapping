"""Ports públicos para estimators substituíveis."""

from __future__ import annotations

from typing import Protocol

from .models import ImuObservation, LidarObservation, MotionCorrectedLidarFrame, StateEstimate


# Define a capacidade estável consumida pelo runtime sem acoplá-lo ao FAST-LIO.
# É satisfeita por adapters reais e fakes de teste.
class StateEstimator(Protocol):
    """Port de estimação de estado LiDAR-inercial."""

    # Processa um scan e sua janela IMU através da implementação selecionada.
    def process(
        self, lidar: LidarObservation, imu_samples: tuple[ImuObservation, ...]
    ) -> tuple[StateEstimate, MotionCorrectedLidarFrame]:
        """Processa um scan LiDAR e amostras IMU temporalmente compatíveis.

        Argumentos:
            lidar: scan LiDAR no frame declarado pela observação.
            imu_samples: janela IMU associada ao scan, em ordem temporal.
        Retorna:
            estimativa de estado e scan corrigido por movimento.
        """

    # Reinicia o ciclo de vida sem trocar a implementação concreta.
    def reset(self) -> None:
        """Redefine o estado de runtime do estimator."""
