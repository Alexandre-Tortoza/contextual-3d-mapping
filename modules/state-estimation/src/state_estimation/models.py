"""Contracts públicos de entrada e saída de state-estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from contextual_mapping_contracts import FrameId, ObservationReference, Pose, Provenance


# Representa uma observação LiDAR canônica antes de alcançar um backend.
# Existe para manter layouts de ROS e datasets fora da API pública do módulo.
@dataclass(frozen=True)
class LidarObservation:
    """Nuvem LiDAR timestampada no frame declarado.

    Argumentos:
        reference: identidade e timestamp da observação de origem.
        points_m: coordenadas x, y, z em metros.
    """

    reference: ObservationReference
    points_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not self.points_m:
            raise ValueError("points_m must not be empty.")
        for point in self.points_m:
            if len(point) != 3 or not all(isfinite(float(value)) for value in point):
                raise ValueError("points_m must contain finite xyz triples.")


# Representa uma leitura IMU no contract independente de middleware. É a
# entrada mínima necessária para bridges de FAST-LIO e estimators substituíveis.
@dataclass(frozen=True)
class ImuObservation:
    """Leitura IMU de velocidade angular e aceleração linear.

    Argumentos:
        reference: identidade e timestamp da observação de origem.
        angular_velocity_rad_s: vetor angular em radianos por segundo.
        linear_acceleration_m_s2: aceleração linear em metros por segundo ao quadrado.
    """

    reference: ObservationReference
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]

    def __post_init__(self) -> None:
        for field in ("angular_velocity_rad_s", "linear_acceleration_m_s2"):
            value = getattr(self, field)
            if len(value) != 3 or not all(isfinite(float(item)) for item in value):
                raise ValueError(f"{field} must contain finite xyz triples.")


# Publica a pose produzida pelo estimator sem expor mensagens de odometria
# externas. Geometric-map e sensor-association usam esta saída diretamente.
@dataclass(frozen=True)
class StateEstimate:
    """Estimativa válida de pose e proveniência do estimator.

    Argumentos:
        pose: pose do frame LiDAR para o frame de mapa.
        reference: observação que ancora temporalmente a estimativa.
        provenance: produtor e observações contribuintes.
    """

    pose: Pose
    reference: ObservationReference
    provenance: Provenance


# Transporta a nuvem deskewed publicada pelo estimator para consumidores de
# geometria. Não cria mapa persistente nem altera a identidade da fonte.
@dataclass(frozen=True)
class MotionCorrectedLidarFrame:
    """Nuvem LiDAR corrigida por movimento com pose de registro.

    Argumentos:
        observation: observação LiDAR corrigida.
        pose: pose usada para registrar a nuvem.
        output_frame: frame no qual os pontos são expressos.
        provenance: proveniência completa do resultado.
    """

    observation: LidarObservation
    pose: Pose
    output_frame: FrameId
    provenance: Provenance
