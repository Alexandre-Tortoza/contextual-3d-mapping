"""Contracts públicos de entrada e saída de state-estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from contextual_mapping_contracts import FrameId, ObservationReference, Pose, Provenance


# Valida vetores físicos antes que alcancem o estimator. Existe para manter a
# mesma regra de dimensão e finitude nas observações LiDAR e IMU.
def _finite_vector3(value: tuple[float, float, float], field: str) -> None:
    """Valida que um vetor possui três componentes finitos.

    Argumentos:
        value: vetor que cruza a fronteira pública.
        field: nome do campo usado no diagnóstico.
    """
    if len(value) != 3 or not all(isfinite(float(item)) for item in value):
        raise ValueError(f"{field} must contain finite xyz triples.")


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

    # Impede scans vazios ou coordenadas não finitas de chegarem ao backend.
    def __post_init__(self) -> None:
        """Valida a presença e as coordenadas dos pontos LiDAR."""
        if not self.points_m:
            raise ValueError("points_m must not be empty.")
        for point in self.points_m:
            _finite_vector3(point, "points_m")


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

    # Valida as duas grandezas vetoriais na entrada pública do estimator.
    def __post_init__(self) -> None:
        """Valida velocidade angular e aceleração linear."""
        for field in ("angular_velocity_rad_s", "linear_acceleration_m_s2"):
            _finite_vector3(getattr(self, field), field)


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

    # Garante que pose e observação descrevam o mesmo instante do resultado.
    def __post_init__(self) -> None:
        """Valida a âncora temporal da estimativa."""
        if self.pose.timestamp_ns != self.reference.timestamp.nanoseconds:
            raise ValueError("pose and reference timestamps must match.")


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

    # Protege consumidores geométricos contra divergência silenciosa entre o
    # frame dos pontos, a origem da pose e o timestamp do scan.
    def __post_init__(self) -> None:
        """Valida frames, timestamp e proveniência da nuvem corrigida."""
        if self.observation.reference.frame_id != self.output_frame:
            raise ValueError("observation frame must match output_frame.")
        if self.pose.transform.source_frame != self.output_frame:
            raise ValueError("pose source frame must match output_frame.")
        if self.pose.timestamp_ns != self.observation.reference.timestamp.nanoseconds:
            raise ValueError("pose and observation timestamps must match.")
        source_ids = {item.observation_id for item in self.provenance.observations}
        if self.observation.reference.observation_id not in source_ids:
            raise ValueError("provenance must include the LiDAR observation.")
