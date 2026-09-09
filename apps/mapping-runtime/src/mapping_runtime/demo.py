"""Fixture executável e determinística do slice RGB–LiDAR M1."""

from __future__ import annotations

from pathlib import Path

from sensor_association import (
    CameraLidarCalibration,
    CameraModel,
    RgbFrame,
    VisualRegionEvidence,
)
from state_estimation import LidarObservation, MotionCorrectedLidarFrame

from contextual_mapping_contracts import (
    FrameId,
    MapId,
    ObservationReference,
    Pose,
    Provenance,
    RigidTransform,
    SourceArtifactReference,
    Timestamp,
)

from .pipeline import SliceBuildRequest, build_associated_slice


# Constrói uma referência completa sem duplicar a configuração comum dos dois
# sensores da fixture. Existe somente para manter o demo legível e auditável.
def _reference(
    observation_id: str,
    sensor_id: str,
    frame_id: FrameId,
    *,
    calibration_id: str | None = None,
) -> ObservationReference:
    """Cria uma referência sintética no clock compartilhado do demo.

    Argumentos:
        observation_id: identidade única da leitura.
        sensor_id: identidade do sensor produtor.
        frame_id: frame físico da observação.
        calibration_id: calibração opcional associada ao sensor.
    Retorna:
        referência canônica pronta para os contracts públicos.
    """
    return ObservationReference(
        observation_id=observation_id,
        dataset_id="m1-demo",
        sequence_id="synthetic-000",
        sensor_id=sensor_id,
        sequence_index=0,
        timestamp=Timestamp(1_000_000_000, "synthetic"),
        frame_id=frame_id,
        calibration_id=calibration_id,
    )


# Executa um slice pequeno sem ROS, GPU ou dados externos. Existe como smoke
# test operacional da composição e gera um artifact que o viewer abre de fato.
def export_demo_slice(destination: Path) -> Path:
    """Gera um artifact RGB–LiDAR sintético para validar o M1.

    Argumentos:
        destination: arquivo JSON que será criado.
    Retorna:
        caminho do artifact criado.
    """
    lidar_frame_id = FrameId("lidar")
    camera_frame_id = FrameId("camera_rgb")
    map_frame_id = FrameId("map")
    lidar_reference = _reference("lidar-000", "lidar", lidar_frame_id)
    rgb_reference = _reference(
        "rgb-000",
        "camera_rgb",
        camera_frame_id,
        calibration_id="demo-camera-lidar-v1",
    )
    lidar = LidarObservation(
        lidar_reference,
        (
            (-1.0, 0.0, 2.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 3.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
        ),
    )
    pose = Pose(
        RigidTransform(lidar_frame_id, map_frame_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        lidar_reference.timestamp.nanoseconds,
    )
    provenance = Provenance("mapping-runtime-demo", (lidar_reference,))
    corrected = MotionCorrectedLidarFrame(lidar, pose, lidar_frame_id, provenance)
    pixels = tuple((40 + x * 35, 40 + y * 35, 180) for y in range(5) for x in range(5))
    rgb = RgbFrame(
        rgb_reference,
        width=5,
        height=5,
        pixels=pixels,
        valid_pixels=frozenset((x, y) for y in range(5) for x in range(5)),
    )
    calibration = CameraLidarCalibration(
        calibration_id="demo-camera-lidar-v1",
        artifact=SourceArtifactReference(
            "fixture://m1-demo/calibration",
            "application/json",
            "sha256:synthetic",
        ),
        model=CameraModel.PINHOLE,
        fx=2.0,
        fy=2.0,
        cx=2.0,
        cy=2.0,
        lidar_to_camera=RigidTransform(
            lidar_frame_id,
            camera_frame_id,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    region = VisualRegionEvidence(
        "demo-region-center",
        frozenset({(2, 2), (2, 3)}),
        label="estrutura central",
        feature_reference="fixture://m1-demo/features/center",
    )
    request = SliceBuildRequest(
        MapId("m1-demo"),
        map_frame_id,
        corrected,
        rgb,
        calibration,
        (region,),
    )
    return build_associated_slice(request, destination)
