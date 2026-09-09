"""Testes de projeção, sincronização, suporte válido e oclusão."""

from __future__ import annotations

import pytest
from geometric_map import GeometryPoint, GeometryReference
from sensor_association import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    RgbFrame,
    VisualRegionEvidence,
    associate_points,
)

from contextual_mapping_contracts import (
    FrameId,
    MapId,
    ObservationReference,
    Provenance,
    RigidTransform,
    SourceArtifactReference,
    Timestamp,
)


# Constrói referências no mesmo dataset e clock para isolar a geometria nos testes.
def _reference(observation_id: str, sensor: str, frame: str, timestamp_ns: int = 10) -> ObservationReference:
    """Cria uma referência canônica de sensor para a fixture."""
    return ObservationReference(
        observation_id,
        "dataset",
        "sequence",
        sensor,
        0,
        Timestamp(timestamp_ns, "clock"),
        FrameId(frame),
        "calibration-1" if sensor == "rgb" else None,
    )


# Constrói um ponto persistido preservando suas coordenadas no frame LiDAR.
def _point(geometry_id: str, coordinates: tuple[float, float, float]) -> GeometryPoint:
    """Cria um ponto geométrico com proveniência LiDAR válida."""
    source = _reference("lidar-1", "lidar", "lidar")
    return GeometryPoint(
        GeometryReference(MapId("map-1"), geometry_id),
        coordinates,
        coordinates,
        source,
        Provenance("fixture", (source,)),
    )


# Fornece calibração identidade entre frames distintos para projeção pinhole.
def _calibration() -> CameraLidarCalibration:
    """Cria uma calibração pinhole simples de teste."""
    return CameraLidarCalibration(
        "calibration-1",
        SourceArtifactReference("fixture://calibration", "application/json"),
        CameraModel.PINHOLE,
        2.0,
        2.0,
        2.0,
        2.0,
        RigidTransform(FrameId("lidar"), FrameId("camera"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


# Exercita associação positiva e z-buffer determinístico para dois pontos no mesmo pixel.
def test_associate_points_colors_visible_point_and_marks_occlusion() -> None:
    """Associa o ponto próximo e marca o ponto distante como ocluído."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        5,
        5,
        tuple((index, 0, 0) for index in range(25)),
        frozenset((x, y) for y in range(5) for x in range(5)),
    )
    region = VisualRegionEvidence("region-1", frozenset({(2, 2)}), "parede", "feature://1")

    result = associate_points(
        (_point("near", (0.0, 0.0, 2.0)), _point("far", (0.0, 0.0, 3.0))),
        rgb,
        _calibration(),
        (region,),
    )

    assert result[0].status is AssociationStatus.ASSOCIATED
    assert result[0].pixel == (2, 2)
    assert result[0].color_rgb == (12, 0, 0)
    assert result[0].region_id == "region-1"
    assert result[1].status is AssociationStatus.OCCLUDED


# Protege a tolerância temporal explícita da associação multimodal.
def test_associate_points_rejects_unsynchronized_rgb() -> None:
    """Rejeita RGB fora da janela temporal solicitada."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera", timestamp_ns=100),
        1,
        1,
        ((0, 0, 0),),
        frozenset({(0, 0)}),
    )

    with pytest.raises(ValueError, match="max_time_delta_ns"):
        associate_points((_point("point", (0.0, 0.0, 1.0)),), rgb, _calibration(), max_time_delta_ns=20)


# Evita associação ambígua quando duas regiões reivindicam o mesmo pixel.
def test_associate_points_rejects_overlapping_regions() -> None:
    """Rejeita masks de regiões que se sobrepõem."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        1,
        1,
        ((0, 0, 0),),
        frozenset({(0, 0)}),
    )
    regions = (
        VisualRegionEvidence("region-1", frozenset({(0, 0)})),
        VisualRegionEvidence("region-2", frozenset({(0, 0)})),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        associate_points((_point("point", (0.0, 0.0, 1.0)),), rgb, _calibration(), regions)


# Impede que a ambiguidade matemática do modelo MEI associe a imagem frontal
# a pontos situados atrás do plano óptico.
def test_associate_points_rejects_mei_ray_behind_front_camera() -> None:
    """Rejeita um raio MEI com z negativo para uma câmera frontal."""
    calibration = CameraLidarCalibration(
        "calibration-1",
        SourceArtifactReference("fixture://calibration", "application/json"),
        CameraModel.MEI,
        2.0,
        2.0,
        2.0,
        2.0,
        RigidTransform(
            FrameId("lidar"),
            FrameId("camera"),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        mirror_xi=1.5,
    )
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        5,
        5,
        tuple((index, 0, 0) for index in range(25)),
        frozenset((x, y) for y in range(5) for x in range(5)),
    )

    result = associate_points((_point("point", (0.0, 0.0, -1.0)),), rgb, calibration)

    assert result[0].status is AssociationStatus.BEHIND_CAMERA


# Preserva suporte explícito a rigs MEI que realmente observam o hemisfério
# traseiro, sem tornar esse comportamento inseguro o default das câmeras.
def test_associate_points_allows_rear_mei_ray_when_calibrated() -> None:
    """Projeta um raio traseiro somente quando a calibração habilita esse domínio."""
    calibration = CameraLidarCalibration(
        "calibration-1",
        SourceArtifactReference("fixture://calibration", "application/json"),
        CameraModel.MEI,
        2.0,
        2.0,
        2.0,
        2.0,
        RigidTransform(
            FrameId("lidar"),
            FrameId("camera"),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        mirror_xi=1.5,
        front_hemisphere_only=False,
    )
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        5,
        5,
        tuple((index, 0, 0) for index in range(25)),
        frozenset((x, y) for y in range(5) for x in range(5)),
    )

    result = associate_points((_point("point", (0.0, 0.0, -1.0)),), rgb, calibration)

    assert result[0].status is AssociationStatus.ASSOCIATED
    assert result[0].pixel == (2, 2)
