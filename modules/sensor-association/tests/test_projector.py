"""Testes de projeção, sincronização, suporte válido e oclusão."""

from __future__ import annotations

import pytest
from geometric_map import GeometryPoint, GeometryReference
from sensor_association import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    RgbFrame,
    VisualRegionEvidence,
    associate_map_points,
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


# Constrói um ponto do mapa persistente, sem coordenada de scan nem observação
# de origem única — que é exatamente a diferença em relação a GeometryPoint.
def _map_point(geometry_id: str, coordinates: tuple[float, float, float]) -> MapAnchoredPoint:
    """Cria um ponto de mapa para a associação ancorada."""
    return MapAnchoredPoint(GeometryReference(MapId("map-1"), geometry_id), coordinates)


# Fornece o transform mapa→câmera usado quando a pose já registrou o mapa.
def _map_to_camera() -> RigidTransform:
    """Cria um transform identidade entre o frame do mapa e o da câmera."""
    return RigidTransform(FrameId("map"), FrameId("camera"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


# Repete, na fronteira ancorada no mapa, as mesmas garantias de associação,
# região e oclusão exigidas da associação por scan.
def test_associate_map_points_colors_visible_point_and_marks_occlusion() -> None:
    """Associa o ponto próximo do mapa e marca o distante como ocluído."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        5,
        5,
        tuple((index, 0, 0) for index in range(25)),
        frozenset((x, y) for y in range(5) for x in range(5)),
    )
    region = VisualRegionEvidence("region-1", frozenset({(2, 2)}), "parede", "feature://1")

    result = associate_map_points(
        (_map_point("near", (0.0, 0.0, 2.0)), _map_point("far", (0.0, 0.0, 3.0))),
        rgb,
        _calibration(),
        _map_to_camera(),
        (region,),
        lidar_observation=_reference("lidar-1", "lidar", "lidar"),
    )

    assert result[0].status is AssociationStatus.ASSOCIATED
    assert result[0].pixel == (2, 2)
    assert result[0].color_rgb == (12, 0, 0)
    assert result[0].region_id == "region-1"
    assert result[0].label == "parede"
    assert result[1].status is AssociationStatus.OCCLUDED


# É a garantia de substituição entre as duas fronteiras: o mesmo ponto físico,
# alcançado por caminhos diferentes, precisa produzir o mesmo pixel e a mesma
# classificação. Sem isso, ancorar no mapa mudaria silenciosamente o resultado.
def test_associate_map_points_concorda_com_a_associacao_por_scan() -> None:
    """Compara as duas fronteiras sobre o mesmo ponto físico."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        5,
        5,
        tuple((index, 0, 0) for index in range(25)),
        frozenset((x, y) for y in range(5) for x in range(5)),
    )
    region = VisualRegionEvidence("region-1", frozenset({(2, 2)}), "parede", "feature://1")
    coordinates = (0.0, 0.0, 2.0)

    by_scan = associate_points((_point("p", coordinates),), rgb, _calibration(), (region,))
    by_map = associate_map_points(
        (_map_point("p", coordinates),),
        rgb,
        _calibration(),
        _map_to_camera(),
        (region,),
        lidar_observation=_reference("lidar-1", "lidar", "lidar"),
    )

    assert by_map[0].status is by_scan[0].status
    assert by_map[0].pixel == by_scan[0].pixel
    assert by_map[0].color_rgb == by_scan[0].color_rgb
    assert by_map[0].region_id == by_scan[0].region_id
    assert by_map[0].label == by_scan[0].label


# Os três motivos de rejeição precisam continuar distinguíveis, porque é o que
# o viewer usa para explicar por que um ponto do mapa ficou sem contexto.
def test_associate_map_points_preserva_os_motivos_de_rejeicao() -> None:
    """Classifica pontos atrás da câmera, fora da imagem e fora do suporte."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        5,
        5,
        tuple((index, 0, 0) for index in range(25)),
        frozenset({(2, 2)}),
    )

    result = associate_map_points(
        (
            _map_point("atras", (0.0, 0.0, -1.0)),
            _map_point("fora-da-imagem", (10.0, 10.0, 1.0)),
            _map_point("fora-do-suporte", (1.0, 0.0, 2.0)),
            _map_point("visivel", (0.0, 0.0, 2.0)),
        ),
        rgb,
        _calibration(),
        _map_to_camera(),
        lidar_observation=_reference("lidar-1", "lidar", "lidar"),
    )

    assert result[0].status is AssociationStatus.BEHIND_CAMERA
    assert result[1].status is AssociationStatus.OUTSIDE_IMAGE
    assert result[2].status is AssociationStatus.OUTSIDE_VALID_SUPPORT
    assert result[3].status is AssociationStatus.ASSOCIATED
    assert all(item.color_rgb is None for item in result[:3])


# O transform precisa terminar no frame da imagem; caso contrário a projeção
# seria feita com uma pose que não descreve aquela câmera.
def test_associate_map_points_exige_transform_para_o_frame_da_imagem() -> None:
    """Rejeita um transform que não termina no frame do RGB."""
    rgb = RgbFrame(_reference("rgb-1", "rgb", "camera"), 1, 1, ((0, 0, 0),), frozenset({(0, 0)}))
    wrong = RigidTransform(FrameId("map"), FrameId("outro"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="map_to_camera target frame"):
        associate_map_points(
            (_map_point("p", (0.0, 0.0, 1.0)),),
            rgb,
            _calibration(),
            wrong,
            lidar_observation=_reference("lidar-1", "lidar", "lidar"),
        )


# É a regressão do defeito observado ao ancorar a associação no mapa: com um
# mapa acumulado e subamostrado, o z-buffer por pixel exato deixa pontos de trás
# passarem pelos vazios entre as amostras da superfície da frente, e eles
# recebem o label do que está na imagem. Medido no corridor-02, isso rotulava
# pontos na altura da parede como "ceiling".
#
# A fixture reproduz a esparsidade: com fx=2 e cx=2, a parede próxima ocupa os
# pixels 2, 4 e 6, e o ponto distante cai no pixel 3 — um vazio que nenhuma
# amostra da parede ocupa.
def _cena_de_parede_esparsa() -> tuple[RgbFrame, tuple[MapAnchoredPoint, ...], MapAnchoredPoint]:
    """Monta uma parede próxima subamostrada e um ponto atrás dela."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"),
        9,
        9,
        tuple((index, 0, 0) for index in range(81)),
        frozenset((x, y) for y in range(9) for x in range(9)),
    )
    parede = tuple(
        _map_point(f"parede-{pixel}", (x, 0.0, 2.0))
        for pixel, x in ((2, 0.0), (4, 2.0), (6, 4.0))
    )
    atras = _map_point("atras", (3.0, 0.0, 6.0))
    return rgb, parede, atras


def test_associate_map_points_ocluye_ponto_atras_de_superficie_esparsa() -> None:
    """Rejeita o ponto distante que cai entre as amostras da parede."""
    rgb, parede, atras = _cena_de_parede_esparsa()

    resultado = associate_map_points(
        (*parede, atras),
        rgb,
        _calibration(),
        _map_to_camera(),
        lidar_observation=_reference("lidar-1", "lidar", "lidar"),
    )

    por_id = {item.geometry.geometry_id: item for item in resultado}
    assert por_id["atras"].status is AssociationStatus.OCCLUDED
    assert all(
        por_id[point.geometry.geometry_id].status is AssociationStatus.ASSOCIATED
        for point in parede
    )


# O z-buffer por pixel exato é o que deixava o ponto passar: ele ocupa um pixel
# livre e nada o contradiz. Este teste fixa esse comportamento como o que a
# célula desligada produz, tornando a diferença entre os dois modos explícita.
def test_associate_map_points_sem_teste_de_superficie_deixa_o_ponto_passar() -> None:
    """Confere que célula zero volta ao z-buffer por pixel exato."""
    rgb, parede, atras = _cena_de_parede_esparsa()

    resultado = associate_map_points(
        (*parede, atras),
        rgb,
        _calibration(),
        _map_to_camera(),
        lidar_observation=_reference("lidar-1", "lidar", "lidar"),
        occlusion_cell_px=0,
    )

    assert all(item.status is AssociationStatus.ASSOCIATED for item in resultado)
