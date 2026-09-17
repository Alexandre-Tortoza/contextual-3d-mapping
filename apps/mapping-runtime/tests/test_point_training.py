"""Integração de targets visuais densos até a fronteira de treino 3D."""

from __future__ import annotations

import json

import numpy as np
import pytest
from geometric_map import GeometryPoint, GeometryReference
from mapping_runtime.corridor02_context import _load_dense_feature_map
from mapping_runtime.point_training import visual_teacher_sample
from sensor_association import (
    CameraLidarCalibration,
    CameraModel,
    DenseFeatureMap,
    RgbFrame,
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


# Cria referências compatíveis para que a fixture exercite associação real,
# em vez de construir PointVisualAssociation manualmente fora do produtor.
def _reference(observation_id: str, sensor: str, frame: str) -> ObservationReference:
    """Monta uma referência de observação válida para o cenário integrado."""
    return ObservationReference(
        observation_id, "fixture", "sequence", sensor, 0, Timestamp(10, "clock"), FrameId(frame),
        "calibration-1" if sensor == "rgb" else None,
    )


# Mantém a calibração pinhole simples e explícita usada por todos os pontos
# da fixture, de forma que identidade e provenance sejam verificáveis.
def _calibration() -> CameraLidarCalibration:
    """Cria uma calibração identidade entre LiDAR e câmera."""
    return CameraLidarCalibration(
        "calibration-1", SourceArtifactReference("fixture://calibration", "application/json"),
        CameraModel.PINHOLE, 2.0, 2.0, 2.0, 2.0,
        RigidTransform(FrameId("lidar"), FrameId("camera"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


# Produz pontos no mesmo pixel para que a associação real marque o segundo
# como ocluído e prove que ele não vira target de professor artificial.
def _point(geometry_id: str, z: float) -> GeometryPoint:
    """Cria um ponto LiDAR persistido para a fixture de oclusão."""
    source = _reference("lidar-1", "lidar", "lidar")
    return GeometryPoint(
        GeometryReference(MapId("map-1"), geometry_id), (0.0, 0.0, z), (0.0, 0.0, z), source,
        Provenance("fixture", (source,)),
    )


def test_visual_teacher_sample_preserves_identity_and_rejects_occluded_points() -> None:
    """Converte somente o ponto visível em target alinhado e reproduz o resultado."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"), 5, 5, ((0, 0, 0),) * 25,
        frozenset((x, y) for y in range(5) for x in range(5)),
    )
    values = np.zeros((5, 5, 2), dtype=np.float32)
    values[2, 2] = (0.25, 0.75)
    points = (_point("near", 2.0), _point("far", 3.0))
    associations = associate_points(
        points, rgb, _calibration(),
        dense_feature_map=DenseFeatureMap(values, "dino/test", "feature-extraction:test", "artifact://dense"),
    )

    sample, provenance = visual_teacher_sample(
        np.asarray([point.coordinates_m for point in points]), FrameId("lidar"), associations
    )
    replay, replay_provenance = visual_teacher_sample(
        np.asarray([point.coordinates_m for point in points]), FrameId("lidar"), associations
    )

    assert sample.correspondence_indices.tolist() == [0]
    assert sample.teacher_features.tolist() == [[0.25, 0.75]]
    assert provenance.embedding_space == "dino/test"
    assert provenance.artifact_references == ("artifact://dense",)
    assert provenance.observation_ids == ("rgb-1",)
    assert provenance.calibration_ids == ("calibration-1",)
    assert replay.teacher_features.tolist() == sample.teacher_features.tolist()
    assert replay_provenance == provenance


def test_visual_teacher_sample_rejects_missing_dense_targets() -> None:
    """Recusa uma entrada sem suporte visual em vez de preencher vetores nulos."""
    rgb = RgbFrame(
        _reference("rgb-1", "rgb", "camera"), 5, 5, ((0, 0, 0),) * 25,
        frozenset((x, y) for y in range(5) for x in range(5)),
    )
    point = _point("point", 2.0)
    associations = associate_points(
        (point,), rgb, _calibration(),
        dense_feature_map=DenseFeatureMap(
            np.ones((5, 5, 2), dtype=np.float32), "dino/test", "feature-extraction:test", "artifact://dense",
            np.zeros((5, 5), dtype=np.bool_),
        ),
    )

    with pytest.raises(ValueError, match="at least one valid"):
        visual_teacher_sample(np.asarray([point.coordinates_m]), FrameId("lidar"), associations)


def test_runtime_reopens_only_a_pixel_aligned_dense_artifact(tmp_path) -> None:
    """Carrega valores, suporte e provenance sem reinferir o mapa visual."""
    frame_dir = tmp_path / "frames" / "frame-1"
    frame_dir.mkdir(parents=True)
    observation = frame_dir / "observation.json"
    observation.write_text("{}", encoding="utf-8")
    values = np.zeros((2, 3, 2), dtype=np.float32)
    values[1, 2] = (0.2, 0.8)
    support = np.ones((2, 3), dtype=np.bool_)
    np.savez_compressed(frame_dir / "dense-features.npz", values=values, valid_support=support)
    (frame_dir / "dense-features.json").write_text(
        json.dumps(
            {
                "representation": "pixel_aligned", "grid_width": 3, "grid_height": 2,
                "model_id": "dino-test", "checkpoint": "checkpoint-1", "generation": "resampled",
            }
        ),
        encoding="utf-8",
    )

    feature_map = _load_dense_feature_map(observation, width=3, height=2)

    assert feature_map is not None
    assert np.allclose(feature_map.values[1, 2], (0.2, 0.8))
    assert feature_map.embedding_space == "dino-test:checkpoint-1:pixel_aligned"
    assert feature_map.artifact_reference.endswith("dense-features.npz")
