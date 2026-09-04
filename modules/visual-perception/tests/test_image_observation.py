"""Testes do contract de entrada de observação de imagem canônica (#153)."""

from __future__ import annotations

import pytest
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from fixtures import image_observation
from visual_perception.domain.image_observation import ImageObservation


# Constrói um par (image, source) mínimo e válido para os testes que verificam
# rejeição de entradas inválidas em ImageObservation.
def _reference() -> tuple[SourceArtifactReference, ObservationReference]:
    return (
        SourceArtifactReference(uri="mem://1", media_type="image/png"),
        ObservationReference(
            observation_id="obs-1",
            dataset_id="ds",
            sequence_id="seq",
            sensor_id="camera_1",
            sequence_index=0,
            timestamp=Timestamp(nanoseconds=0, clock_id="rosbag"),
            frame_id=FrameId("camera_1_optical_frame"),
        ),
    )


# Confirma que a fixture padrão de observação constrói um objeto válido com o id esperado.
def test_minimal_valid_observation() -> None:
    observation = image_observation()
    assert observation.observation_id == "frame-0001"


# Protege o invariante de que largura e altura devem ser positivas.
def test_rejects_non_positive_dimensions() -> None:
    image, source = _reference()
    with pytest.raises(ValueError):
        ImageObservation(0, 0, "rgb8", image, source)


# Protege o invariante de que só encodings suportados (ex: rgb8) são aceitos — um
# encoding desconhecido deve falhar cedo, não silenciosamente mais adiante no pipeline.
def test_rejects_unsupported_encoding() -> None:
    image, source = _reference()
    with pytest.raises(ValueError):
        ImageObservation(10, 10, "yuv420", image, source)


# Garante que a ObservationReference de origem (sensor, dataset, etc.) é preservada sem
# alteração, já que é usada como proveniência downstream.
def test_preserves_source_observation_reference() -> None:
    observation = image_observation()
    assert observation.source.sensor_id == "camera_1"
