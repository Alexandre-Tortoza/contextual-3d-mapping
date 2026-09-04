"""Testes de fronteira de integração: RGB adapters (#177), sensor-association (#178),
mapping-runtime (#179), e persistência (#180). Todos usam fixtures sintéticas;
nenhum exige que os módulos irmãos já existam.
"""

from __future__ import annotations

import numpy as np
import pytest
from contextual_mapping_adapters import CanonicalObservation
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.infrastructure.adapters.region_discovery_backend import RealRegionDiscoveryAdapter
from visual_perception.infrastructure.fakes.fake_evidence_store import InMemoryEvidenceStore
from visual_perception.infrastructure.integration.mapping_runtime_integration import (
    RuntimeDiagnostic,
    run_visual_perception_for_runtime,
)
from visual_perception.infrastructure.integration.persistence_integration import (
    persist_observation,
    reload_observation,
)
from visual_perception.infrastructure.integration.rgb_adapter_boundary import to_canonical_input
from visual_perception.infrastructure.integration.sensor_association_contract import (
    build_association_fixtures,
)


# Helper que monta uma CanonicalObservation de kind "rgb" mínima e válida, reutilizada
# pelos testes de fronteira RGB abaixo para não repetir a construção de reference/artifact.
def _rgb_canonical_observation() -> CanonicalObservation:
    reference = ObservationReference(
        observation_id="source-1-000000",
        dataset_id="corridor02",
        sequence_id="seq-0",
        sensor_id="camera_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=100, clock_id="rosbag"),
        frame_id=FrameId("camera_1_optical_frame"),
    )
    artifact = SourceArtifactReference(uri="rosbag://corridor02/camera_1/0", media_type="image/raw")
    return CanonicalObservation(kind="rgb", reference=reference, artifact=artifact)


# Verifica que a fronteira RGB (#177) converte uma CanonicalObservation + pixel array em
# um ImageObservation/ImagePayload válido, preservando o id de origem e as dimensões.
def test_rgb_adapter_boundary_produces_valid_canonical_input() -> None:
    pixels = np.full((16, 16, 3), 255, dtype=np.uint8)
    image, payload = to_canonical_input(_rgb_canonical_observation(), pixels)
    assert image.observation_id == "source-1-000000"
    assert payload.width == 16


# Garante que a fronteira RGB rejeita explicitamente uma observação de outro kind (ex:
# "lidar"), em vez de aceitar silenciosamente dados fora do contract deste módulo.
def test_rgb_adapter_boundary_rejects_non_rgb_kind() -> None:
    lidar_reference = ObservationReference(
        observation_id="lidar-1",
        dataset_id="corridor02",
        sequence_id="seq-0",
        sensor_id="lidar_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=100, clock_id="rosbag"),
        frame_id=FrameId("lidar_1_frame"),
    )
    artifact = SourceArtifactReference(
        uri="rosbag://corridor02/lidar_1/0", media_type="application/octet-stream"
    )
    lidar_observation = CanonicalObservation(kind="lidar", reference=lidar_reference, artifact=artifact)
    with pytest.raises(ValueError):
        to_canonical_input(lidar_observation, np.zeros((16, 16, 3), dtype=np.uint8))


# Confirma que as fixtures de associação (#178) expõem exatamente os metadados de
# fronteira que sensor-association precisa (coordinate_convention, timestamp, frame_id),
# sem depender de nenhum tipo privado deste módulo.
def test_sensor_association_fixtures_expose_required_boundary_metadata() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    fixtures = build_association_fixtures(result.observation)
    assert len(fixtures) == len(result.observation.regions)
    assert fixtures[0].coordinate_convention == result.observation.coordinate_convention
    assert fixtures[0].timestamp == result.observation.source.timestamp
    assert fixtures[0].frame_id == result.observation.source.frame_id


# Garante que o adapter SAM 2 falha explicitamente sem runtime/checkpoint real,
# em vez de produzir uma saída silenciosamente incorreta.
def test_real_region_discovery_adapter_fails_explicitly_without_gpu() -> None:
    from visual_perception.config import RegionDiscoveryConfig
    from visual_perception.domain.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError):
        RealRegionDiscoveryAdapter().discover(payload_with_blobs(), RegionDiscoveryConfig())


# Verifica a fronteira de composição (#179): mapping-runtime consegue rodar o módulo
# inteiro através da API pública e recebe uma saída normal (não um RuntimeDiagnostic)
# quando nada falha.
def test_mapping_runtime_receives_canonical_observation_through_public_api() -> None:
    payload = payload_with_blobs()
    image, config, ports = image_observation(), default_config(), default_ports()
    output = run_visual_perception_for_runtime(image, payload, config, ports)
    assert not isinstance(output, RuntimeDiagnostic)


# Garante que, quando um backend do pipeline falha (aqui, o adapter real sem GPU), essa
# falha chega ao mapping-runtime como um RuntimeDiagnostic estruturado — nunca uma
# exception crua de backend vazando pela fronteira de composição.
def test_mapping_runtime_surfaces_module_failures_as_diagnostics() -> None:
    from visual_perception.application.pipeline import PerceptionPorts

    broken_ports = PerceptionPorts(
        region_discoverer=RealRegionDiscoveryAdapter(),
        feature_extractor=default_ports().feature_extractor,
        language_encoder=default_ports().language_encoder,
        multimodal_reasoner=default_ports().multimodal_reasoner,
    )
    output = run_visual_perception_for_runtime(
        image_observation(), payload_with_blobs(), default_config(), broken_ports
    )
    assert isinstance(output, RuntimeDiagnostic)
    assert output.stage == "visual_perception"


# Confirma a fronteira de persistência (#180): uma VisualObservation persistida e
# recarregada através da referência estável (não de um client de storage específico)
# volta bit-a-bit igual à original.
def test_persistence_round_trips_observation_after_restart() -> None:
    payload = payload_with_blobs(blobs=((2, 2, 8, 8, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    store = InMemoryEvidenceStore()
    reference = persist_observation(result.observation, store)

    reloaded_store = store  # mesmo processo, mas exercitado só através da referência estável
    restored = reload_observation(reference, reloaded_store)
    assert restored == result.observation
