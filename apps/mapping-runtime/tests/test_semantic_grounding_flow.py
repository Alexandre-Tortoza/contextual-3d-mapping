"""Integração de grounding visual, ownership, projeção e ablações temporais."""

from dataclasses import replace
from math import sqrt

import numpy as np
import pytest
from geometric_map import GeometryReference
from mapping_runtime.corridor02_context import _pose_matrix, _region_evidence, _sample_pose
from sensor_association import (
    AssociationStatus,
    BoundaryPolicy,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    RgbFrame,
    SemanticAssociationStatus,
    associate_map_points,
)
from state_estimation import interpolate_pose

from contextual_mapping_contracts import (
    FrameId,
    MapId,
    ObservationReference,
    Pose,
    RigidTransform,
    SourceArtifactReference,
    Timestamp,
)
from visual_perception import (
    GroundingPrediction,
    GroundingStatus,
    ImageAreaMasks,
    ImagePayload,
    Mask,
    ModuleConfig,
    create_perception_ports,
    ground_regions,
    run_canonical_pipeline,
    serialize_observation,
)
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.references import ModelProvenance
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner


# Fornece um segmentador controlado para que todas as outras fronteiras sejam
# reais; a geometria nova é independente do label e da discovery ampla.
class ObjectGrounder:
    """Retorna o footprint da porta de uma fixture, ou uma falha controlada."""

    # Recebe somente a condição de falha a exercitar no pipeline.
    def __init__(self, fail: bool = False) -> None:
        """Define se a segmentação deve se abster."""
        self.fail = fail

    # Simula a fronteira de modelo sem substituir o stage de grounding real.
    def ground(self, image, requests, config):
        """Segmenta um objeto central que ocupa só parte de discovery."""
        data = np.zeros((32, 32), dtype=np.bool_)
        data[6:24, 8:18] = True
        mask = Mask(data, 32, 32)
        return tuple(GroundingPrediction(
            item.region_id, item.concept, None if self.fail else mask,
            None if self.fail else mask.bounding_box(),
            provenance=(ModelProvenance("semantic_grounding", "fixture", "flow/1"),),
            status=GroundingStatus.FAILED if self.fail else GroundingStatus.REFINED,
            reason="Falha de grounding simulada." if self.fail else None,
        ) for item in requests)


# Compõe o pipeline visual completo com RGB contendo discovery mais ampla que
# a porta e leva sua saída serializada à fronteira real do mapping-runtime.
def _flow(*, fail: bool = False, policy: BoundaryPolicy | None = None, invalid_pixel=None):
    """Executa reconhecimento → grounding → evidence → associação 3D."""
    reference = ObservationReference("rgb", "dataset", "sequence", "camera", 0, Timestamp(100, "clock"), FrameId("camera"))
    image = ImageObservation(32, 32, "rgb8", SourceArtifactReference("fixture://raw", "image/png"), reference)
    pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
    pixels[2:27, 2:25] = (180, 20, 20)
    config = ModuleConfig()
    ports = replace(create_perception_ports(config), semantic_grounder=ObjectGrounder(fail),
        multimodal_reasoner=FakeMultimodalReasoner(region_response_fn=lambda request: {"label": "door", "kind": "thing"}))
    result = run_canonical_pipeline(image, ImagePayload(pixels, 32, 32), config, ports)
    valid = np.ones((32, 32), dtype=np.bool_)
    if invalid_pixel is not None:
        valid[invalid_pixel[1], invalid_pixel[0]] = False
    evidence, metadata = _region_evidence(serialize_observation(result.observation), valid, "fixture-frame", boundary_policy=policy)
    rgb = RgbFrame(reference, 32, 32, tuple(map(tuple, pixels.reshape(-1, 3).tolist())),
        frozenset((x, y) for y in range(32) for x in range(32) if valid[y, x]))
    calibration = CameraLidarCalibration("calibration", SourceArtifactReference("fixture://calibration", "application/json"),
        CameraModel.PINHOLE, 1, 1, 0, 0, RigidTransform(FrameId("lidar"), FrameId("camera"), (0, 0, 0), (0, 0, 0, 1)))
    points = tuple(MapAnchoredPoint(GeometryReference(MapId("map"), name), (float(x), float(y), 1.0))
        for name, x, y in (("interior", 12, 15), ("boundary", 8, 15), ("wall", 22, 15)))
    lidar = replace(reference, observation_id="lidar", sensor_id="lidar", frame_id=FrameId("lidar"))
    associations = associate_map_points(points, rgb, calibration,
        RigidTransform(FrameId("map"), FrameId("camera"), (0, 0, 0), (0, 0, 0, 1)), evidence,
        lidar_observation=lidar, boundary_policy=policy)
    return result, associations, metadata


# O ponto da parede estava em discovery e agora está fora do footprint; a
# boundary permanece tentativa e somente o interior vira contribuição forte.
def test_visual_grounding_to_sensor_association_excludes_wall_and_marks_boundary() -> None:
    """Protege o fluxo entre módulos e a diferença entre RGB e semântica."""
    visual, (interior, boundary, wall), metadata = _flow()
    assert visual.observation.regions[0].mask.data[15, 22]
    assert interior.label == "door" and interior.semantic_status is SemanticAssociationStatus.INTERIOR
    assert interior.distance_to_boundary_px == 4
    assert boundary.label is None and boundary.tentative_label == "door"
    assert boundary.semantic_status is SemanticAssociationStatus.BOUNDARY
    assert boundary.distance_to_boundary_px == 0
    assert wall.label is None and wall.tentative_label is None
    assert wall.semantic_status is SemanticAssociationStatus.OUTSIDE
    assert all(item.status is AssociationStatus.ASSOCIATED for item in (interior, boundary, wall))
    diagnostics = next(iter(metadata.values()))["spatial_diagnostics"]
    assert diagnostics["safe_interior_pixel_count"] + diagnostics["boundary_pixel_count"] == diagnostics["association_area"]
    assert interior.grounding_reference


# Com a segmentação indisponível, as claims do pipeline não desaparecem; só o
# footprint forte fica ausente, inclusive no ponto que antes seria interior.
def test_grounding_failure_preserves_claim_without_authorizing_spatial_label() -> None:
    """Falha observável atravessa runtime e associação sem fallback implícito."""
    visual, associations, metadata = _flow(fail=True)
    assert visual.observation.regions[0].claims[0].value == "door"
    assert all(item.label is None for item in associations)
    assert next(iter(metadata.values()))["spatial_diagnostics"]["semantic_grounding_status"] == "grounding_failed"


# A ablação da margem não pode restaurar pixels de parede fora da segmentação.
def test_boundary_policy_is_independent_of_grounding_and_configurable() -> None:
    """Desligar boundary só muda pixels que pertencem à semantic mask."""
    _, (_, boundary, wall), _ = _flow(policy=BoundaryPolicy(enabled=False))
    assert boundary.label == "door" and wall.label is None
    _, (interior, _, _), _ = _flow(policy=BoundaryPolicy(registration_sigma_px=2.0))
    assert interior.label is None and interior.tentative_label == "door"


# Um pixel excluído pela área válida continua rejeitado antes da semântica.
def test_valid_area_exclusion_precedes_grounding_association() -> None:
    """Grounding não contorna o suporte óptico do frame."""
    _, (interior, _, _), _ = _flow(invalid_pixel=(12, 15))
    assert interior.status is AssociationStatus.OUTSIDE_VALID_SUPPORT
    assert interior.label is None and interior.color_rgb is None


# Grounding tem de respeitar o rig mesmo se o segmentador preencher seus pixels.
# A máscara aceita atravessa a serialização e a mesma conversão usada no runtime.
def test_grounding_to_runtime_preserves_ego_exclusion() -> None:
    """Pixels do rig não são reintroduzidos como footprint semântico."""
    visual, _, _ = _flow()
    ego = np.zeros((32, 32), dtype=np.bool_)
    ego[20:32] = True
    valid = Mask(np.ones((32, 32), dtype=np.bool_), 32, 32)
    regions = ground_regions(visual.observation.regions,
        ImagePayload(np.zeros((32, 32, 3), dtype=np.uint8), 32, 32), ObjectGrounder(),
        ModuleConfig().semantic_grounding, ImageAreaMasks(valid, Mask(ego, 32, 32)))
    observation = replace(visual.observation, regions=regions)
    evidence, _ = _region_evidence(serialize_observation(observation), valid.data, "ego-frame")
    assert evidence
    assert all(y < 20 for region in evidence for _, y in region.pixels)


# Um buraco pequeno é boundary interna, mesmo quando fica longe do contorno
# externo. A decisão deve depender do footprint, não apenas de sua box.
def test_grounded_mask_hole_marks_boundary_without_erasing_the_mask() -> None:
    """Um ponto ao lado de um buraco perde força, mantendo a hipótese tentativa."""
    _, (interior, _, _), _ = _flow(invalid_pixel=(11, 15))
    assert interior.status is AssociationStatus.ASSOCIATED
    assert interior.semantic_status is SemanticAssociationStatus.BOUNDARY
    assert interior.tentative_label == "door" and interior.label is None


# Um artifact legado continua legível sem alegar grounding que nunca ocorreu.
def test_legacy_payload_requires_explicit_ablation_to_publish_discovery() -> None:
    """Uma migração de schema não transforma discovery em evidência forte."""
    visual, _, _ = _flow()
    payload = serialize_observation(visual.observation)
    payload["schema_version"] = 4
    for region in payload["regions"]:
        region.pop("grounding")
    evidence, metadata = _region_evidence(payload, np.ones((32, 32), dtype=np.bool_), "legacy")
    assert not evidence and metadata
    legacy, _ = _region_evidence(payload, np.ones((32, 32), dtype=np.bool_), "legacy", legacy_discovery=True)
    assert legacy and all(item.grounding_status == "legacy_discovery" for item in legacy)


# Translação e rotação intermediárias precisam corresponder ao timestamp RGB,
# preservando os dois apoios temporais e permitindo comparação com nearest.
def test_pose_interpolation_uses_linear_translation_and_shortest_slerp() -> None:
    """Interpola metade de uma translação e de uma rotação de 90 graus."""
    before = _pose_matrix((0, 0, 0), (0, 0, 0, 1))
    after = _pose_matrix((2, 0, 0), (0, 0, sqrt(0.5), sqrt(0.5)))
    stamp, matrix, provenance = _sample_pose(((100, before), (200, after)), 150, "interpolated")
    assert stamp == 150 and matrix[:3, 3] == pytest.approx((1, 0, 0))
    assert matrix[:3, :3] @ np.array([1, 0, 0]) == pytest.approx((sqrt(0.5), sqrt(0.5), 0))
    assert provenance["support_timestamps_ns"] == [100, 200] and provenance["fraction"] == 0.5
    nearest, _, _ = _sample_pose(((100, before), (200, after)), 150, "nearest")
    assert nearest == 100
    with pytest.raises(ValueError, match="outside"):
        _sample_pose(((100, before), (200, after)), 250, "interpolated")
    with pytest.raises(ValueError, match="gap"):
        _sample_pose(((100, before), (200, after)), 150, "interpolated", max_gap_ns=50)


# Quaternions com sinais opostos descrevem a mesma orientação; SLERP não deve
# dar uma volta inteira nem produzir divisão por zero nesse caso.
def test_pose_interpolation_handles_antipodal_quaternions() -> None:
    """Mantém orientação unitária para quaternions equivalentes."""
    before = Pose(RigidTransform(FrameId("body"), FrameId("map"), (0, 0, 0), (0, 0, 0, 1)), 100)
    after = Pose(replace(before.transform, rotation_xyzw=(0, 0, 0, -1)), 200)
    assert interpolate_pose(before, after, 150).transform.rotation_xyzw == (0, 0, 0, 1)
