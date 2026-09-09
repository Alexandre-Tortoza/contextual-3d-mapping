"""Composição contextual auditável para uma observação do corridor-02."""

from __future__ import annotations

import json
import math
import shutil
import struct
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

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


# Agrupa os arquivos externos necessários para não esconder paths ou escolher
# implicitamente uma execução de percepção no composition root.
@dataclass(frozen=True)
class Corridor02ContextRequest:
    """Entradas explícitas para associar uma observação visual ao mapa.

    Argumentos:
        geometric_slice: slice geométrico FAST-LIO usado como base.
        bag: rosbag original que preserva RGB e LiDAR timestampados.
        intrinsics: calibração intrínseca MEI do dataset.
        extrinsics: calibração extrínseca LiDAR–IMU–câmera.
        ground_truth: trajetória usada apenas para registrar o scan de evidência.
        visual_observation: saída canônica de visual-perception.
        raw_image: imagem exata consumida por visual-perception.
        overlay_image: preview auditável das regiões e labels.
        valid_area_mask: suporte óptico válido da imagem omnidirecional.
        destination: artifact contextual de destino.
        camera_sequence_index: índice original do frame RGB no rosbag.
    """

    geometric_slice: Path
    bag: Path
    intrinsics: Path
    extrinsics: Path
    ground_truth: Path
    visual_observation: Path
    raw_image: Path
    overlay_image: Path
    valid_area_mask: Path
    destination: Path
    camera_sequence_index: int = 0

    # Falha antes de ler o rosbag quando a composição aponta para artifacts
    # ausentes ou para um índice impossível de interpretar.
    def __post_init__(self) -> None:
        """Valida presença dos arquivos e índice da observação RGB."""
        for path in (
            self.geometric_slice,
            self.bag,
            self.intrinsics,
            self.extrinsics,
            self.ground_truth,
            self.visual_observation,
            self.raw_image,
            self.overlay_image,
            self.valid_area_mask,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.camera_sequence_index < 0:
            raise ValueError("camera_sequence_index must be non-negative.")


# Converte uma matriz de rotação válida em quaternion xyzw para usar o contract
# compartilhado sem introduzir scipy como dependência desta composição.
def _matrix_to_quaternion(matrix: Any) -> tuple[float, float, float, float]:
    """Converte uma matriz 3×3 em quaternion unitário no formato xyzw."""
    import numpy as np

    candidate = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(candidate))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            float((candidate[2, 1] - candidate[1, 2]) / scale),
            float((candidate[0, 2] - candidate[2, 0]) / scale),
            float((candidate[1, 0] - candidate[0, 1]) / scale),
            0.25 * scale,
        )
    else:
        axis = int(np.argmax(np.diag(candidate)))
        next_axis, last_axis = (axis + 1) % 3, (axis + 2) % 3
        scale = math.sqrt(
            1.0 + candidate[axis, axis] - candidate[next_axis, next_axis] - candidate[last_axis, last_axis]
        ) * 2.0
        values = [0.0, 0.0, 0.0, 0.0]
        values[axis] = 0.25 * scale
        values[3] = float((candidate[last_axis, next_axis] - candidate[next_axis, last_axis]) / scale)
        values[next_axis] = float((candidate[next_axis, axis] + candidate[axis, next_axis]) / scale)
        values[last_axis] = float((candidate[last_axis, axis] + candidate[axis, last_axis]) / scale)
        quaternion = tuple(values)
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


# Lê os YAML do dataset e deriva diretamente LiDAR→câmera. O adapter não
# publica as matrizes de terceiros; devolve o contract de calibração do módulo.
def _load_calibration(intrinsics_path: Path, extrinsics_path: Path) -> CameraLidarCalibration:
    """Carrega a calibração MEI e compõe a extrínseca LiDAR→câmera."""
    import numpy as np
    import yaml

    intrinsics_payload = yaml.safe_load(intrinsics_path.read_text(encoding="utf-8"))["rgb_camera"]
    extrinsics_payload = yaml.safe_load(extrinsics_path.read_text(encoding="utf-8"))
    laser_to_imu = np.asarray(extrinsics_payload["laser_to_imu"]["data"], dtype=np.float64).reshape(4, 4)
    camera_to_imu = np.asarray(extrinsics_payload["rgb_camera_to_imu"]["data"], dtype=np.float64).reshape(4, 4)
    laser_to_camera = np.linalg.inv(camera_to_imu) @ laser_to_imu
    projection = intrinsics_payload["projection_parameters"]
    mirror = intrinsics_payload["mirror_parameters"]
    distortion = intrinsics_payload["distortion_parameters"]
    return CameraLidarCalibration(
        calibration_id="corridor-02-camera-lidar-mei-v1",
        artifact=SourceArtifactReference(extrinsics_path.resolve().as_uri(), "application/yaml"),
        model=CameraModel.MEI,
        fx=float(projection["gamma1"]),
        fy=float(projection["gamma2"]),
        cx=float(projection["u0"]),
        cy=float(projection["v0"]),
        lidar_to_camera=RigidTransform(
            FrameId("cmu_rc1_velodyne"),
            FrameId("camera_1_optical_frame"),
            tuple(float(value) for value in laser_to_camera[:3, 3]),
            _matrix_to_quaternion(laser_to_camera[:3, :3]),
        ),
        mirror_xi=float(mirror["xi"]),
        distortion_k1=float(distortion["k1"]),
        distortion_k2=float(distortion["k2"]),
        distortion_p1=float(distortion["p1"]),
        distortion_p2=float(distortion["p2"]),
    )


# Extrai o timestamp do header ROS sem usar o horário de gravação do bag,
# pois o dataset preserva dois clocks diferentes no mesmo arquivo.
def _header_timestamp_ns(message: Any) -> int:
    """Retorna o timestamp inteiro do header de uma mensagem ROS."""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


# Busca o frame RGB pelo índice original e o scan LiDAR temporalmente mais
# próximo. Isso restaura a identidade que o benchmark visual local não reteve.
def _read_synchronized_messages(bag: Path, camera_sequence_index: int) -> tuple[Any, int, Any, int, int]:
    """Lê um frame RGB e o scan LiDAR mais próximo usando timestamps dos headers."""
    from rosbags.highlevel import AnyReader

    with AnyReader([bag]) as reader:
        camera_connections = [item for item in reader.connections if item.topic == "/camera_1/image_raw"]
        lidar_connections = [item for item in reader.connections if item.topic == "/velodyne_points"]
        if len(camera_connections) != 1 or len(lidar_connections) != 1:
            raise ValueError("corridor-02 bag must contain one RGB and one LiDAR connection.")
        camera_message = None
        camera_timestamp_ns = None
        for index, (connection, _, rawdata) in enumerate(reader.messages(connections=camera_connections)):
            if index == camera_sequence_index:
                camera_message = reader.deserialize(rawdata, connection.msgtype)
                camera_timestamp_ns = _header_timestamp_ns(camera_message)
                break
        if camera_message is None or camera_timestamp_ns is None:
            raise ValueError("camera_sequence_index exceeds the RGB stream.")
        nearest: tuple[int, Any, int, int] | None = None
        for index, (connection, _, rawdata) in enumerate(reader.messages(connections=lidar_connections)):
            message = reader.deserialize(rawdata, connection.msgtype)
            timestamp_ns = _header_timestamp_ns(message)
            delta_ns = abs(timestamp_ns - camera_timestamp_ns)
            if nearest is None or (delta_ns, index) < (nearest[0], nearest[2]):
                nearest = (delta_ns, message, index, timestamp_ns)
            if timestamp_ns > camera_timestamp_ns and nearest is not None and delta_ns > nearest[0]:
                break
    if nearest is None:
        raise ValueError("corridor-02 bag does not contain LiDAR messages.")
    return camera_message, camera_timestamp_ns, nearest[1], nearest[3], nearest[2]


# Decodifica somente os campos físicos usados pelo slice e mantém o índice do
# registro original para formar geometry_id e proveniência estáveis.
def _decode_point_cloud(message: Any) -> tuple[tuple[int, tuple[float, float, float], float | None], ...]:
    """Decodifica x, y, z e intensidade de uma mensagem PointCloud2."""
    fields = {field.name: field for field in message.fields}
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PointCloud2 must contain x, y and z fields.")
    endian = ">" if message.is_bigendian else "<"
    float32_datatype = 7
    if any(fields[axis].datatype != float32_datatype for axis in ("x", "y", "z")):
        raise ValueError("corridor-02 PointCloud2 coordinates must be float32.")
    intensity_field = fields.get("intensity")
    payload = bytes(message.data)
    records: list[tuple[int, tuple[float, float, float], float | None]] = []
    for index in range(int(message.width) * int(message.height)):
        offset = index * int(message.point_step)
        coordinates = tuple(
            float(struct.unpack_from(endian + "f", payload, offset + int(fields[axis].offset))[0])
            for axis in ("x", "y", "z")
        )
        if not all(math.isfinite(value) for value in coordinates):
            continue
        intensity = None
        if intensity_field is not None and intensity_field.datatype == float32_datatype:
            intensity = float(struct.unpack_from(endian + "f", payload, offset + int(intensity_field.offset))[0])
        records.append((index, coordinates, intensity))
    return tuple(records)


# Encontra a pose mais próxima e a torna relativa à primeira pose da sequência,
# alinhando o scan de evidência ao frame inicial usado pelo mapa FAST-LIO.
def _relative_pose(ground_truth: Path, timestamp_ns: int) -> tuple[Any, str]:
    """Retorna a matriz de pose relativa mais próxima e sua linha de origem."""
    import numpy as np

    rows = [line.split() for line in ground_truth.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("ground-truth trajectory must not be empty.")
    target_seconds = timestamp_ns / 1_000_000_000
    selected = min(rows, key=lambda row: abs(float(row[0]) - target_seconds))

    # Converte quaternion xyzw e translação no transform homogêneo gravado pelo dataset.
    def transform(row: list[str]) -> Any:
        """Converte uma linha da trajetória em matriz homogênea 4×4."""
        values = [float(value) for value in row]
        tx, ty, tz, qx, qy, qz, qw = values[1:]
        rotation = np.asarray(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
        value = np.eye(4, dtype=np.float64)
        value[:3, :3] = rotation
        value[:3, 3] = (tx, ty, tz)
        return value

    return np.linalg.inv(transform(rows[0])) @ transform(selected), " ".join(selected)


# Extrai o claim primário sem promover alternativas do VLM. O estado de
# suporte continua explícito para o viewer não apresentar predição como verdade.
def _primary_claim(region: dict[str, Any]) -> dict[str, Any] | None:
    """Seleciona o claim de label primário de uma região visual."""
    labels = [claim for claim in region.get("claims", ()) if claim.get("kind") == "label"]
    return next((claim for claim in labels if claim.get("role") == "primary"), labels[0] if labels else None)


# Materializa masks não sobrepostas, priorizando regiões menores e mais
# específicas. Isso adapta proposals sobrepostas ao contract ponto→região único.
def _region_evidence(
    visual_payload: dict[str, Any], valid_mask: Any
) -> tuple[tuple[VisualRegionEvidence, ...], dict[str, dict[str, Any]]]:
    """Converte regiões canônicas em evidências 2D disjuntas e metadados de inspeção."""
    import numpy as np

    height, width = valid_mask.shape
    candidates: list[tuple[int, float, dict[str, Any], dict[str, Any], Any]] = []
    for region in visual_payload["regions"]:
        claim = _primary_claim(region)
        if claim is None:
            continue
        mask_payload = region["mask"]
        if (mask_payload["width"], mask_payload["height"]) != (width, height):
            raise ValueError("visual region mask dimensions must match the RGB frame.")
        flat = np.zeros(width * height, dtype=np.bool_)
        cursor = 0
        value = False
        for run_length in mask_payload["rle"]:
            if value:
                flat[cursor : cursor + run_length] = True
            cursor += run_length
            value = not value
        if cursor != width * height:
            raise ValueError("visual region RLE must cover the complete image.")
        mask = flat.reshape(height, width) & valid_mask
        area = int(mask.sum())
        if area:
            confidence = claim.get("confidence") or {}
            candidates.append((area, -float(confidence.get("value") or 0.0), region, claim, mask))
    ownership = np.full((height, width), -1, dtype=np.int32)
    evidence: list[VisualRegionEvidence] = []
    metadata: dict[str, dict[str, Any]] = {}
    for _, _, region, claim, mask in sorted(candidates, key=lambda item: (item[0], item[1], item[2]["region_id"])):
        region_index = len(evidence)
        accepted = mask & (ownership < 0)
        y_values, x_values = np.nonzero(accepted)
        if not len(x_values):
            continue
        ownership[accepted] = region_index
        region_id = str(region["region_id"])
        evidence.append(
            VisualRegionEvidence(
                region_id,
                frozenset(zip(x_values.tolist(), y_values.tolist(), strict=True)),
                label=str(claim["value"]),
                feature_reference=region.get("visual_embedding_ref"),
            )
        )
        metadata[region_id] = {
            "region_id": region_id,
            "label": claim["value"],
            "confidence": claim.get("confidence"),
            "support": claim.get("support"),
            "category": claim.get("category"),
            "region_kind": claim.get("region_kind"),
            "geometric_confidence": region.get("geometric_confidence"),
            "box": region.get("box"),
            "claims": [
                {
                    "kind": item.get("kind"),
                    "value": item.get("value"),
                    "confidence": item.get("confidence"),
                    "support_state": (item.get("support") or {}).get("state"),
                    "role": item.get("role"),
                }
                for item in region.get("claims", ())
            ],
            "provenance": claim.get("provenance"),
        }
    return tuple(evidence), metadata


# Gera uma paleta estável por label para a camada semântica. A cor é uma
# visualização de um claim, e não a cor física RGB observada pelo sensor.
def _semantic_color(label: str) -> tuple[int, int, int]:
    """Retorna uma cor RGB determinística derivada do label canônico."""
    digest = sha256(label.encode("utf-8")).digest()
    return tuple(70 + value % 166 for value in digest[:3])  # type: ignore[return-value]


# Constrói GeometryPoint com coordenadas de origem LiDAR e coordenadas globais
# separadas; essa distinção é necessária para projetar e renderizar corretamente.
def _geometry_points(
    records: tuple[tuple[int, tuple[float, float, float], float | None], ...],
    pose: Any,
    lidar_reference: ObservationReference,
    map_id: MapId,
) -> tuple[tuple[GeometryPoint, float | None], ...]:
    """Registra os pontos do scan no mapa preservando coordenadas e índices de origem."""
    import numpy as np

    provenance = Provenance(
        "corridor-02-context-association",
        (lidar_reference,),
        (SourceArtifactReference("rosbag://corridor-02/velodyne_points", "sensor_msgs/PointCloud2"),),
    )
    result: list[tuple[GeometryPoint, float | None]] = []
    for source_index, source_coordinates, intensity in records:
        homogeneous = pose @ np.asarray((*source_coordinates, 1.0), dtype=np.float64)
        geometry = GeometryPoint(
            GeometryReference(map_id, f"{map_id}:rgb-lidar:{lidar_reference.sequence_index}:{source_index}"),
            tuple(float(value) for value in homogeneous[:3]),
            source_coordinates,
            lidar_reference,
            provenance,
        )
        result.append((geometry, intensity))
    return tuple(result)


# Monta o artifact v2 unindo a geometria completa amostrada a um scan com
# evidência RGB/semântica real. Somente pontos visíveis são adicionados à camada.
def export_corridor02_context(request: Corridor02ContextRequest) -> Path:
    """Exporta um mapa com associação RGB, contexto VLM e previews auditáveis."""
    import numpy as np
    from PIL import Image

    base = json.loads(request.geometric_slice.read_text(encoding="utf-8"))
    if base.get("schema_version") != 1 or not isinstance(base.get("points"), list):
        raise ValueError("geometric_slice must be a schema_version 1 point slice.")
    visual_payload = json.loads(request.visual_observation.read_text(encoding="utf-8"))
    camera_message, camera_timestamp_ns, lidar_message, lidar_timestamp_ns, lidar_index = (
        _read_synchronized_messages(request.bag, request.camera_sequence_index)
    )
    if (int(camera_message.width), int(camera_message.height)) != (
        int(visual_payload["image_width"]),
        int(visual_payload["image_height"]),
    ):
        raise ValueError("visual observation dimensions differ from the source RGB message.")
    raw_rgb = np.asarray(Image.open(request.raw_image).convert("RGB"), dtype=np.uint8)
    valid_mask = np.asarray(Image.open(request.valid_area_mask).convert("L"), dtype=np.uint8) > 0
    if raw_rgb.shape[:2] != valid_mask.shape:
        raise ValueError("valid-area mask dimensions must match the RGB image.")
    camera_frame = FrameId("camera_1_optical_frame")
    lidar_frame = FrameId(str(lidar_message.header.frame_id))
    calibration = _load_calibration(request.intrinsics, request.extrinsics)
    if calibration.lidar_to_camera.source_frame != lidar_frame:
        raise ValueError("LiDAR message frame differs from the calibration source frame.")
    rgb_reference = ObservationReference(
        observation_id=f"corridor-02:camera_1:{request.camera_sequence_index}",
        dataset_id="corridor-02",
        sequence_id="corridor-02",
        sensor_id="camera_1",
        sequence_index=request.camera_sequence_index,
        timestamp=Timestamp(camera_timestamp_ns, "corridor-02-sensor"),
        frame_id=camera_frame,
        calibration_id=calibration.calibration_id,
    )
    lidar_reference = ObservationReference(
        observation_id=f"corridor-02:velodyne:{lidar_index}",
        dataset_id="corridor-02",
        sequence_id="corridor-02",
        sensor_id="velodyne",
        sequence_index=lidar_index,
        timestamp=Timestamp(lidar_timestamp_ns, "corridor-02-sensor"),
        frame_id=lidar_frame,
    )
    rgb = RgbFrame(
        rgb_reference,
        width=raw_rgb.shape[1],
        height=raw_rgb.shape[0],
        pixels=tuple(tuple(int(channel) for channel in pixel) for pixel in raw_rgb.reshape(-1, 3)),
        valid_pixels=frozenset(
            (int(x), int(y)) for y, x in zip(*np.nonzero(valid_mask), strict=True)
        ),
    )
    regions, region_metadata = _region_evidence(visual_payload, valid_mask)
    pose, pose_source = _relative_pose(request.ground_truth, lidar_timestamp_ns)
    map_id = MapId(str(base["map_id"]))
    geometry_with_intensity = _geometry_points(
        _decode_point_cloud(lidar_message), pose, lidar_reference, map_id
    )
    geometry = tuple(item[0] for item in geometry_with_intensity)
    associations = associate_points(
        geometry,
        rgb,
        calibration,
        regions,
        max_time_delta_ns=100_000_000,
    )
    intensity_by_id = {
        point.reference.geometry_id: intensity for point, intensity in geometry_with_intensity
    }
    associated_records = []
    for point, association in zip(geometry, associations, strict=True):
        if association.status is not AssociationStatus.ASSOCIATED:
            continue
        associated_records.append(
            {
                "geometry_id": point.reference.geometry_id,
                "coordinates_m": point.coordinates_m,
                "source_coordinates_m": point.source_coordinates_m,
                "intensity": intensity_by_id[point.reference.geometry_id],
                "association": {
                    "status": association.status.value,
                    "rgb_observation_id": rgb_reference.observation_id,
                    "rgb_timestamp_ns": camera_timestamp_ns,
                    "lidar_observation_id": lidar_reference.observation_id,
                    "lidar_timestamp_ns": lidar_timestamp_ns,
                    "time_delta_ns": abs(camera_timestamp_ns - lidar_timestamp_ns),
                    "pixel": association.pixel,
                    "color_rgb": association.color_rgb,
                    "region_id": association.region_id,
                    "label": association.label,
                    "semantic_color_rgb": (
                        None if association.label is None else _semantic_color(association.label)
                    ),
                    "calibration_id": calibration.calibration_id,
                },
                "provenance": {
                    "producer": point.provenance.producer,
                    "observation_ids": [lidar_reference.observation_id, rgb_reference.observation_id],
                    "pose_source": pose_source,
                },
            }
        )
    assets = request.destination.parent / f"{request.destination.stem}-assets"
    assets.mkdir(parents=True, exist_ok=True)
    raw_destination = assets / "corridor-02-000-raw.png"
    overlay_destination = assets / "corridor-02-000-regions.png"
    shutil.copyfile(request.raw_image, raw_destination)
    shutil.copyfile(request.overlay_image, overlay_destination)
    observation_id = rgb_reference.observation_id
    scene_claims = [
        {
            "kind": claim.get("kind"),
            "value": claim.get("value"),
            "confidence": claim.get("confidence"),
            "support_state": (claim.get("support") or {}).get("state"),
            "provenance": claim.get("provenance"),
        }
        for claim in visual_payload.get("scene_context", {}).get("claims", ())
    ]
    base["schema_version"] = 2
    base["artifact_type"] = "contextual_rgb_lidar_slice"
    base["capabilities"] = {
        "geometric_map": "available",
        "rgb_association": "partial",
        "semantic_overlay": "partial_unverified",
        "evidence_inspection": "available",
    }
    base["observations"] = [
        {
            "observation_id": observation_id,
            "timestamp_ns": camera_timestamp_ns,
            "sensor_id": rgb_reference.sensor_id,
            "frame_id": str(rgb_reference.frame_id),
            "width": rgb.width,
            "height": rgb.height,
            "raw_image_uri": f"{assets.name}/{raw_destination.name}",
            "overlay_image_uri": f"{assets.name}/{overlay_destination.name}",
            "scene_claims": scene_claims,
            "visual_artifact": request.visual_observation.resolve().as_uri(),
            "visual_artifact_declared_timestamp_ns": visual_payload["source"]["timestamp"]["nanoseconds"],
            "timestamp_resolution": "restored_from_rosbag_sequence_index",
        }
    ]
    base["regions"] = list(region_metadata.values())
    base["points"].extend(associated_records)
    base["context_summary"] = {
        "visual_observation_count": 1,
        "contextual_point_count": sum(
            point.get("association", {}).get("region_id") is not None for point in associated_records
        ),
        "rgb_point_count": len(associated_records),
        "unobserved_geometric_point_count": len(base["points"]) - len(associated_records),
        "claim_status": "predição VLM não verificada",
    }
    request.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = request.destination.with_suffix(request.destination.suffix + ".tmp")
    temporary.write_text(json.dumps(base, separators=(",", ":")), encoding="utf-8")
    temporary.replace(request.destination)
    return request.destination
