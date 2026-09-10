"""Composição contextual auditável de um trecho do corridor-02."""

from __future__ import annotations

import json
import math
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geometric_map import GeometryReference
from semantic_fusion import SemanticContribution, fuse_point_contributions
from sensor_association import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    RgbFrame,
    VisualRegionEvidence,
    associate_map_points,
)

from contextual_mapping_contracts import (
    FrameId,
    MapId,
    ObservationReference,
    RigidTransform,
    SourceArtifactReference,
    Timestamp,
)

# Ordem de informatividade das rejeições. Um ponto que a câmera enquadrou e
# perdeu por oclusão diz algo sobre a cena; um ponto que ficou atrás da câmera
# em todos os keyframes diz apenas que ninguém olhou para lá. O viewer mostra a
# rejeição mais informativa entre as observações.
_REJECTION_PRIORITY = (
    AssociationStatus.OCCLUDED,
    AssociationStatus.OUTSIDE_VALID_SUPPORT,
    AssociationStatus.OUTSIDE_IMAGE,
    AssociationStatus.BEHIND_CAMERA,
)


# Agrupa os arquivos de um keyframe visual e sua identidade temporal. As três
# identidades vêm de `mapping-runtime bag-window`, porque o bag preserva dois
# relógios e nenhuma delas é derivável das outras.
@dataclass(frozen=True)
class Corridor02Keyframe:
    """Entradas de um keyframe visual do trecho.

    Argumentos:
        frame_id: identidade do frame usada nos artifacts de percepção.
        camera_sequence_index: posição do frame no stream RGB do bag.
        header_timestamp_ns: instante do frame no relógio do dataset.
        bag_timestamp_ns: instante de gravação, usado para ler a mensagem.
        visual_observation: saída canônica de visual-perception.
        raw_image: imagem exata consumida por visual-perception.
        overlay_image: preview auditável das regiões e labels.
        valid_area_mask: suporte óptico válido da imagem omnidirecional.
    """

    frame_id: str
    camera_sequence_index: int
    header_timestamp_ns: int
    bag_timestamp_ns: int
    visual_observation: Path
    raw_image: Path
    overlay_image: Path
    valid_area_mask: Path

    # Falha antes de abrir o bag quando a composição aponta para artifacts
    # ausentes, onde o erro seria atribuído à leitura do dataset.
    def __post_init__(self) -> None:
        """Valida presença dos artifacts de percepção do keyframe."""
        for path in (
            self.visual_observation,
            self.raw_image,
            self.overlay_image,
            self.valid_area_mask,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.camera_sequence_index < 0:
            raise ValueError("camera_sequence_index must be non-negative.")


# Agrupa os arquivos externos necessários para não esconder paths ou escolher
# implicitamente uma execução de percepção no composition root.
@dataclass(frozen=True)
class Corridor02ContextRequest:
    """Entradas explícitas para contextualizar um trecho do mapa.

    Argumentos:
        geometric_slice: slice geométrico FAST-LIO usado como base.
        bag: rosbag original que preserva RGB e LiDAR timestampados.
        intrinsics: calibração intrínseca MEI do dataset.
        extrinsics: calibração extrínseca LiDAR–IMU–câmera.
        keyframes: keyframes visuais que contextualizam o trecho.
        destination: artifact contextual de destino.
        odometry: trajetória estimada pelo FAST-LIO, fonte de pose preferencial.
        ground_truth: trajetória do dataset, usada quando não há odometria.
        pose_anchor_ns: instante em que o mapa foi iniciado, usado para alinhar
            o ground-truth à origem do mapa FAST-LIO.
    """

    geometric_slice: Path
    bag: Path
    intrinsics: Path
    extrinsics: Path
    keyframes: tuple[Corridor02Keyframe, ...]
    destination: Path
    odometry: Path | None = None
    ground_truth: Path | None = None
    pose_anchor_ns: int | None = None

    # Exige uma fonte de pose e ao menos um keyframe antes de qualquer leitura,
    # porque ambos são pré-condições da composição, e não erros de dados.
    def __post_init__(self) -> None:
        """Valida arquivos, keyframes e disponibilidade de uma fonte de pose."""
        for path in (self.geometric_slice, self.bag, self.intrinsics, self.extrinsics):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not self.keyframes:
            raise ValueError("at least one keyframe is required.")
        available = [path for path in (self.odometry, self.ground_truth) if path is not None]
        if not available:
            raise ValueError("either odometry or ground_truth must be provided.")
        for path in available:
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.ground_truth is not None and self.odometry is None and self.pose_anchor_ns is None:
            raise ValueError("ground-truth poses require pose_anchor_ns.")


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


# Monta a matriz homogênea de uma pose translação + quaternion xyzw. Existe
# porque as duas fontes de pose deste trecho — odometria e ground-truth —
# publicam poses nesse mesmo formato.
def _pose_matrix(translation: tuple[float, float, float], rotation_xyzw: tuple[float, ...]) -> Any:
    """Converte translação e quaternion em matriz homogênea 4×4."""
    import numpy as np

    qx, qy, qz, qw = (float(value) for value in rotation_xyzw)
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
    value[:3, 3] = translation
    return value


# Lê os YAML do dataset e deriva as extrínsecas necessárias. O adapter não
# publica as matrizes de terceiros; devolve o contract de calibração do módulo
# e, separadamente, a extrínseca LiDAR→IMU exigida pelas fontes de pose.
def _load_calibration(intrinsics_path: Path, extrinsics_path: Path) -> tuple[CameraLidarCalibration, Any]:
    """Carrega a calibração MEI, a extrínseca LiDAR→câmera e a LiDAR→IMU."""
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
    calibration = CameraLidarCalibration(
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
    return calibration, laser_to_imu


# Extrai o timestamp do header ROS sem usar o horário de gravação do bag,
# pois o dataset preserva dois clocks diferentes no mesmo arquivo.
def _header_timestamp_ns(message: Any) -> int:
    """Retorna o timestamp inteiro do header de uma mensagem ROS."""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


# Lê a odometria gravada durante a execução do FAST-LIO. É a fonte de pose
# preferencial porque descreve exatamente a trajetória que originou o mapa;
# com o contexto ancorado nos pontos do mapa, um erro de pose vira pixel errado
# e, portanto, label errado.
def _load_odometry(path: Path) -> tuple[tuple[int, Any], ...]:
    """Carrega poses corpo→mapa a partir do CSV de `rostopic echo -p`.

    Argumentos:
        path: CSV produzido durante a execução do FAST-LIO.
    Retorna:
        pares ``(timestamp_ns, matriz 4×4)`` ordenados por tempo.
    Levanta:
        ValueError: se o CSV não declarar as colunas de pose esperadas.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"odometry file {path} does not contain samples.")
    header = [column.strip() for column in lines[0].split(",")]
    required = {
        "stamp": "field.header.stamp",
        "x": "field.pose.pose.position.x",
        "y": "field.pose.pose.position.y",
        "z": "field.pose.pose.position.z",
        "qx": "field.pose.pose.orientation.x",
        "qy": "field.pose.pose.orientation.y",
        "qz": "field.pose.pose.orientation.z",
        "qw": "field.pose.pose.orientation.w",
    }
    missing = [column for column in required.values() if column not in header]
    if missing:
        raise ValueError(f"odometry file {path} is missing columns: {missing}.")
    index = {name: header.index(column) for name, column in required.items()}
    samples: list[tuple[int, Any]] = []
    for line in lines[1:]:
        values = line.split(",")
        if len(values) < len(header):
            continue
        translation = tuple(float(values[index[axis]]) for axis in ("x", "y", "z"))
        rotation = tuple(float(values[index[axis]]) for axis in ("qx", "qy", "qz", "qw"))
        samples.append((int(values[index["stamp"]]), _pose_matrix(translation, rotation)))
    if not samples:
        raise ValueError(f"odometry file {path} does not contain usable samples.")
    return tuple(sorted(samples, key=lambda item: item[0]))


# Lê a trajetória de ground-truth e a realinha à origem do mapa. O FAST-LIO
# começa o mapa na pose do primeiro scan processado, então a âncora precisa ser
# aquele instante — e não a primeira linha do arquivo, que descreve o início do
# dataset inteiro.
def _load_ground_truth(path: Path, anchor_ns: int) -> tuple[tuple[int, Any], ...]:
    """Carrega poses corpo→mapa a partir da trajetória do dataset.

    Argumentos:
        path: arquivo de trajetória ``timestamp tx ty tz qx qy qz qw``.
        anchor_ns: instante que define a origem do mapa.
    Retorna:
        pares ``(timestamp_ns, matriz 4×4)`` relativos à âncora.
    Levanta:
        ValueError: se a trajetória estiver vazia.
    """
    import numpy as np

    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("ground-truth trajectory must not be empty.")
    samples = [
        (
            int(round(float(row[0]) * 1_000_000_000)),
            _pose_matrix(
                (float(row[1]), float(row[2]), float(row[3])),
                (float(row[4]), float(row[5]), float(row[6]), float(row[7])),
            ),
        )
        for row in rows
    ]
    samples.sort(key=lambda item: item[0])
    anchor = min(samples, key=lambda item: abs(item[0] - anchor_ns))[1]
    inverse_anchor = np.linalg.inv(anchor)
    return tuple((timestamp, inverse_anchor @ matrix) for timestamp, matrix in samples)


# Seleciona a pose mais próxima de um instante. A escolha é por vizinho mais
# próximo, e não por interpolação, para que a proveniência aponte para uma
# amostra real da trajetória.
def _nearest_pose(samples: tuple[tuple[int, Any], ...], timestamp_ns: int) -> tuple[int, Any]:
    """Retorna a amostra de pose mais próxima do instante pedido."""
    times = [sample[0] for sample in samples]
    position = bisect_left(times, timestamp_ns)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(samples)]
    return samples[min(candidates, key=lambda index: abs(times[index] - timestamp_ns))]


# Lê uma mensagem identificada pelo instante de gravação. Existe para que a
# composição não percorra o stream inteiro a cada keyframe: neste dataset o
# stream RGB tem dezenas de gigabytes.
def _message_at(reader: Any, connections: list[Any], bag_timestamp_ns: int) -> Any:
    """Desserializa a mensagem gravada em um instante conhecido."""
    for connection, _, rawdata in reader.messages(
        connections=connections, start=bag_timestamp_ns, stop=bag_timestamp_ns + 1
    ):
        return reader.deserialize(rawdata, connection.msgtype)
    raise ValueError(f"no message recorded at bag timestamp {bag_timestamp_ns}.")


# Encontra o scan LiDAR sincronizado com um keyframe RGB. O scan não fornece
# geometria — o mapa já a fornece — mas ancora temporalmente a pose usada na
# projeção, e essa âncora precisa ser um instante medido, não estimado.
def _nearest_lidar(
    reader: Any,
    connections: list[Any],
    bag_times: list[int],
    camera_header_ns: int,
    clock_offset_ns: int,
) -> tuple[int, int]:
    """Retorna o índice e o timestamp de header do scan LiDAR mais próximo."""
    estimate = camera_header_ns + clock_offset_ns
    position = bisect_left(bag_times, estimate)
    candidates = [index for index in (position - 1, position, position + 1) if 0 <= index < len(bag_times)]
    if not candidates:
        raise ValueError("bag does not contain LiDAR messages.")
    measured = {
        index: _header_timestamp_ns(_message_at(reader, connections, bag_times[index]))
        for index in candidates
    }
    best = min(measured, key=lambda index: (abs(measured[index] - camera_header_ns), index))
    return best, measured[best]


# Extrai o claim primário sem promover alternativas do VLM. O estado de
# suporte continua explícito para o viewer não apresentar predição como verdade.
def _primary_claim(region: dict[str, Any]) -> dict[str, Any] | None:
    """Seleciona o claim de label primário de uma região visual."""
    labels = [claim for claim in region.get("claims", ()) if claim.get("kind") == "label"]
    return next((claim for claim in labels if claim.get("role") == "primary"), labels[0] if labels else None)


# Materializa masks não sobrepostas, priorizando regiões menores e mais
# específicas. Isso adapta proposals sobrepostas ao contract ponto→região único.
# Os identificadores recebem o prefixo do keyframe porque, com vários frames, o
# mesmo identificador de região pode reaparecer em observações diferentes.
def _region_evidence(
    visual_payload: dict[str, Any], valid_mask: Any, frame_id: str
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
        region_id = f"{frame_id}:{region['region_id']}"
        support = region.get("support") or {}
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
            "observation_frame_id": frame_id,
            "label": claim["value"],
            "confidence": claim.get("confidence"),
            "support": claim.get("support"),
            "category": claim.get("category"),
            "region_kind": claim.get("region_kind"),
            "geometric_confidence": region.get("geometric_confidence"),
            "visual_support": support.get("visual_support"),
            "region_quality": support.get("region_quality"),
            "calibrated_confidence": support.get("calibrated_confidence"),
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


# Constrói o frame RGB consumido pela associação. Converte via ``tolist`` em vez
# de laços Python porque a conversão acontece uma vez por keyframe e domina o
# custo da composição quando o trecho tem muitos frames.
def _rgb_frame(reference: ObservationReference, raw_rgb: Any, valid_mask: Any) -> RgbFrame:
    """Converte imagem e máscara de suporte no contract de frame RGB."""
    import numpy as np

    y_values, x_values = np.nonzero(valid_mask)
    return RgbFrame(
        reference,
        width=int(raw_rgb.shape[1]),
        height=int(raw_rgb.shape[0]),
        pixels=tuple(map(tuple, raw_rgb.reshape(-1, 3).tolist())),
        valid_pixels=frozenset(zip(x_values.tolist(), y_values.tolist(), strict=True)),
    )


# Monta o artifact v2 rotulando os próprios pontos do mapa persistente. Anexar
# o scan colorido ao mapa criava duas amostragens da mesma superfície a poucos
# centímetros uma da outra, das quais só uma carregava contexto — o "ponto
# fantasma" observado no viewer. Ancorando no mapa, cada ponto persistido passa
# a ter ou não label, e a ausência passa a declarar seu motivo.
def export_corridor02_context(request: Corridor02ContextRequest) -> Path:
    """Exporta um trecho do mapa contextualizado por múltiplos keyframes.

    Argumentos:
        request: entradas explícitas do trecho, keyframes e fonte de pose.
    Retorna:
        caminho do artifact contextual gravado.
    Levanta:
        ValueError: se o slice, a calibração ou as observações forem incompatíveis.
    """
    import numpy as np
    from PIL import Image
    from rosbags.highlevel import AnyReader

    base = json.loads(request.geometric_slice.read_text(encoding="utf-8"))
    if base.get("schema_version") != 1 or not isinstance(base.get("points"), list):
        raise ValueError("geometric_slice must be a schema_version 1 point slice.")
    map_id = MapId(str(base["map_id"]))
    map_frame = FrameId(str(base["map_frame"]))
    camera_frame = FrameId("camera_1_optical_frame")
    calibration, laser_to_imu = _load_calibration(request.intrinsics, request.extrinsics)
    laser_to_camera = _pose_matrix(
        calibration.lidar_to_camera.translation_m, calibration.lidar_to_camera.rotation_xyzw
    )
    map_points = tuple(
        MapAnchoredPoint(
            GeometryReference(map_id, str(point["geometry_id"])),
            tuple(float(value) for value in point["coordinates_m"]),
        )
        for point in base["points"]
    )
    if request.odometry is not None:
        poses = _load_odometry(request.odometry)
        pose_source = "fastlio_odometry"
    else:
        assert request.ground_truth is not None and request.pose_anchor_ns is not None
        poses = _load_ground_truth(request.ground_truth, request.pose_anchor_ns)
        pose_source = "dataset_ground_truth"

    assets = request.destination.parent / f"{request.destination.stem}-assets"
    assets.mkdir(parents=True, exist_ok=True)
    associated: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, AssociationStatus] = {}
    observations: list[dict[str, Any]] = []
    regions: dict[str, dict[str, Any]] = {}

    with AnyReader([request.bag]) as reader:
        camera_connections = [item for item in reader.connections if item.topic == "/camera_1/image_raw"]
        lidar_connections = [item for item in reader.connections if item.topic == "/velodyne_points"]
        if len(camera_connections) != 1 or len(lidar_connections) != 1:
            raise ValueError("corridor-02 bag must contain one RGB and one LiDAR connection.")
        lidar_identifiers = {item.id for item in lidar_connections}
        lidar_bag_times = sorted(
            entry.time
            for source in reader.readers
            for connection_id, entries in source.indexes.items()
            if connection_id in lidar_identifiers
            for entry in entries
        )
        for keyframe in request.keyframes:
            camera_message = _message_at(reader, camera_connections, keyframe.bag_timestamp_ns)
            camera_timestamp_ns = _header_timestamp_ns(camera_message)
            if camera_timestamp_ns != keyframe.header_timestamp_ns:
                raise ValueError(
                    f"keyframe {keyframe.frame_id} header timestamp does not match the recorded message."
                )
            visual_payload = json.loads(keyframe.visual_observation.read_text(encoding="utf-8"))
            if (int(camera_message.width), int(camera_message.height)) != (
                int(visual_payload["image_width"]),
                int(visual_payload["image_height"]),
            ):
                raise ValueError("visual observation dimensions differ from the source RGB message.")
            lidar_index, lidar_timestamp_ns = _nearest_lidar(
                reader,
                lidar_connections,
                lidar_bag_times,
                camera_timestamp_ns,
                keyframe.bag_timestamp_ns - camera_timestamp_ns,
            )
            pose_timestamp_ns, body_to_map = _nearest_pose(poses, camera_timestamp_ns)
            lidar_to_map = body_to_map @ laser_to_imu
            map_to_camera_matrix = laser_to_camera @ np.linalg.inv(lidar_to_map)
            map_to_camera = RigidTransform(
                map_frame,
                camera_frame,
                tuple(float(value) for value in map_to_camera_matrix[:3, 3]),
                _matrix_to_quaternion(map_to_camera_matrix[:3, :3]),
            )
            raw_rgb = np.asarray(Image.open(keyframe.raw_image).convert("RGB"), dtype=np.uint8)
            valid_mask = np.asarray(Image.open(keyframe.valid_area_mask).convert("L"), dtype=np.uint8) > 0
            if raw_rgb.shape[:2] != valid_mask.shape:
                raise ValueError("valid-area mask dimensions must match the RGB image.")
            observation_id = f"corridor-02:camera_1:{keyframe.camera_sequence_index}"
            rgb_reference = ObservationReference(
                observation_id=observation_id,
                dataset_id="corridor-02",
                sequence_id="corridor-02",
                sensor_id="camera_1",
                sequence_index=keyframe.camera_sequence_index,
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
                frame_id=FrameId("cmu_rc1_velodyne"),
            )
            rgb = _rgb_frame(rgb_reference, raw_rgb, valid_mask)
            evidence, region_metadata = _region_evidence(visual_payload, valid_mask, keyframe.frame_id)
            regions.update(region_metadata)
            results = associate_map_points(
                map_points,
                rgb,
                calibration,
                map_to_camera,
                evidence,
                lidar_observation=lidar_reference,
                max_time_delta_ns=100_000_000,
            )
            for association in results:
                geometry_id = association.geometry.geometry_id
                if association.status is not AssociationStatus.ASSOCIATED:
                    current = rejected.get(geometry_id)
                    if current is None or _REJECTION_PRIORITY.index(
                        association.status
                    ) < _REJECTION_PRIORITY.index(current):
                        rejected[geometry_id] = association.status
                    continue
                associated.setdefault(geometry_id, []).append(
                    {
                        "observation_id": observation_id,
                        "observation_timestamp_ns": camera_timestamp_ns,
                        "pixel": association.pixel,
                        "color_rgb": association.color_rgb,
                        "region_id": association.region_id,
                        "label": association.label,
                        "lidar_observation_id": lidar_reference.observation_id,
                        "lidar_timestamp_ns": lidar_timestamp_ns,
                    }
                )
            raw_destination = assets / f"{keyframe.frame_id}-raw.png"
            overlay_destination = assets / f"{keyframe.frame_id}-regions.png"
            shutil.copyfile(keyframe.raw_image, raw_destination)
            shutil.copyfile(keyframe.overlay_image, overlay_destination)
            observations.append(
                {
                    "observation_id": observation_id,
                    "frame_id": str(camera_frame),
                    "visual_frame_id": keyframe.frame_id,
                    "timestamp_ns": camera_timestamp_ns,
                    "sensor_id": rgb_reference.sensor_id,
                    "camera_sequence_index": keyframe.camera_sequence_index,
                    "width": rgb.width,
                    "height": rgb.height,
                    "raw_image_uri": f"{assets.name}/{raw_destination.name}",
                    "overlay_image_uri": f"{assets.name}/{overlay_destination.name}",
                    "scene_claims": [
                        {
                            "kind": claim.get("kind"),
                            "value": claim.get("value"),
                            "confidence": claim.get("confidence"),
                            "support_state": (claim.get("support") or {}).get("state"),
                            "provenance": claim.get("provenance"),
                        }
                        for claim in visual_payload.get("scene_context", {}).get("claims", ())
                    ],
                    "visual_artifact": keyframe.visual_observation.resolve().as_uri(),
                    "lidar_observation_id": lidar_reference.observation_id,
                    "lidar_timestamp_ns": lidar_timestamp_ns,
                    "time_delta_ns": abs(camera_timestamp_ns - lidar_timestamp_ns),
                    "pose_source": pose_source,
                    "pose_timestamp_ns": pose_timestamp_ns,
                }
            )

    contextual_point_count = 0
    multi_observation_point_count = 0
    for point in base["points"]:
        geometry_id = str(point["geometry_id"])
        records = associated.get(geometry_id)
        if not records:
            status = rejected.get(geometry_id)
            if status is not None:
                point["context"] = {
                    "status": status.value,
                    "observations_considered": len(request.keyframes),
                }
            continue
        if len(records) > 1:
            multi_observation_point_count += 1
        contributions = [
            SemanticContribution(
                observation_id=record["observation_id"],
                timestamp_ns=record["observation_timestamp_ns"],
                region_id=str(record["region_id"]),
                label=str(record["label"]),
                confidence=(regions[record["region_id"]].get("confidence") or {}).get("value"),
                calibrated_confidence=regions[record["region_id"]].get("calibrated_confidence"),
                support_state=(regions[record["region_id"]].get("support") or {}).get("state"),
                visual_support=regions[record["region_id"]].get("visual_support"),
                region_quality=regions[record["region_id"]].get("region_quality"),
            )
            for record in records
            if record["label"] and record["region_id"]
        ]
        if contributions:
            fused = fuse_point_contributions(contributions)
            primary = next(
                record
                for record in records
                if record["observation_id"] == fused.observation_id
                and record["region_id"] == fused.region_id
            )
            contextual_point_count += 1
        else:
            fused = None
            primary = records[0]
        # O payload por ponto carrega apenas o que é próprio do ponto. Instante,
        # scan sincronizado e calibração pertencem à observação e ao artifact, e
        # repeti-los em cada um de centenas de milhares de pontos multiplicaria
        # o tamanho do arquivo sem acrescentar informação.
        context = {
            "status": AssociationStatus.ASSOCIATED.value,
            "observation_id": primary["observation_id"],
            "pixel": primary["pixel"],
            "color_rgb": primary["color_rgb"],
            "region_id": primary["region_id"],
            "label": primary["label"],
        }
        if fused is not None:
            context["confidence"] = fused.confidence
        if fused is not None and fused.contribution_count > 1:
            context["observation_count"] = fused.contribution_count
            context["agreement"] = fused.agreement
            context["contributions"] = [
                {
                    "observation_id": item.observation_id,
                    "region_id": item.region_id,
                    "label": item.label,
                    "confidence": item.confidence,
                }
                for item in fused.contributions
            ]
        point["context"] = context

    base["schema_version"] = 2
    base["artifact_type"] = "contextual_rgb_lidar_slice"
    base["calibration_id"] = calibration.calibration_id
    base["capabilities"] = {
        "geometric_map": "available",
        "rgb_association": "partial",
        "semantic_overlay": "partial_unverified",
        "evidence_inspection": "available",
        "temporal_fusion": "available" if len(request.keyframes) > 1 else "single_observation",
    }
    base["observations"] = observations
    base["regions"] = list(regions.values())
    base["context_summary"] = {
        "visual_observation_count": len(request.keyframes),
        "contextual_point_count": contextual_point_count,
        "rgb_point_count": len(associated),
        "unobserved_geometric_point_count": len(base["points"]) - len(associated),
        "multi_observation_point_count": multi_observation_point_count,
        "pose_source": pose_source,
        "claim_status": "predição VLM não verificada",
    }
    request.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = request.destination.with_suffix(request.destination.suffix + ".tmp")
    temporary.write_text(json.dumps(base, separators=(",", ":")), encoding="utf-8")
    temporary.replace(request.destination)
    return request.destination
