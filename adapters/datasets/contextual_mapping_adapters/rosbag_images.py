"""Inspeção e extração de streams de imagem armazenados em rosbags."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_MESSAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
_EXCLUDED_RGB_HINTS = ("depth", "infra", "ir")


# Descreve um tópico de imagem sem expor objetos do runtime rosbags. Existe
# para que aplicações possam mostrar opções e custos antes de abrir o stream.
@dataclass(frozen=True)
class RosbagImageTopic:
    """Metadados estáveis de um tópico de imagem em uma rosbag.

    Argumentos:
        name: nome completo do tópico ROS.
        message_type: tipo ROS das mensagens.
        message_count: quantidade de imagens gravadas.
    """

    name: str
    message_type: str
    message_count: int


# Resume uma rosbag para seleção interativa sem manter o arquivo aberto.
@dataclass(frozen=True)
class RosbagRecording:
    """Descrição de uma rosbag e de seus streams de imagem.

    Argumentos:
        path: caminho do arquivo inspecionado.
        duration_s: duração total no relógio de gravação.
        image_topics: tópicos de imagem disponíveis.
    """

    path: Path
    duration_s: float
    image_topics: tuple[RosbagImageTopic, ...]


# Inspeciona metadados baratos da rosbag para alimentar menus e validações.
# A função importa rosbags localmente para manter o adapter importável sem o
# extra opcional quando apenas seus contracts leves são usados.
def inspect_rosbag(path: Path) -> RosbagRecording:
    """Inspeciona duração e tópicos de imagem de uma rosbag.

    Argumentos:
        path: arquivo rosbag existente.
    Retorna:
        descrição fechada da gravação.
    Levanta:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se não houver tópico de imagem.
    """
    from rosbags.highlevel import AnyReader

    if not path.is_file():
        raise FileNotFoundError(f"rosbag não encontrada: {path}")
    with AnyReader([path]) as reader:
        topics = tuple(
            RosbagImageTopic(connection.topic, connection.msgtype, reader.topics[connection.topic].msgcount)
            for connection in reader.connections
            if connection.msgtype in IMAGE_MESSAGE_TYPES
        )
        unique = {item.name: item for item in topics}
        if not unique:
            raise ValueError(f"a rosbag não contém tópicos de imagem: {path}")
        return RosbagRecording(
            path=path.resolve(),
            duration_s=(reader.end_time - reader.start_time) / 1_000_000_000,
            image_topics=tuple(sorted(unique.values(), key=lambda item: item.name)),
        )


# Seleciona deterministicamente o stream RGB principal ou valida a escolha do
# usuário. Fica pura para que menus possam testá-la sem abrir uma rosbag real.
def select_rgb_topic(recording: RosbagRecording, requested: str | None = None) -> RosbagImageTopic:
    """Seleciona o tópico RGB de uma gravação.

    Argumentos:
        recording: metadados previamente inspecionados.
        requested: tópico explícito ou ``None`` para detecção automática.
    Retorna:
        tópico de imagem selecionado.
    Levanta:
        ValueError: se o tópico explícito não existir.
    """
    by_name = {item.name: item for item in recording.image_topics}
    if requested is not None:
        try:
            return by_name[requested]
        except KeyError as error:
            raise ValueError(f"o tópico pedido não é um stream de imagem: {requested}") from error
    candidates = [
        item
        for item in recording.image_topics
        if not any(hint in item.name.lower() for hint in _EXCLUDED_RGB_HINTS)
    ]
    return max(candidates or list(recording.image_topics), key=lambda item: item.message_count)


# Converte uma mensagem ROS de imagem em RGB uint8. Existe no adapter porque
# detalhes de encoding e compressão pertencem à integração com o dataset.
def decode_image_message(message: object, message_type: str) -> np.ndarray:
    """Decodifica uma mensagem ROS como array RGB ``(H, W, 3)``.

    Argumentos:
        message: mensagem desserializada pelo runtime rosbags.
        message_type: tipo ROS declarado pela conexão.
    Retorna:
        pixels RGB em ``uint8``.
    Levanta:
        ValueError: se o encoding da imagem não for suportado.
    """
    if message_type == "sensor_msgs/msg/CompressedImage":
        return np.array(Image.open(io.BytesIO(bytes(message.data))).convert("RGB"))  # type: ignore[attr-defined]

    height = int(message.height)  # type: ignore[attr-defined]
    width = int(message.width)  # type: ignore[attr-defined]
    encoding = str(message.encoding)  # type: ignore[attr-defined]
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)  # type: ignore[attr-defined]
    if encoding == "bgr8":
        return data.reshape(height, width, 3)[:, :, ::-1]
    if encoding == "rgb8":
        return data.reshape(height, width, 3)
    if encoding == "mono8":
        return np.repeat(data.reshape(height, width, 1), 3, axis=2)
    raise ValueError(f"encoding de imagem não suportado: {encoding!r}")


# Extrai mensagens identificadas pelo timestamp de gravação e preserva o índice
# original no nome. É usada pela CLI e pelo harness legado de visual-perception.
def extract_rosbag_frames(
    bag_path: Path,
    output_dir: Path,
    *,
    topic: str,
    frame_indices_by_timestamp: dict[int, int],
    frame_id_prefix: str,
) -> list[Path]:
    """Extrai frames específicos de uma rosbag como PNG.

    Argumentos:
        bag_path: rosbag de origem.
        output_dir: diretório dos PNGs.
        topic: tópico de imagem selecionado.
        frame_indices_by_timestamp: associação entre timestamp do bag e índice no stream.
        frame_id_prefix: prefixo estável usado nos nomes de frame.
    Retorna:
        arquivos escritos em ordem de timestamp.
    Levanta:
        ValueError: se o tópico não existir ou algum frame não for encontrado.
    """
    from rosbags.highlevel import AnyReader

    output_dir.mkdir(parents=True, exist_ok=True)
    pending = dict(frame_indices_by_timestamp)
    written: list[Path] = []
    with AnyReader([bag_path]) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        if not connections:
            raise ValueError(f"tópico {topic!r} ausente em {bag_path}")
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            sequence_index = pending.pop(timestamp, None)
            if sequence_index is None:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            pixels = decode_image_message(message, connection.msgtype)
            destination = output_dir / f"{frame_id_prefix}-{sequence_index:05d}.png"
            Image.fromarray(pixels).save(destination)
            written.append(destination)
            if not pending:
                break
    if pending:
        raise ValueError(f"{len(pending)} frames pedidos não foram encontrados em {bag_path}")
    return written


# Extrai uma amostra uniforme por contagem para preservar o comportamento do
# harness histórico, cuja seleção não usa um artifact BagWindow.
def extract_uniform_rosbag_frames(
    bag_path: Path,
    output_dir: Path,
    *,
    count: int,
    topic: str | None = None,
    frame_id_prefix: str | None = None,
) -> list[Path]:
    """Extrai uma quantidade uniforme de frames ao longo da gravação.

    Argumentos:
        bag_path: rosbag de origem.
        output_dir: diretório dos PNGs.
        count: quantidade máxima desejada.
        topic: tópico explícito ou ``None`` para detecção.
        frame_id_prefix: prefixo de saída; usa o stem do bag quando ausente.
    Retorna:
        arquivos PNG escritos.
    Levanta:
        ValueError: se ``count`` não for positivo.
    """
    from rosbags.highlevel import AnyReader

    if count <= 0:
        raise ValueError("count deve ser positivo")
    recording = inspect_rosbag(bag_path)
    selected = select_rgb_topic(recording, topic)
    stride = max(selected.message_count // count, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with AnyReader([bag_path]) as reader:
        connections = [item for item in reader.connections if item.topic == selected.name]
        for index, (connection, _, rawdata) in enumerate(reader.messages(connections=connections)):
            if index % stride != 0 or len(written) >= count:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            pixels = decode_image_message(message, connection.msgtype)
            destination = output_dir / f"{frame_id_prefix or bag_path.stem}-{len(written):03d}.png"
            Image.fromarray(pixels).save(destination)
            written.append(destination)
    return written
