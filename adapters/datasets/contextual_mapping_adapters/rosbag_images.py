"""Inspeção e extração de streams de imagem armazenados em rosbags."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_MESSAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
_EXCLUDED_RGB_HINTS = ("depth", "infra", "ir")

#: Relógio dos timestamps registrados na extração: o instante de gravação da
#: mensagem no bag, o mesmo usado por ``bag_timestamp_ns`` nas janelas.
ROSBAG_CLOCK_ID = "rosbag"

#: Identidade do formato do registro de proveniência escrito ao lado de cada
#: frame. Só esta versão é lida.
_FRAME_PROVENANCE_SCHEMA = "rosbag-frame-provenance/1"


# Registra de onde um frame extraído veio. Existe porque o PNG sozinho não
# carrega timestamp nem posição no stream, e sem isso um consumidor acabava
# inventando valores: a validação de referência gravava o mesmo timestamp em
# todos os frames. É escrito pela extração e lido por read_frame_provenance.
@dataclass(frozen=True)
class ExtractedFrameProvenance:
    """Fatos da mensagem de origem de um frame extraído de uma rosbag.

    Argumentos:
        recording_id: identidade da gravação, derivada do nome do bag.
        topic: tópico ROS de onde a imagem foi lida.
        frame_id: ``header.frame_id`` da mensagem, ou ``None`` quando vazio.
        timestamp_ns: instante de gravação da mensagem no relógio ``rosbag``.
        sequence_index: posição da mensagem no stream do tópico.
    """

    recording_id: str
    topic: str
    frame_id: str | None
    timestamp_ns: int
    sequence_index: int

    # Valida os tipos e limites no momento em que o registro é criado ou lido,
    # para que um arquivo corrompido falhe na leitura e não mais adiante.
    def __post_init__(self) -> None:
        """Rejeita registros sem identidade ou com valores fora do domínio."""
        for field_name in ("recording_id", "topic"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string, got {value!r}.")
        if self.frame_id is not None and (not isinstance(self.frame_id, str) or not self.frame_id):
            raise ValueError(f"frame_id must be None or a non-empty string, got {self.frame_id!r}.")
        for field_name in ("timestamp_ns", "sequence_index"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}.")


# Localiza o registro de proveniência de um frame. Existe para que escrita e
# leitura concordem sobre o nome sem que consumidores repitam a convenção.
def frame_provenance_path(image_path: Path) -> Path:
    """Retorna o caminho do registro de proveniência de um frame.

    Argumentos:
        image_path: PNG extraído.
    Retorna:
        o arquivo JSON com o mesmo nome do frame.
    """
    return image_path.with_suffix(".json")


# Grava o registro de proveniência ao lado do frame. Chamada pelas duas
# funções de extração a cada PNG escrito.
def write_frame_provenance(image_path: Path, provenance: ExtractedFrameProvenance) -> Path:
    """Escreve o registro de proveniência de um frame extraído.

    Argumentos:
        image_path: PNG ao qual o registro pertence.
        provenance: fatos da mensagem de origem.
    Retorna:
        caminho do registro escrito.
    """
    destination = frame_provenance_path(image_path)
    record = {"schema": _FRAME_PROVENANCE_SCHEMA, **asdict(provenance)}
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


# Lê o registro de proveniência de um frame. É a fronteira pública pela qual
# consumidores (como a validação de referência de visual-perception) obtêm
# timestamp e posição reais do frame em vez de reconstruí-los.
def read_frame_provenance(image_path: Path) -> ExtractedFrameProvenance:
    """Lê e valida o registro de proveniência de um frame extraído.

    Argumentos:
        image_path: PNG extraído.
    Retorna:
        a proveniência registrada na extração.
    Levanta:
        FileNotFoundError: se o frame não tiver registro — frames extraídos
            antes deste registro existir precisam ser extraídos de novo.
        ValueError: se o registro tiver outro formato ou valores inválidos.
    """
    path = frame_provenance_path(image_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"frame provenance record missing: {path}. Re-extract the frames from the rosbag."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != _FRAME_PROVENANCE_SCHEMA:
        raise ValueError(f"{path} is not a {_FRAME_PROVENANCE_SCHEMA!r} record.")
    fields = ("recording_id", "topic", "frame_id", "timestamp_ns", "sequence_index")
    missing = [name for name in fields if name not in record]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}.")
    return ExtractedFrameProvenance(**{name: record[name] for name in fields})


# Lê o frame de coordenadas declarado no header de uma mensagem ROS. Local
# porque a forma da mensagem pertence ao runtime rosbags.
def _header_frame_id(message: object) -> str | None:
    """Retorna ``header.frame_id`` da mensagem, ou ``None`` quando ausente ou vazio."""
    frame_id = getattr(getattr(message, "header", None), "frame_id", None)
    return frame_id if isinstance(frame_id, str) and frame_id else None


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
            write_frame_provenance(
                destination,
                ExtractedFrameProvenance(
                    recording_id=bag_path.stem,
                    topic=topic,
                    frame_id=_header_frame_id(message),
                    timestamp_ns=timestamp,
                    sequence_index=sequence_index,
                ),
            )
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
        for index, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            if index % stride != 0 or len(written) >= count:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            pixels = decode_image_message(message, connection.msgtype)
            destination = output_dir / f"{frame_id_prefix or bag_path.stem}-{len(written):03d}.png"
            Image.fromarray(pixels).save(destination)
            write_frame_provenance(
                destination,
                ExtractedFrameProvenance(
                    recording_id=bag_path.stem,
                    topic=selected.name,
                    frame_id=_header_frame_id(message),
                    timestamp_ns=timestamp,
                    sequence_index=index,
                ),
            )
            written.append(destination)
    return written


# Extrai um lote resumível do stream inteiro: passo fixo e faixa de índices
# explícita, nomeando cada PNG pelo índice real no stream (não por um
# contador de frames escritos). Existe para processar um bag grande em
# lotes sucessivos sem reprocessar do zero: duas chamadas com o mesmo
# ``stride`` sobre o mesmo bag produzem os mesmos nomes de arquivo para os
# mesmos frames, então um consumidor pode checar o que já existe em disco e
# só extrair a faixa seguinte.
def extract_strided_rosbag_frames(
    bag_path: Path,
    output_dir: Path,
    *,
    stride: int,
    start_index: int = 0,
    end_index: int | None = None,
    topic: str | None = None,
    frame_id_prefix: str | None = None,
) -> list[Path]:
    """Extrai frames em passo fixo dentro de uma faixa de índices do stream.

    Argumentos:
        bag_path: rosbag de origem.
        output_dir: diretório dos PNGs.
        stride: intervalo entre frames extraídos (1 a cada ``stride``).
        start_index: primeiro índice elegível, inclusive.
        end_index: índice-limite, exclusivo; ``None`` para o fim do stream.
        topic: tópico explícito ou ``None`` para detecção.
        frame_id_prefix: prefixo de saída; usa o stem do bag quando ausente.
    Retorna:
        arquivos PNG escritos, nomeados pelo índice real no stream.
    Levanta:
        ValueError: se ``stride`` não for positivo ou a faixa for inválida.
    """
    from rosbags.highlevel import AnyReader

    if stride <= 0:
        raise ValueError("stride deve ser positivo")
    if start_index < 0 or (end_index is not None and end_index <= start_index):
        raise ValueError("faixa de índices inválida: start_index deve ser >= 0 e menor que end_index")
    recording = inspect_rosbag(bag_path)
    selected = select_rgb_topic(recording, topic)
    limit = selected.message_count if end_index is None else end_index
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with AnyReader([bag_path]) as reader:
        connections = [item for item in reader.connections if item.topic == selected.name]
        for index, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            if index < start_index:
                continue
            if index >= limit:
                break
            if index % stride != 0:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            pixels = decode_image_message(message, connection.msgtype)
            destination = output_dir / f"{frame_id_prefix or bag_path.stem}-{index:05d}.png"
            Image.fromarray(pixels).save(destination)
            write_frame_provenance(
                destination,
                ExtractedFrameProvenance(
                    recording_id=bag_path.stem,
                    topic=selected.name,
                    frame_id=_header_frame_id(message),
                    timestamp_ns=timestamp,
                    sequence_index=index,
                ),
            )
            written.append(destination)
    return written
